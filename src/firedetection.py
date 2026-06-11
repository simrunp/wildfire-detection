"""
TORCH/SPARK fire detection pipeline — ground simulation using MODIS MOD021KM data.

Implements the two-stage algorithm from the TORCH white paper:
  1. LWIR temporal differencing: flag pixels that heated rapidly between passes
  2. SWIR confirmation gate: verify flagged pixels show fire radiative power at 1.64 µm
  3. Event packet output: bounding box, FRP proxy, pixel count

Reference: TORCH White Paper, University of Chicago Space Program, May 2026.
"""

import os
import numpy as np
from pyhdf.SD import SD, SDC
from scipy.ndimage import uniform_filter, label
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Planck inversion constants for converting MODIS radiance → brightness temp.
# These are fixed physical constants; vc values are per-band from the MODIS
# calibration spec (Wan & Li 1997).
# ---------------------------------------------------------------------------
C1 = 1.191042e8   # µW / (m² · sr · cm⁻⁴)
C2 = 1.4387752e4  # µm · K
VC_B20 = 2557.45  # Band 20 central wavenumber, ~3.9 µm — primary fire channel
VC_B31 = 924.28   # Band 31 central wavenumber, ~11 µm — ambient background channel

# ---------------------------------------------------------------------------
# Detection thresholds — all adjustable here without touching algorithm logic.
# SPATIAL_WINDOW: size of the local neighborhood used to compute background stats.
# DELTA_SIGMA / SWIR_SIGMA: how many std devs above local background triggers a flag.
# ABS_TEMP_K: absolute floor — pixels below this can't be fire regardless of delta.
# DELTA_T_MIN: minimum B20-B31 contrast — fire pixels are much hotter in B20 than B31.
# MIN_CLUSTER_SIZE: isolated single pixels are noise; require a minimum connected region.
# ---------------------------------------------------------------------------
SPATIAL_WINDOW   = 21
DELTA_SIGMA      = 3.0
SWIR_SIGMA       = 2.5
ABS_TEMP_K       = 320.0
DELTA_T_MIN      = 10.0
MIN_CLUSTER_SIZE = 4


# ---------------------------------------------------------------------------
# Step 1: Load and calibrate raw MODIS HDF data.
#
# MODIS stores raw integer counts in HDF4 files. Each dataset ships with
# per-band scale factors and offsets in its metadata. We apply those here to
# get physical radiance (W/m²/sr/µm) and reflectance (unitless 0–1).
# Then we invert the Planck function to get brightness temperature in Kelvin.
# ---------------------------------------------------------------------------

def _calibrate_radiance(dataset, raw):
    # Converts raw digital counts → spectral radiance using per-band scale/offset
    scales  = dataset.attributes()["radiance_scales"]
    offsets = dataset.attributes()["radiance_offsets"]
    out = np.zeros_like(raw, dtype=np.float32)
    for i in range(raw.shape[0]):
        out[i] = scales[i] * (raw[i].astype(np.float32) - offsets[i])
    return out


def _calibrate_reflectance(dataset, raw):
    # Same as above but for reflective (SWIR) bands — outputs 0–1 reflectance
    scales  = dataset.attributes()["reflectance_scales"]
    offsets = dataset.attributes()["reflectance_offsets"]
    out = np.zeros_like(raw, dtype=np.float32)
    for i in range(raw.shape[0]):
        out[i] = scales[i] * (raw[i].astype(np.float32) - offsets[i])
    return out


def _radiance_to_bt(radiance, vc):
    # Planck inversion: L(T) = C1*vc³ / (exp(C2*vc/T) - 1), solved for T
    r = np.where(radiance <= 0, np.nan, radiance)
    return (C2 * vc) / np.log((C1 * vc**3 / r) + 1.0)


def load_modis(file_path):
    """Load a MOD021KM HDF file and return calibrated brightness temps and SWIR reflectance."""
    hdf = SD(file_path, SDC.READ)

    lwir_ds  = hdf.select("EV_1KM_Emissive")
    lwir_cal = _calibrate_radiance(lwir_ds, lwir_ds.get())

    swir_ds  = hdf.select("EV_500_Aggr1km_RefSB")
    swir_cal = _calibrate_reflectance(swir_ds, swir_ds.get())

    # Band 20 (~3.9 µm): saturates over flames, giving high contrast vs cool ground
    # Band 31 (~11 µm): tracks ambient surface temperature — used as background reference
    # Band 6  (~1.64 µm): SWIR fire radiative power band, same as VIIRS I3/M10
    t20  = _radiance_to_bt(lwir_cal[0],  VC_B20)
    t31  = _radiance_to_bt(lwir_cal[10], VC_B31)
    swir = swir_cal[1]

    return t20, t31, swir


# ---------------------------------------------------------------------------
# Step 2a: Spatially-adaptive anomaly detection (shared by both modes).
#
# Rather than a fixed global threshold, we compute per-pixel local statistics
# in a sliding window. A pixel is anomalous if it exceeds its neighborhood
# mean by more than `sigma` standard deviations. This adapts automatically to
# hot deserts, coastlines, and other warm-but-uniform backgrounds.
# ---------------------------------------------------------------------------

def _spatial_anomaly(band, window, sigma):
    local_mean = uniform_filter(band, size=window)
    local_sq   = uniform_filter(band ** 2, size=window)
    local_std  = np.sqrt(np.clip(local_sq - local_mean ** 2, 0, None))
    return band > (local_mean + sigma * local_std)


# ---------------------------------------------------------------------------
# Step 2b: Single-frame fallback (used only for the first swath).
#
# When there is no previous frame to difference against, we fall back to
# purely spatial contextual detection on the raw brightness temperature.
# Less sensitive than temporal mode — more likely to flag warm terrain.
# ---------------------------------------------------------------------------

def detect_single(t20, t31, swir):
    """Spatial-only detection: LWIR contextual anomaly gated by SWIR confirmation."""
    delta_t = t20 - t31

    # LWIR candidate: statistically anomalous, above absolute floor, with fire-like B20/B31 contrast
    lwir_candidate = (
        _spatial_anomaly(t20, SPATIAL_WINDOW, DELTA_SIGMA) &
        (t20 > ABS_TEMP_K) &
        (delta_t > DELTA_T_MIN)
    )

    # SWIR gate: only check SWIR pixels where LWIR already raised a flag
    swir_confirmed = np.zeros_like(lwir_candidate)
    if np.any(lwir_candidate):
        swir_anomaly   = _spatial_anomaly(swir, SPATIAL_WINDOW, SWIR_SIGMA)
        swir_confirmed = lwir_candidate & swir_anomaly

    return swir_confirmed


# ---------------------------------------------------------------------------
# Step 2c: Temporal differencing — primary algorithm (white paper §2.1).
#
# Subtracts the previous swath's B20 temperature from the current one.
# Pixels that are slowly warming (diurnal heating, solar loading on terrain)
# produce small deltas and are ignored. A pixel that jumped by >3σ in a single
# ~5-minute inter-swath interval is characteristic of ignition.
#
# SWIR is only queried after LWIR raises a candidate — this is the two-stage
# confirmation gate from the white paper. Industrial heat sources and sun glint
# are hot in LWIR but do not produce simultaneous SWIR FRP signatures.
# ---------------------------------------------------------------------------

def detect_temporal(t20_curr, t31_curr, swir_curr, t20_prev):
    """Inter-frame delta detection with SWIR cross-modal validation."""

    # Inter-frame thermal delta: how much did each pixel heat up since last pass?
    delta = t20_curr - t20_prev

    # Flag pixels where the delta is anomalously large relative to local neighborhood
    lwir_candidate = (
        _spatial_anomaly(delta, SPATIAL_WINDOW, DELTA_SIGMA) &
        (delta > 0)  # warming only — cooling anomalies are not fire
    )

    # Absolute physical guards: must be genuinely hot and show fire-like spectral contrast
    delta_t        = t20_curr - t31_curr
    lwir_candidate = lwir_candidate & (t20_curr > ABS_TEMP_K) & (delta_t > DELTA_T_MIN)

    # SWIR confirmation gate: only evaluate where LWIR already flagged a candidate
    swir_confirmed = np.zeros_like(lwir_candidate)
    if np.any(lwir_candidate):
        swir_anomaly   = _spatial_anomaly(swir_curr, SPATIAL_WINDOW, SWIR_SIGMA)
        swir_confirmed = lwir_candidate & swir_anomaly

    return swir_confirmed, delta


# ---------------------------------------------------------------------------
# Step 3: Post-processing — cluster filter and event packet assembly.
#
# Real fires cover multiple contiguous pixels at 1 km resolution.
# Single isolated flagged pixels are almost always noise (detector transients,
# cosmic rays, sub-pixel calibration artifacts). We label connected components
# and discard any region smaller than MIN_CLUSTER_SIZE pixels.
#
# The event packet format mirrors what TORCH would transmit: bounding box
# (for geolocation), pixel count (fire size proxy), and a fire radiative power
# proxy (relative intensity). On the actual satellite this packet triggers the
# optical downlink via PULSE-A.
# ---------------------------------------------------------------------------

def _cluster_filter(fire_mask, min_size):
    """Remove connected components smaller than min_size pixels (noise rejection)."""
    labeled, num_regions = label(fire_mask)
    if num_regions == 0:
        return fire_mask, 0
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    valid = np.where(sizes >= min_size)[0]
    return np.isin(labeled, valid), len(valid)


def _fire_radiative_power_estimate(t20, fire_mask):
    # FRP proxy: sum of excess brightness temp above scene median over fire pixels.
    # Not in physical watts — use as a relative intensity ranking between events.
    if not np.any(fire_mask):
        return 0.0
    background = np.nanmedian(t20[~fire_mask])
    excess = np.where(fire_mask, t20 - background, 0.0)
    return float(np.sum(np.clip(excess, 0, None)))


def build_event_packet(fire_mask, t20, filename):
    """Assemble a confirmed-event output record (bounding box, pixel count, FRP proxy)."""
    if not np.any(fire_mask):
        return None
    rows, cols = np.where(fire_mask)
    return {
        "file":        filename,
        "pixel_count": int(np.sum(fire_mask)),
        "bbox_rows":   (int(rows.min()), int(rows.max())),
        "bbox_cols":   (int(cols.min()), int(cols.max())),
        "frp_proxy":   _fire_radiative_power_estimate(t20, fire_mask),
    }


# ---------------------------------------------------------------------------
# Pipeline entry point — processes all HDF files in a folder in sorted order.
# ---------------------------------------------------------------------------

def run_pipeline(data_folder="data", visualize=True):
    """
    Run the TORCH detection pipeline over all .hdf files in data_folder.

    Consecutive same-shape swaths use temporal differencing.
    The first swath (no prior frame) uses single-frame spatial fallback.
    """
    files = sorted([f for f in os.listdir(data_folder) if f.endswith(".hdf")])
    if not files:
        print("No .hdf files found in", data_folder)
        return []

    print(f"Found {len(files)} swath(s): {files}\n")

    frames = []
    for f in files:
        try:
            t20, t31, swir = load_modis(os.path.join(data_folder, f))
            frames.append((f, t20, t31, swir))
            print(f"Loaded {f}  shape={t20.shape}")
        except Exception as e:
            print(f"Failed to load {f}: {e}")

    print()
    event_packets = []

    for i, (fname, t20, t31, swir) in enumerate(frames):
        print(f"--- {fname} ---")

        if i > 0 and frames[i-1][1].shape == t20.shape:
            raw_mask, delta = detect_temporal(t20, t31, swir, frames[i-1][1])
            method = "temporal differencing"
        else:
            raw_mask = detect_single(t20, t31, swir)
            delta    = None
            method   = "single-frame spatial (no prior frame)"

        clean_mask, n_clusters = _cluster_filter(raw_mask, MIN_CLUSTER_SIZE)

        print(f"  Method            : {method}")
        print(f"  Raw fire pixels   : {int(np.sum(raw_mask))}")
        print(f"  Clean fire pixels : {int(np.sum(clean_mask))}")
        print(f"  Fire clusters     : {n_clusters}")

        packet = build_event_packet(clean_mask, t20, fname)
        if packet:
            print(f"  EVENT PACKET:")
            print(f"    Pixel count : {packet['pixel_count']}")
            print(f"    Bounding box: rows {packet['bbox_rows']}, cols {packet['bbox_cols']}")
            print(f"    FRP proxy   : {packet['frp_proxy']:.1f}")
            event_packets.append(packet)
        else:
            print("  No confirmed fire event.")
        print()

        if visualize:
            _plot_result(fname, t20, clean_mask, delta)

    print("===== SUMMARY =====")
    print(f"Swaths processed : {len(frames)}")
    print(f"Events confirmed : {len(event_packets)}")
    for p in event_packets:
        print(f"  {p['file']}: {p['pixel_count']} px, FRP proxy {p['frp_proxy']:.1f}")

    return event_packets


# ---------------------------------------------------------------------------
# Visualization — three panels: brightness temp, inter-frame delta, fire mask
# ---------------------------------------------------------------------------

def _plot_result(title, t20, fire_mask, delta=None):
    ncols = 3 if delta is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))

    axes[0].imshow(t20, cmap="inferno")
    axes[0].set_title("B20 Brightness Temp (K)")

    if delta is not None:
        axes[1].imshow(delta, cmap="RdBu_r", vmin=-10, vmax=10)
        axes[1].set_title("Inter-frame ΔT (K)")
        ax_fire = axes[2]
    else:
        ax_fire = axes[1]

    ax_fire.imshow(t20, cmap="gray")
    if np.any(fire_mask):
        overlay = np.zeros((*fire_mask.shape, 4), dtype=float)
        overlay[fire_mask] = [1, 0.2, 0, 0.8]
        ax_fire.imshow(overlay)
    ax_fire.set_title(f"Fire detections ({int(np.sum(fire_mask))} px)")

    fig.suptitle(title, fontsize=9)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_pipeline(data_folder="data", visualize=True)
