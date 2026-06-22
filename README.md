# TORCH Fire Detection Pipeline for NASA Space-to-Soil Challenge

Ground-based simulation of the TORCH onboard fire detection algorithm

TORCH (Thermal Optical Recognition and Communication Hardware) is a 3U CubeSat instrument designed to detect wildfires from low Earth orbit and deliver alerts in under 5 minutes via optical downlink. This repository simulates the onboard detection pipeline using real MODIS satellite data.

---

# Algorithm

The pipeline implements the two-stage detection approach described in the TORCH white paper:

# ** Stage 1 — Thermal infrared temporal differencing**
Each swath's Band 20 brightness temperature (3.9 µm) is subtracted pixel-by-pixel from the previous swath. Pixels that heated rapidly between passes are flagged as candidates. This rejects slowly-varying backgrounds like diurnal terrain heating and coastal thermal gradients, which would overwhelm a static threshold.

# ** Stage 2 — SWIR cross-modal confirmation (day) / tightened thermal-only gate (night)**
Candidate pixels are checked against Band 6 reflectance (1.64 µm). Both channels must confirm before a detection is recorded. Sun glint, industrial heat sources, and detector transients are hot in thermal IR but do not produce simultaneous SWIR fire radiative power signatures.

Band 6 is solar-reflective and carries no usable signal at night (white paper §3.3), so the pipeline detects day/night per swath from the SWIR channel's own variance — a flat, near-zero-variance scene means no sunlight reached the sensor. At night the SWIR gate is skipped, and the thermal-differencing threshold is raised (3.0σ → 4.5σ) to compensate for losing the second confirmation channel. The very first swath in a night sequence has no prior frame *and* no SWIR signal, so detection is skipped for that single frame rather than reported unreliably.

# Post-processing
Connected components smaller than 4 pixels are discarded as noise. Confirmed detections are packaged as event records containing a pixel bounding box, pixel count, and a fire radiative power proxy — mirroring the alert packet TORCH would transmit to the ground station via PULSE-A optical downlink.

The algorithm adapts NASA's MODIS/VIIRS contextual fire detection approach (Justice et al. 2002, Schroeder et al. 2014) for onboard execution rather than ground processing.

---

# Data

The pipeline uses **MODIS MOD021KM** (Terra Level 1B, 1 km resolution) HDF4 files. Each file is one swath — a ~2030 × 1354 pixel strip (~2330 km wide) captured during a single satellite pass.

Consecutive swaths over the same region, separated by ~5 minutes, form a temporal pair for differencing. The pipeline processes all `.hdf` files in the `data/` folder in sorted (chronological) order.

# Download real fire data

Requires a free NASA EarthData account: [urs.earthdata.nasa.gov](https://urs.earthdata.nasa.gov/users/new)

```bash
# Download MODIS swaths for the 2020 Creek Fire (default) → data/creek_fire/
python download_modis.py

# List all built-in fire events
python download_modis.py --list

# Download a specific named event → data/bootleg_fire/
python download_modis.py --event bootleg_fire

# Download for a custom location and date → data/custom/
python download_modis.py --lat 37.5 --lon -119.5 --date 2020-09-09
```

Each event downloads into its own subfolder so files from different fires are never mixed. Built-in events include the Creek Fire (2020), Dixie Fire (2021), Bootleg Fire (2021), Camp Fire (2018), and Rim Fire (2013) — all large California/Oregon fires with confirmed MODIS detections.

---

# Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy scipy pyhdf matplotlib earthaccess
```

---

# Usage

```bash
source venv/bin/activate

# Run on a downloaded event
python src/firedetection.py --data data/creek_fire
python src/firedetection.py --data data/bootleg_fire

# Skip the visualization (faster, just prints results)
python src/firedetection.py --data data/creek_fire --no-viz
```

The pipeline loads all `.hdf` files from the specified folder, runs detection, prints a summary, and displays a three-panel visualization per swath: brightness temperature, inter-frame delta, and confirmed fire pixels overlaid in orange.

**Example output (Creek Fire, September 9, 2020):**
```
Found 4 swath(s)

--- MOD021KM.A2020253.0525... ---   # 10:25 PM local, nighttime
  Method: skipped — no prior frame, night pass (insufficient confirmation channels)
  Channel mode: night (LWIR only)
  No confirmed fire event.

--- MOD021KM.A2020253.0530... ---   # 10:30 PM local, nighttime
  Method: temporal differencing
  Channel mode: night (LWIR only)   # SWIR inactive at night — tightened threshold used instead
  No confirmed fire event.

--- MOD021KM.A2020253.1905... ---   # 12:05 PM local, daytime
  Method: temporal differencing
  Channel mode: day (LWIR + SWIR)
  Clean fire pixels : 106
  Fire clusters     : 17
  EVENT PACKET — 106 px, bbox rows (58, 2006), FRP proxy 1190998

--- MOD021KM.A2020253.1910... ---   # 12:10 PM local, daytime
  Method: temporal differencing
  Channel mode: day (LWIR + SWIR)
  Clean fire pixels : 140
  Fire clusters     : 21
  EVENT PACKET — 140 px, bbox rows (316, 2019), FRP proxy 285140

===== SUMMARY =====
Swaths processed : 4
Events confirmed : 2
```

Note: on the Camp Fire event (2018-11-08), the same night-mode logic confirms a fire signature on a nighttime swath via temporal differencing alone — demonstrating that thermal-only detection can work at night when the fire is already established, even without SWIR confirmation.

---

## Repository Structure

```
wildfire_project/
├── src/
│   └── firedetection.py        # Detection pipeline
├── data/
│   ├── creek_fire/             # MODIS swaths per event (download separately)
│   ├── bootleg_fire/
│   └── ...
├── download_modis.py           # NASA EarthData downloader
└── README.md
```

---

## References
- TORCH White Paper, University of Chicago Space Program, May 2026.
