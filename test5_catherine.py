"""
This is a prototype for single-swath fire detection using a combination of thermal anomaly and SWIR
"""
import numpy as np
from pyhdf.SD import SD, SDC
from scipy.ndimage import uniform_filter, label
import matplotlib.pyplot as plt


# -------------------------------
# 1. LOAD MODIS BANDS
# -------------------------------
def load_modis(file_path):
    hdf = SD(file_path, SDC.READ)

    # LWIR emissive bands (contains thermal IR)
    lwir = hdf.select("EV_1KM_Emissive").get()

    # SWIR / reflective bands
    swir = hdf.select("EV_500_Aggr1km_RefSB").get()

    b20 = lwir[0].astype(float)     # ~3.9 µm fire channel
    b31 = lwir[10].astype(float)    # ~11 µm background channel
    swir = swir[1].astype(float)    # ~1.64 µm (band 6, better fire band than band 5)

    return b20, b31, swir


# -------------------------------
# 2. BRIGHTNESS TEMPERATURE (Planck inversion)
# -------------------------------
# MODIS calibration constants
C1 = 1.191042e8   # µW / (m² · sr · cm⁻⁴)
C2 = 1.4387752e4  # µm · K
VC_B20 = 2557.45  # central wavenumber for B20 (cm⁻¹)
VC_B31 = 924.28   # central wavenumber for B31 (cm⁻¹)

def radiance_to_bt(radiance, vc):
    radiance = np.where(radiance <= 0, np.nan, radiance)
    return (C2 * vc) / np.log((C1 * vc**3 / radiance) + 1.0)


# -------------------------------
# 3. FIRE DETECTION (SINGLE SWATH)
# -------------------------------
def detect_fire(file_path):

    b20, b31, swir = load_modis(file_path)

    print("Loaded swath shape:", b20.shape)

    # Convert to brightness temperature (Kelvin)
    t20 = radiance_to_bt(b20, VC_B20)
    t31 = radiance_to_bt(b31, VC_B31)

    # -------------------------------
    # 4. CORE FIRE SIGNALS
    # -------------------------------

    # (A) Thermal contrast
    delta_t = t20 - t31

    # (B) Contextual background model (local stats)
    window = 31

    local_mean = uniform_filter(t20, size=window)
    local_sq = uniform_filter(t20**2, size=window)
    local_std = np.sqrt(np.clip(local_sq - local_mean**2, 0, None))

    thermal_anomaly = t20 > (local_mean + 3 * local_std)

    # (C) Absolute threshold — physically meaningful now that we have real Kelvin
    absolute_hot = t20 > 310.0      # K
    high_contrast = delta_t > 10.0  # K

    # (D) SWIR confirmation — local anomaly instead of global percentile
    swir_mean = uniform_filter(swir, size=window)
    swir_sq   = uniform_filter(swir**2, size=window)
    swir_std  = np.sqrt(np.clip(swir_sq - swir_mean**2, 0, None))
    swir_anomaly = swir > (swir_mean + 3 * swir_std)

    # -------------------------------
    # 5. COMBINE LOGIC
    # -------------------------------
    fire_mask = (absolute_hot | thermal_anomaly) & high_contrast & swir_anomaly

    # -------------------------------
    # 6. REMOVE NOISE (CONNECTED COMPONENTS)
    # -------------------------------
    labeled, num = label(fire_mask)

    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    min_size = 34  
    valid_labels = np.where(sizes >= min_size)[0]

    clean_mask = np.isin(labeled, valid_labels)

    # -------------------------------
    # OUTPUT STATS
    # -------------------------------
    print("Raw fire pixels:", np.sum(fire_mask))
    print("Clean fire pixels:", np.sum(clean_mask))
    print("Detected fire regions:", num)

    return clean_mask, t20


# -------------------------------
# 7. RUN + VISUALIZE
# -------------------------------
if __name__ == "__main__":

    file_path = "MOD021KM.A2026119.2315.061.2026119235638.NRT.hdf"

    fire_mask, thermal = detect_fire(file_path)

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.imshow(thermal, cmap="inferno")
    plt.title("Brightness Temperature B20 (K)")

    plt.subplot(1,2,2)
    plt.imshow(fire_mask, cmap="hot")
    plt.title("Detected Fire Mask")

    plt.tight_layout()
    plt.show()