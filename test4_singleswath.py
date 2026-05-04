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

    # -------------------------------
    # Band selection (MODIS convention)
    # -------------------------------
    # Band indexing varies slightly by product, but typically:
    # B20/B21 ~ 3.7–3.9 µm (fire sensitive)
    # B31     ~ 11 µm (background thermal)
    # SWIR    ~ 1.6 µm band

    b20 = lwir[0].astype(float)     # ~3.9 µm fire channel (proxy)
    b31 = lwir[10].astype(float)    # ~11 µm background channel
    swir = swir[0].astype(float) # ~1.6 µm reflectance

    return b20, b31, swir


# -------------------------------
# 2. BRIGHTNESS TEMPERATURE (simplified proxy)
# -------------------------------
def radiance_to_bt_proxy(band):
    """
    Full Planck conversion requires calibration constants.
    For prototype + anomaly detection, we use normalized radiance proxy.
    """
    band = band.astype(float)

    # avoid division issues
    band = np.nan_to_num(band)

    return (band - np.mean(band)) / (np.std(band) + 1e-6)


# -------------------------------
# 3. FIRE DETECTION (SINGLE SWATH)
# -------------------------------
def detect_fire(file_path):

    # -------------------------------
    # Load data
    # -------------------------------
    b20, b31, swir = load_modis(file_path)

    print("Loaded swath shape:", b20.shape)

    # -------------------------------
    # Convert to normalized "temperature-like" space
    # -------------------------------
    t20 = radiance_to_bt_proxy(b20)
    t31 = radiance_to_bt_proxy(b31)
    s = radiance_to_bt_proxy(swir)

    # -------------------------------
    # 4. CORE FIRE SIGNALS
    # -------------------------------

    # (A) Thermal anomaly (hotspot detection)
    delta_t = t20 - t31

    # (B) Contextual background model (local stats)
    window = 31

    local_mean = uniform_filter(t20, size=window)
    local_sq = uniform_filter(t20**2, size=window)
    local_std = np.sqrt(local_sq - local_mean**2)

    thermal_anomaly = t20 > (local_mean + 3 * local_std)

    # (C) Absolute hotspot reinforcement (MODIS-style idea)
    absolute_hot = delta_t > np.percentile(delta_t, 99)

    # (D) SWIR confirmation (fire radiative signature)
    swir_anomaly = s > np.percentile(s, 99)

    # -------------------------------
    # 5. COMBINE LOGIC (KEY STEP)
    # -------------------------------
    fire_mask = thermal_anomaly & absolute_hot & swir_anomaly

    # -------------------------------
    # 6. REMOVE NOISE (CONNECTED COMPONENTS)
    # -------------------------------
    labeled, num = label(fire_mask)

    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0  # background label
    min_size = 10
    valid_labels = np.where(sizes > min_size)[0]  # keep only clusters larger than 10 pixels  

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
# NO fire detected in this image! 
if __name__ == "__main__":

    file_path = "MOD021KM.A2026121.1845.061.2026121193026.NRT.hdf"

    fire_mask, thermal = detect_fire(file_path)

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.imshow(thermal, cmap="inferno")
    plt.title("Normalized Thermal Signal")

    plt.subplot(1,2,2)
    plt.imshow(fire_mask, cmap="hot")
    plt.title("Detected Fire Mask")

    plt.tight_layout()
    plt.show()