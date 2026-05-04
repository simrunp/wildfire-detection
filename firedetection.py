"""
This is a prototype for single-swath fire detection using a combination of thermal anomaly and SWIR
"""
import numpy as np
from pyhdf.SD import SD, SDC
from scipy.ndimage import uniform_filter, label
import matplotlib.pyplot as plt
import os
# -------------------------------
# 0. Calibrate Radiance 
# -------------------------------
def calibrate_radiance(dataset, data):
    scales = dataset.attributes()['radiance_scales']
    offsets = dataset.attributes()['radiance_offsets']
    
    calibrated = np.zeros_like(data, dtype=float)
    
    for i in range(data.shape[0]):
        calibrated[i] = scales[i] * (data[i] - offsets[i])
    
    return calibrated
def calibrate_reflectance(dataset, data):
    scales = dataset.attributes()['reflectance_scales']
    offsets = dataset.attributes()['reflectance_offsets']
    
    calibrated = np.zeros_like(data, dtype=float)
    
    for i in range(data.shape[0]):
        calibrated[i] = scales[i] * (data[i] - offsets[i])
    
    return calibrated
# -------------------------------
# 1. LOAD MODIS BANDS
# -------------------------------
def load_modis(file_path):
    hdf = SD(file_path, SDC.READ)

    # LWIR emissive bands (contains thermal IR)
    lwir_ds = hdf.select("EV_1KM_Emissive")
    lwir_raw = lwir_ds.get() 
    lwir = calibrate_radiance(lwir_ds, lwir_raw)
    # SWIR / reflective bands
    swir_ds = hdf.select("EV_500_Aggr1km_RefSB")
    swir_raw = swir_ds.get()
    swir = calibrate_reflectance(swir_ds, swir_raw)

    b20 = lwir[0].astype(float)     # ~3.9 µm fire channel
    b31 = lwir[10].astype(float)    # ~11 µm background channel
    swir_band = swir[1].astype(float)    # ~1.64 µm (band 6, better fire band than band 5)

    return b20, b31, swir_band


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
    absolute_hot = t20 > 330.0      # K
    high_contrast = delta_t > 15.0  # K

    # (D) SWIR confirmation — local anomaly instead of global percentile
    swir_mean = uniform_filter(swir, size=window)
    swir_sq   = uniform_filter(swir**2, size=window)
    swir_std  = np.sqrt(np.clip(swir_sq - swir_mean**2, 0, None))
    swir_anomaly = swir > (swir_mean + 3 * swir_std)

    # -------------------------------
    # 5. COMBINE LOGIC
    # -------------------------------
    fire_mask = absolute_hot & thermal_anomaly & high_contrast & swir_anomaly

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

    data_folder = "data"
    
    # get all .hdf files only
    files = sorted([
        f for f in os.listdir(data_folder)
        if f.endswith(".hdf")
    ])

    print("Files found:", files)

    results = []

    for file in files:
        file_path = os.path.join(data_folder, file)

        print("\n==============================")
        print(f"Processing: {file}")
        print("==============================")

        try:
            fire_mask, thermal = detect_fire(file_path)

            fire_pixels = int(np.sum(fire_mask))

            results.append({
                "file": file,
                "fire_pixels": fire_pixels
            })

        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue

    # -------------------------------
    # SUMMARY
    # -------------------------------
    print("\n===== FINAL RESULTS =====")
    for r in results:
        print(f"{r['file']}: {r['fire_pixels']} fire pixels")