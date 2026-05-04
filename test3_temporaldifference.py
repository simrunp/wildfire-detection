"""
This uses an image from above Egypt from 4/30 and then 5/1 to show temporal 
differencing. Though, it still detects fire pixels because of the large change in 
time, so I don't think we can do temporal changes cause we don't have a new photo 
of the same region every 5 minutes. 
"""
import numpy as np
from pyhdf.SD import SD, SDC
from scipy.ndimage import uniform_filter, label
import matplotlib.pyplot as plt

# -------------------------------
# LOAD DATA FUNCTION
# -------------------------------
def load_bands(file_path):
    hdf = SD(file_path, SDC.READ)

    # datasets
    lwir = hdf.select("EV_1KM_Emissive").get()
    swir = hdf.select("EV_500_Aggr1km_RefSB").get()

    # select bands
    b20 = lwir[0].astype(float)     # ~3.7 µm (fire sensitive)
    b31 = lwir[10].astype(float)    # ~11 µm (background)
    swir_band = swir[0].astype(float)  # ~1.6 µm

    return b20, b31, swir_band


# -------------------------------
# FIRE DETECTION PIPELINE
# -------------------------------
def detect_fire(file_t, file_t1):
    # load two consecutive frames
    b20_t, b31_t, swir_t = load_bands(file_t)
    b20_t1, b31_t1, swir_t1 = load_bands(file_t1)

    print("Loaded data shapes:", b20_t.shape)

    # -------------------------------
    # 1. TEMPORAL DIFFERENCING
    # -------------------------------
    delta_lwir = b20_t - b20_t1
    delta_swir = swir_t - swir_t1

    # -------------------------------
    # 2. SPATIALLY-ADAPTIVE THRESHOLD
    # -------------------------------
    window_size = 31

    local_mean = uniform_filter(delta_lwir, size=window_size)
    local_sq_mean = uniform_filter(delta_lwir**2, size=window_size)
    local_std = np.sqrt(local_sq_mean - local_mean**2)

    candidate = delta_lwir > (local_mean + 3 * local_std)

    # -------------------------------
    # 3. THERMAL CONTRAST (PHYSICS)
    # -------------------------------
    delta_T = b20_t - b31_t

    thermal_condition = delta_T > np.percentile(delta_T, 99)

    # -------------------------------
    # 4. SWIR VALIDATION
    # -------------------------------
    swir_condition = delta_swir > np.percentile(delta_swir, 99)

    # -------------------------------
    # 5. COMBINE CONDITIONS
    # -------------------------------
    fire_mask = candidate & thermal_condition & swir_condition

    # -------------------------------
    # 6. REMOVE NOISE (CLUSTER FILTER)
    # -------------------------------
    labeled, num_features = label(fire_mask)

    sizes = np.bincount(labeled.ravel())

    min_pixels = 8  # tune this
    clean_mask = sizes[labeled] > min_pixels

    print("Detected fire pixels:", np.sum(clean_mask))
    print("Detected regions:", num_features)

    return clean_mask, delta_lwir


# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":

    # IMPORTANT: use TWO consecutive MODIS files
    file_t = "MOD021KM.A2026121.1845.061.2026121193026.NRT.hdf"
    file_t1 = "MOD021KM.A2026122.0655.061.2026122080341.NRT.hdf"

    fire_mask, delta = detect_fire(file_t, file_t1)

    # -------------------------------
    # VISUALIZATION
    # -------------------------------
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(delta, cmap="hot")
    plt.title("LWIR Frame Difference")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(fire_mask, cmap="hot")
    plt.title("Detected Fires")
    plt.colorbar()

    plt.tight_layout()
    plt.show()