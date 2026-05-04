import numpy as np
from pyhdf.SD import SD, SDC
import matplotlib.pyplot as plt

def detect_fire(lwir, swir):
    # pick key bands
    b20 = lwir[0, :, :]   # ~3.7 µm (fire sensitive)
    b31 = lwir[10, :, :]  # ~11 µm (background)
    b32 = lwir[11, :, :]  # ~12 µm (background)
    swir_band = swir[1, :, :] # ~1.6 µm (background)

    # normalize (important because raw values differ)
    b20_n = (b20 - np.mean(b20)) / np.std(b20)
    b31_n = (b31 - np.mean(b31)) / np.std(b31)
    b32_n = (b32 - np.mean(b32)) / np.std(b32)
    swir_norm = (swir_band - np.mean(swir_band)) / np.std(swir_band)

   # thermal contrast 
    thermal_anomaly  = b20_n - b31_n
    thermal_hot = thermal_anomaly > 2.5  # threshold for fire detection
    swir_hot = swir_norm > 2.5  # threshold for SWIR brightness
    # fire mask 
    fire_mask = thermal_hot & swir_hot
    print (f"Detected {np.sum(fire_mask)} fire pixels")
    return fire_mask
def main():
    # loading data
    file_path = "MOD021KM.A2026120.1405.061.2026120145559.NRT.hdf"

    hdf = SD(file_path, SDC.READ)
    lwir = hdf.select("EV_1KM_Emissive").get() # LW IR dataset
    swir = hdf.select("EV_500_Aggr1km_RefSB").get()  # SW IR dataset
    fire_mask = detect_fire(lwir, swir)
    plt.imshow(fire_mask, cmap="hot")
    plt.title("Detected Fire Pixels")
    plt.show()

if __name__ == "__main__":
    main()
