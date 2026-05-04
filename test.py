from pyhdf.SD import SD, SDC

file_path = "MOD021KM.A2026120.1405.061.2026120145559.NRT.hdf"

hdf = SD(file_path, SDC.READ)
# printing all the datasets in the file
"""
print("Datasets in file:")
datasets = hdf.datasets()

for name in datasets.keys():
    print(name) 
"""
# reading the LW and SW IR datasets
lw_ir = hdf.select("EV_1KM_Emissive").get() # LW IR dataset
sw_ir = hdf.select("EV_500_Aggr1km_RefSB").get()  # SW IR dataset
# printing the shapes of the datasets
# the shape output will be in format (band, image_height, image_width)
print("Shape of LW IR dataset:", lw_ir.shape)
print("Shape of SW IR dataset:", sw_ir.shape)