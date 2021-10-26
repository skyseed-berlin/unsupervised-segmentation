import rasterio
import numpy as np 

def main(file_list, path_out):

    with rasterio.open(file_list[0]) as src0:
        meta = src0.meta

    meta.update(count = len(file_list))

    with rasterio.open(path_out, "w", **meta) as dst:
    
        for id, layer in enumerate(file_list, start=1):
            print(id, layer)
            with rasterio.open(layer) as src1:
                band = src1.read(1)
                print(band[~np.isnan(band)].max())
                dst.write_band(id, src1.read(1))


