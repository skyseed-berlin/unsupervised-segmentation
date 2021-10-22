from src.models.kmeans import KMeansSegmentation
import cv2 
import rasterio
import argparse
import pathlib
import numpy 

def main(input_img, output, k, margins=True):

    file_type = pathlib.Path(input_img).suffix

    if file_type in [".jpg", ".png"]:
        img = cv2.imread(input_img)
    
    elif file_type == ".tif":
        raster = rasterio.open(input_img)
        img = numpy.transpose(raster.read())
        numpy.nan_to_num(img, copy=False, nan=0.0)
    
    else: 
        raise ValueError("Unsupported file type: " + file_type)
        
    km = KMeansSegmentation(image=img, K=k)

    # declare optimization parameters as top level at some point
    segmented = km.fit(max_iter=100, epsilon=0.001, attempts=10, margins=margins)

    reshaped = segmented.reshape((img.shape[0], img.shape[1], -1))

    if file_type == ".tif":
        transposed = cv2.rotate(cv2.flip(reshaped, 0), cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(filename=output, img=transposed)
    else:
        cv2.imwrite(filename=output, img=reshaped)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='K Means segmentation of an image of arbitrary depth')

    parser.add_argument('--input', help='input image file name', required=True)
    parser.add_argument('--k', help='integer specifying the number of segments', required=True)
    parser.add_argument('--output', help='output image file name', required=True)

    args = parser.parse_args()

    main(input_img=args.input, output=args.output, k=int(args.k))


