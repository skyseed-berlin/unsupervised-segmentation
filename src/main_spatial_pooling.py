from src.models.spatial_pooling import SpatialPooling
import argparse
import cv2 as cv
import numpy as np 

def main(input, window_size, target_segment, output):

    """
        Takes pre-segmented input image and saves a greyscale output image reflecting
        the percentage of target segment (list or numpy array) in windows of specified window_size (in pixels).

        Example usage: main(input="path/to/input_image.png", target_segment=[5,5,5], output="path/to/output_image.png")
    """

    img = cv.imread(input)

    # for now use target_segment=[55, 68, 71] for example image

    # possible: pick segment with the lowest total channel value that is not pitch black
    # segments = np.unique(img.reshape(-1, img.shape[2]), axis=0)

    # darkness = np.sum(segments, axis=0)
    # target_segment = segments[np.where(darkness == np.amin(darkness[darkness>0])), :].flatten()

    pooled = SpatialPooling(img, target_segment=target_segment)

    result = pooled.fraction_of_target_segment(window_size=window_size) # this needs to be reduced to actual dimensions (remove overlaps)

    result_uint = np.array(result*255).astype('uint8')
    gray_image = cv.cvtColor(result_uint, cv.COLOR_GRAY2BGR)

    cv.imwrite(output, gray_image)



if __name__ == "__main__":

    # parse command-line arguments

    parser = argparse.ArgumentParser(description='Spatial Smoothing of a Segmented Image')

    parser.add_argument('--input', help='input image file name', required=True)
    parser.add_argument('--target_segment', help='channel values of target segment', required=True)
    parser.add_argument('--output', help='output image file name', required=True)
    parser.add_argument('--window_size_px', help='size of sliding window in pixel', required=False)
    parser.add_argument('--window_size_m', help='size of sliding window in m (gsd)', required=False)
    parser.add_argument('--gsd', help='ground sampling distance. resolution of aerial image (in sqm/pixel)', required=False)

    args = parser.parse_args()

    if args.window_size_px is not None:
        
        window_size = args.window_size_px

    elif args.window_size_m is not None and args.gsd is not None:
        
        window_size = round(float(args.window_size_m)/float(args.gsd))
    
    else:
        parser.error("either --window_size_px or --window_size_m and --gsd have to be set!")

    # run main fcn 

    main(input=args.input, window_size=window_size, target_segment=eval(args.target_segment), output=args.output)