from src.models.spatial_pooling import SpatialPooling
import argparse
import cv2

# parse command-line arguments

parser = argparse.ArgumentParser(description='Spatial Smoothing of a Segmented Image')

parser.add_argument('--input', help='input image file name', required=True)
parser.add_argument('--target_segment', help='channel values of target segment', required=True)
parser.add_argument('--output', help='output image file name', required=True)

args = parser.parse_args()
img = cv2.imread(args.input)

pooled = SpatialPooling(img, flight_height=9, window_size_in_m=2, target_segment=args.target_segment)

result = pooled.fraction_of_target_segment() # this needs to be reduced to actual dimensions (remove overlaps)

# cv2.imwrite(args.output, result)

