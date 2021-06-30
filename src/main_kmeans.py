from src.models.kmeans import KMeansSegmentation
import cv2 as cv
import argparse

def main(input_img, output, k, margins=True):

    img = cv.imread(input_img)

    km = KMeansSegmentation(image=img, K=k)

    # declare optimization parameters as top level at some point
    segmented = km.fit(max_iter=100, epsilon=0.001, attempts=10, margins=margins)

    cv.imwrite(filename=output, img=segmented)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='K Means segmentation of an image of arbitrary depth')

    parser.add_argument('--input', help='input image file name', required=True)
    parser.add_argument('--k', help='integer specifying the number of segments', required=True)
    parser.add_argument('--output', help='output image file name', required=True)

    args = parser.parse_args()

    main(input_img=args.input, output=args.output, k=int(args.k))


