from PIL import Image
import argparse

parser = argparse.ArgumentParser(description='Converting .tif to .png file')
parser.add_argument('--input', metavar='FILENAME',
                    help='input image file name', required=True)
parser.add_argument('--output', metavar='FILENAME',
                    help='output image file name', required=True)
args = parser.parse_args()

im = Image.open(args.input)

if im.mode == "CMYK":
    im = im.convert("RGB")

im.save(args.output)