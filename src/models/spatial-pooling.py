import numpy as np 
import os 
import cv2

PATH = os.path.expanduser("~/Documents/Skyseed/unsupervised-segmentation/results/kmeans/")

small_test_img = os.path.join(PATH, "9054717_5512099_segmented_k=3.png")
img = cv2.imread(os.path.join(small_test_img))

# set sliding window parameters
FLIGHT_HEIGHT = 9
TARGET_WINDOW_SIZE_IN_M = 2
# set later as hyperparameter - work with grid first (easier as it needs no averaging step)
# STRIDE = 2

TARGET_SEGMENT = [96,107,113]

gsd = FLIGHT_HEIGHT/18.9 # for our Drone with Phantom P4 Multispectral camera
window_size = round(TARGET_WINDOW_SIZE_IN_M/gsd)

# label segments > determine which one is the good one

windows = np.lib.stride_tricks.sliding_window_view(img, (window_size, window_size, img.shape[2]))
windows_flat = windows.reshape(-1, window_size, window_size, img.shape[2])
windows_flat.shape

class SpatialPooling:

    def __init__(self):

        self.input_image = None

def perc_segment(window, target_segment):

    """
        For a 3d numpy array (window) of arbitrary shape, return the number of occurences (pixels)
        that belong to a target segment, exluding pixels that have value zero for all bands as they
        are assumed to happen only at the margin.

        Note that length of target_segment and depth of window array have to match!
    """

    assert window.shape[2] == len(target_segment)

    # detect black pixels to be ignored
    ignoremask = np.all(window == [0]*window.shape[2], axis=2) 
    ignore_count = np.count_nonzero(ignoremask)

    # create indicator for target segment
    target_segment_count = np.count_nonzero(np.all(window == target_segment, axis = 2))
    
    perc_target_segment = target_segment_count/(window.shape[0]*window.shape[1] - ignore_count)

    # replace non-black pixels with target segment percentage and reduce to one value per pixel
    window_out = np.float16(window)
    window_out[~np.array(ignoremask), :] = [perc_target_segment]*window.shape[2]