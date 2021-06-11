import numpy as np 
import os 
import cv2

class SpatialPooling:

    def __init__(self, img, flight_height, window_size_in_m, target_segment):

        self.img = img
        self.target_segment = target_segment

        self.gsd = flight_height/18.9
        self.window_size = round(window_size_in_m/self.gsd)

    @staticmethod
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

        return window_out[:,:,0]


    def fraction_of_target_segment(self):

        """
            Returns a float32 numpy array with the fraction of target segment contained in a specific window.

            The method will result in one constant value for each window. 
        """

        windows = np.lib.stride_tricks.sliding_window_view(self.img, (self.window_size, self.window_size, self.img.shape[2]))
        windows_flat = windows.reshape(-1, self.window_size, self.window_size, self.img.shape[2])

        result = np.apply_along_axis(lambda x: self.perc_segment(window=x, target_segment=self.target_segment), 0, windows_flat)
        self.result = result

        return result