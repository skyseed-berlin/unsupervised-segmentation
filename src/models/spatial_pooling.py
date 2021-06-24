import numpy as np 
from itertools import product
from src.helper.formatting import stack_image_windows

class SpatialPooling:

    def __init__(self, img, target_segment):

        self.img = img
        self.target_segment = target_segment

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
            
        try:
            perc_target_segment = target_segment_count/(window.shape[0]*window.shape[1] - ignore_count)

             # replace non-black pixels with target segment percentage and reduce to one value per pixel
            window_out = np.float16(window)
            window_out[~np.array(ignoremask), :] = [perc_target_segment]*window.shape[2]

            return window_out[:,:,0]
        
        # if all pixels in a window are black: return zero array
        except ZeroDivisionError:

            window_out = np.zeros((window.shape[0], window.shape[1]))

            return window_out

       
    def fraction_of_target_segment(self, window_size):

        """
            Returns a float32 numpy array with the fraction of target segment contained in a specific window.

            The method will result in one constant value for each window. 
        """

        self.window_size = window_size

        # zero-padding if applicable to avoid incomplete remainder windows at the edges
    
        padding_w = window_size - self.img.shape[0]%window_size 
        padding_h = window_size - self.img.shape[1]%window_size
        self.img_padded = np.pad(self.img, [(0, padding_w), (0, padding_h), (0, 0)], constant_values=0)            

        windows = np.lib.stride_tricks.sliding_window_view(self.img_padded, (self.window_size, self.window_size, self.img_padded.shape[2]))

        # only keep windows with stride=window_size
        indices = [i for i in list(product(np.arange(windows.shape[0]),np.arange(windows.shape[1]))) if i[0]%self.window_size == 0 and i[1]%self.window_size == 0]
        self.windows_flat = np.array([windows[i, j, 0, :, :, :] for i,j in indices])

        # this is necessary when skipping the above reduction
        # windows_flat = windows.reshape(-1, self.window_size, self.window_size, self.img.shape[2])

        # might be slow for large images! 
        result = np.array([self.perc_segment(window=self.windows_flat[i, :, :, :], target_segment=self.target_segment) for i in np.arange(self.windows_flat.shape[0])])

        # intermediate step required here once we allow for stride < window_size
        # reshape to original image size
        self.result = stack_image_windows(arr=result, shape=self.img_padded.shape)
        
        # cut padded margins
        self.result = self.result[:self.img.shape[0], :self.img.shape[1]]

        return self.result