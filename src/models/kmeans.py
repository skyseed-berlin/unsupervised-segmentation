# Perform k-means segmentation on an input image represented by a 2d numpy array with any number of color bands

import numpy as np 
import cv2
import matplotlib.pyplot as plt

class KMeansSegmentation:

    def __init__(self, image, K):

        self._image = image
        self._K = K
        self.flat_image = np.float32(image.reshape((-1, 3)))
      
    def clean_margins(self):

        """
            Produce a numpy array excluding all black pixels (at margins) and save as attribute flat_image_cleaned
        """

        rowmask = np.all(self.flat_image == [0]*self.flat_image.shape[1], axis = 1)

        self.flat_image_cleaned = self.flat_image[~np.array(rowmask), :]

        return self.flat_image_cleaned 

    def plot_hist(self, band, cleaned = False):

        """
            Produce cleaned and uncleaned color histograms of specified color band indicated by parameter band (integer)
            
            Note that opencv uses BGR order instead of RGB for the first three color bands.
        """

        if cleaned:

            try:
                img = self.flat_image_cleaned[:, band]
            except AttributeError:

                img = self.clean_margins()[:, band]
        
        else:
            img = self.flat_image[:, band]

        plt.hist(img[:, band])
        # can also save this somewhere instead
        plt.show()

    def fit(self, max_iter, epsilon, attempts):

        self._optimization = {"max_iter": max_iter, "epsilon": epsilon, "attempts": attempts}

        try: 
            img = self.flat_image_cleaned
        except AttributeError:

            img = self.clean_margins()

        result = cv2.kmeans(img, self._K, None,
                            criteria=((cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER), max_iter, epsilon),
                            attempts=attempts,
                            flags=cv2.KMEANS_PP_CENTERS)

        self.centers = result[2]
        self.segmented_image = self.centers[result[1].flatten()]

        return self.segmented_image