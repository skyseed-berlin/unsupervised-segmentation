import numpy

def stack_image_windows(arr, shape):
    """
    Return an array with given shape where shape[0] * shape[1] = arr.size

    If arr is of shape (n, nrows, ncols), i.e. n windows of shape (nrows, ncols),
    then the returned array arranges the windows from left to right & top to bottom, 
    preserving structures within windows.
    """
    n, nrows, ncols = arr.shape
    reshaped = (arr.reshape(shape[0]//nrows, -1, nrows, ncols)
               .swapaxes(1,2)
               .reshape(shape[0], shape[1]))

    return reshaped


# eventually windows could be created using the following function as a basis rather than the numpy function to allow for custom strides (later)
# def blockshaped(arr, nrows, ncols):
#     """
#     Return an array of shape (n, nrows, ncols) where
#     n * nrows * ncols = arr.size

#     If arr is a 2D array, the returned array looks like n subblocks with
#     each subblock preserving the "physical" layout of arr.
#     """
#     h, w = arr.shape
#     return (arr.reshape(h//nrows, nrows, -1, ncols)
#                .swapaxes(1,2)
#                .reshape(-1, nrows, ncols))