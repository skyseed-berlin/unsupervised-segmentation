from src.models.spatial_pooling import SpatialPooling
import pytest 
import numpy as np
from itertools import permutations 

# test baseline window function for percentage of target segment. 

# set up test data: 4x4 windows of varying depth

@pytest.fixture(scope="module", params=[3,5])
def image_depth(request):

    return request.param


@pytest.fixture(scope="module")
def target_segment(image_depth):

    np.random.seed(14)
    target_segment = np.random.randint(0, 255, image_depth)

    return target_segment


# varying fractions of target segment (0 - half - all)
@pytest.fixture(scope="module", params=[0, 8, 16]) 
def array_nomargins(image_depth, target_segment, request):    
    
    np.random.seed(207)

    # this doesn't work correclty still (same goes for black pixel insertion below)
    sample_array = np.random.randint(255, size=(4,4,image_depth))
    
    perms = list(permutations(np.arange(4)+1, 2))
    idx = np.random.choice(len(perms), size=request.param, replace=False)

    x = [perms[i][0] for i in idx]
    y = [perms[i][1] for i in idx], 

    sample_array[x, y, :] = target_segment

    return sample_array, request.param


# varying fractions of entirely black pixels (0 - half - all)
@pytest.fixture(scope="module", params = [0, 8, 16])
def array_margins(image_depth, array_nomargins, request):

    # introduce black pixels
    black = np.array([0]*image_depth)

    np.random.seed(145)
    idx = np.random.randint(array_nomargins.shape[0], size=request.param)

    sample_array = array_nomargins[idx, idx, :] = black

    return sample_array, request.param


# start testing

def test_baseline_nomargins(array_nomargins, target_segment): 

    testresult = SpatialPooling.perc_segment(window=array_nomargins[0], target_segment=target_segment)

    assert testresult.shape[0:2] == array_nomargins[0].shape[0:2], \
        "First two dimensions of result array should be the same as for input array."

    assert len(np.unique(testresult)) == 1, \
        "Result array should have constant value"

    true_value = array_nomargins[1]/16
    assert (testresult == true_value).all(), \
        "Result value should be " + str(true_value)



