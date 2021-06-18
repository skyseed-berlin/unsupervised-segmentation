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
    
    perms = list(permutations(np.arange(4), 2))
    for i in np.arange(0,4):
        perms.append((i,i))
    idx = np.random.choice(len(perms), size=request.param, replace=False)

    x = [perms[i][0] for i in idx]
    y = [perms[i][1] for i in idx] 

    sample_array[x, y, :] = target_segment

    return sample_array, request.param


# varying fractions of entirely black pixels (0 - half - all)
@pytest.fixture(scope="module", params = [8, 16])
def array_margins(image_depth, array_nomargins, request):

    # introduce black pixels
    black = np.array([0]*image_depth)

    perms = list(permutations(np.arange(4), 2))
    for i in np.arange(0,4):
        perms.append((i,i))

    idx = np.random.choice(len(perms), size=request.param, replace=False)

    x = [perms[i][0] for i in idx]
    y = [perms[i][1] for i in idx] 

    sample_array = np.copy(array_nomargins[0])
    sample_array[x, y, :] = black

    return sample_array, array_nomargins[1], request.param


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


def test_baseline_margins(array_margins, target_segment):

    testresult = SpatialPooling.perc_segment(window=array_margins[0], target_segment=target_segment)

    assert testresult.shape[0:2] == array_margins[0].shape[0:2], \
        "First two dimensions of result array should be the same as for input array."

    assert len(np.unique(testresult)) <= 2, \
        "Result array should have at most two constant value (zero and fraction of target segment)"

    assert 0 in np.unique(testresult), \
        "Result array should contain at least some black pixels"


def test_margins_ignored_correctly(target_segment):

    testarray = np.zeros((2,2,3), dtype=int)
    testarray[1, 1, :] = np.array([1,2,3])

    testresult = SpatialPooling.perc_segment(window=testarray, target_segment=np.array([1,2,3]))

    assert set(np.unique(testresult)) == {0,1}, \
        "Result array should contain zeros and onces only. Black pixels are probably not ignored correctly!"

    assert testresult[1, 1] == 1, \
        "Non-black pixel not replaced with fraction at correct position."

    assert (np.delete(testresult, 1, 1) == 0).all(), \
        "Black pixels should have value zero at correct positions after transformation." 