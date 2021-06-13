from src.models.spatial_pooling import SpatialPooling
import pytest 
import numpy as np 

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

    sample_array = np.random.randint(255, (4,4,image_depth))
    idx = np.random.randint(sample_array.shape[0], size=request.param)

    sample_array[idx, idx, :] = target_segment

    return sample_array


# varying fractions of entirely black pixels (0 - half - all)
@pytest.fixture(scope="module", params = [0, 8, 16])
def array_margins(image_depth, array_nomargins, request):

    # introduce black pixels
    black = np.array([0]*image_depth)

    np.random.seed(145)
    idx = np.random.randint(array_nomargins.shape[0], size=request.param)

    sample_array = array_nomargins[idx, idx, :] = black

    return sample_array


# start testing

def test_baseline_nomargins(testarray, target_segment): 
    test = SpatialPooling.perc_segment(window=testarray, target_segment=target_segment)



