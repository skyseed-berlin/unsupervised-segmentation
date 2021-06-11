from models.spatial_pooling import SpatialPooling
import pytest 
import numpy as np 

# test baseline window function for percentage of target segment
@pytest.fixture(scope="module")
def testarray(margins=False, depth=3):    
    
    testwindow = np.int32(np.random.rand(4,4,depth)*255)

    if margins:
        testwindow = 

def test_baseline_nomargins(testarray): 
    SpatialPooling.perc_segment(window=testarray(), target_segment=)

def test_baseline_wmargins():
    SpatialPooling.perc_segment(window=, target_segment=)


