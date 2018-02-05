test_x_path = "/shared/datasets/ucfTrainTestlist/test.txt"                            
test_y_path = "/shared/datasets/ucfTrainTestlist/test_y.txt"                         

temporal_path_hor = "temporal/tvl1_flow/u"
temporal_path_ver = "temporal/tvl1_flow/v"

spatial_path = "spatial/jpegs_256"

temporal_model_path = "temporal1.h5"
spatial_model_path = "spatial1.h5"


def get_test_x():
    return test_x_path

def get_test_y():
    return test_y_path

def get_temporal_path_hor():
    return temporal_path_hor + '/'

def get_temporal_path_ver():
    return temporal_path_ver + '/'

def get_spatial_path():
    return spatial_path + '/'

def get_temporal_model_path():
    return temporal_model_path

def get_spatial_model_path():
    return spatial_model_path

