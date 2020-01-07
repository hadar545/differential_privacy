import numpy as np
import os

def check_dir_exists(dir_path):
    if os.path.isdir(dir_path):
        return dir_path
    else:
        os.mkdir(dir_path)
        return dir_path

def print_arr_id(arr, arr_name='_', numpyCreature=1, return_as_str=0):
    """
    This is a helper function, mostly intended for the coding process alone,
    it prints out a quick recap of properties of the input array
    :param arr: input array
    :param arr_name: input array name
    :param numpyCreature: 1 if this is a numpy array
    :return: -
    """
    # todo implement the return_as_str option
    if (numpyCreature):
        print("\n" + "array id for: " + arr_name +
              "\n dimensions: %s \n shape: %s \n size: %s \n dtype: %s \n nonzeros: %s \n nonzeros percentage: %s \n max: %s \n min: %s \n"
              % (arr.ndim, arr.shape, arr.size, arr.dtype, np.count_nonzero(arr),
                 "{0:.2f}".format(np.count_nonzero(arr) / arr.size), np.max(arr), np.min(arr)))
    else:
        print("\n" + arr_name + "  shape: %s \n" % (arr.shape))

def get_file_name(path, suffix=None):
    last_slash = max(path.rfind("\\"), path.rfind("/"))
    start_of_name = (last_slash + 1) if (last_slash != (-1)) else 0
    start_of_suffix = path.rfind(suffix) if suffix is not None else len(path)
    return path[start_of_name: start_of_suffix]
