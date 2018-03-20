import numpy as np


def calib_raw_read(calib_file):
    file = open(calib_file, 'r')
    data = np.loadtxt(file)
    # File format is 4 rows:
    # X AoD Position
    # PSD Value
    # Y AoD Position
    # PSD Value
    return data


class exp_data:
    # I don't know if there's a better way to define
    # a class, but this is just so I can make a 'struct'
    def __init__(self, name):
        self.name = name


def param_read(param_loc, exp_loc, name, mean_shift=False):
    """
    A program to read the parameters of our experiment.
        param_loc - location of our parameters file
        exp_loc - location of our experiment file
        name - name of our experiment
        mean_shift - do we want to shift everything to be
            zero mean? Shouldn't use this unless you're only
            looking at a single trajectory!
    """
    # Get the info
    struct = exp_data(name)
    file = open(param_loc, "r")
    lines = np.float32(file.read().split('\t'))
    assert lines[15] == np.float32(0.4) or lines[15] == np.float32(0.3),\
        "Unsupported Version! Use only Versions 0.3 or 0.4"
    struct.Xmin = lines[0]
    struct.Ymax = lines[1]
    struct.Xpoints = lines[2]
    struct.Ymin = lines[3]
    struct.Ymax = lines[4]
    struct.Ypoints = lines[5]
    struct.AoD_midpoint_X = lines[6]
    struct.AoD_midpoint_Y = lines[7]
    struct.AoD_factor = lines[8]
    struct.samples_per_stop = lines[9]
    struct.time_between_position_samples = lines[10]    # us
    struct.position_samples_per_integration = lines[11]
    struct.AoD_Xfactor = lines[13]
    struct.AoD_Yfactor = lines[14]
    file.close()

    # Get the experiment data from another file
    file = open(exp_loc, "r")
    data = np.loadtxt(file)
    X_positions = data[:, 0]
    Y_positions = data[:, 1]
    Intensities = data[:, 2]
    Z_positions = data[:, 3]
    X_AOD_position = data[:, 4]
    Y_AOD_position = data[:, 5]
    # Do we shift it so the mean position is zero?
    if mean_shift:
        struct.X_positions = X_positions - np.mean(X_positions)
        struct.Y_positions = Y_positions - np.mean(Y_positions)
    else:
        struct.X_positions = X_positions
        struct.Y_positions = Y_positions
    return struct
