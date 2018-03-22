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


def param_read(param_loc, exp_loc, name, mean_shift=False, raw_loc='none'):
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
    if lines[-1] == np.float32(0.4):
        struct.Xmin = lines[0]                              # MHz
        struct.Ymax = lines[1]                              # MHz
        struct.Xpoints = lines[2]                           # Number
        struct.Ymin = lines[3]                              # MHz
        struct.Ymax = lines[4]                              # MHz
        struct.Ypoints = lines[5]                           # Number
        struct.AoD_midpoint_X = lines[6]                    # MHz
        struct.AoD_midpoint_Y = lines[7]                    # MHz
        struct.AoD_factor = lines[8]                        # Volts
        struct.samples_per_stop = lines[9]                  # Number
        struct.time_between_position_samples = lines[10]    # us
        struct.position_samples_per_integration = lines[11] # Number
        struct.AoD_Xfactor = lines[13]                      # um/MHz
        struct.AoD_Yfactor = lines[14]                      # um/MHz
        file.close()

    elif lines[-1] == np.float32(0.3):
        #   An older version of the info file
        struct.Xmin = lines[0]                          # MHz
        struct.Ymax = lines[1]                          # MHz
        struct.Xpoints = lines[2]                       # Number
        struct.Ymin = lines[3]                          # MHz
        struct.Ymax = lines[4]                          # MHz
        struct.Ypoints = lines[5]                       # Number
        struct.AoD_midpoint_X = lines[6]                # MHz
        struct.AoD_midpoint_Y = lines[7]                # MHz
        struct.AoD_factor = lines[8]                    # Volts
        struct.samples_per_stop = lines[9]              # Number
        struct.time_unit = lines[10]                    # us
        struct.time_units_per_sample = lines[11]        # Number
        struct.AoD_Xfactor = lines[13]                  # um/MHz
        struct.AoD_Yfactor = lines[14]                  # um/MHz
        file.close()

    elif lines[-1] == np.float32(0.45):
        # This is 0.4, but we also have a 'RAW' data file
        struct.Xmin = lines[0]                              # MHz
        struct.Ymax = lines[1]                              # MHz
        struct.Xpoints = lines[2]                           # Number
        struct.Ymin = lines[3]                              # MHz
        struct.Ymax = lines[4]                              # MHz
        struct.Ypoints = lines[5]                           # Number
        struct.AoD_midpoint_X = lines[6]                    # MHz
        struct.AoD_midpoint_Y = lines[7]                    # MHz
        struct.AoD_factor = lines[8]                        # Volts
        struct.samples_per_stop = lines[9]                  # Number
        struct.time_between_position_samples = lines[10]    # us
        struct.position_samples_per_integration = lines[11] # Number
        struct.AoD_Xfactor = lines[13]                      # um/MHz
        struct.AoD_Yfactor = lines[14]                      # um/MHz
        file.close()
        # Read in the raw pin file
        file = open(raw_loc, "r")
        raw = np.float32(file.read().split('\t'))
        file.close()
        # Split it into 5 pins
        assert np.mod(np.size(raw), 5) == 0
        IPin = np.zeros(np.size(raw)//5)
        Pin1 = np.zeros(np.size(raw)//5)
        Pin2 = np.zeros(np.size(raw)//5)
        Pin3 = np.zeros(np.size(raw)//5)
        Pin4 = np.zeros(np.size(raw)//5)
        for i in range(np.size(raw)//5):
            Pin1[i] = raw[i*5]
            Pin2[i] = raw[i*5+1]
            Pin3[i] = raw[i*5+2]
            Pin4[i] = raw[i*5+3]
            IPin[i] = raw[i*5+4]
        exp_data.IPin = IPin
        exp_data.Pin1 = Pin1
        exp_data.Pin2 = Pin2
        exp_data.Pin3 = Pin3
        exp_data.Pin4 = Pin4

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
