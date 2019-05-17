import numpy as np


def apply_calib(raw_data, calib_data, scan_centre, conv_factor):
    """
    Calibrate our data from raw pinouts
    raw_data should be a struct that has at least:
        Pin1
        Pin2
        Pin3
        Pin4 (shocking!)
    calib_data should be two y=b+mx coeffs (output of calib_read)
    scan_centre is the X,Y centre of the whole scan, in MHz
        (this is taken as 0,0)
    conv_factor is the X,Y conversion betw. MHz and nm, nm/MHz
    """
    X_volts = raw_data.Pin1/(-raw_data.Pin3)
    Y_volts = raw_data.Pin2/raw_data.Pin4
    X_MHz = calib_data[0, 0] + X_volts*calib_data[0, 1] - scan_centre[0]
    Y_MHz = calib_data[1, 0] + Y_volts*calib_data[1, 1] - scan_centre[0]
    X_nm = X_MHz*conv_factor[0]
    Y_nm = Y_MHz*conv_factor[1]
    return X_nm, Y_nm


def calib_raw_read(calib_file=False):
    if calib_file:
        file = open(calib_file, 'r')
        data = np.loadtxt(file)
        file.close()
        # File format is 4 rows:
        # X AoD Position
        # PSD Value
        # Y AoD Position
        # PSD Value
        return data
    else:
        return 0


def calib_read(calib_file=False):
    if calib_file:
        file = open(calib_file, 'r')
        data = np.loadtxt(file)
        calib = exp_data('calib')
        file.close()
        # 2x rows of b+mx
        # Row 1 is for x dir, row 2 is y dir
        return data
    else:
        return 0


class exp_data:
    # I don't know if there's a better way to define
    # a class, but this is just so I can make a 'struct'
    def __init__(self, name):
        self.name = name


def param_read(param_loc=False,
               exp_loc=False,
               name=False,
               mean_shift=False,
               raw_loc=False):
    """
    A program to read the parameters of our experiment.
        param_loc - location of our parameters file
        exp_loc - location of our experiment file
        name - name of our experiment
        mean_shift - do we want to shift everything to be
            zero mean? Shouldn't use this unless you're only
            looking at a single trajectory!
    """
    struct = exp_data(name)
    file = open(param_loc, "r")
    lines = np.float32(file.read().split('\t'))
    v_num = lines[-1]
    if param_loc:
        if v_num == np.float32(0.4) or v_num == np.float32(0.45) or v_num == np.float32(0.46) or v_num == np.float32(0.47):
            struct.Xmin = lines[0]                              # MHz
            struct.Xmax = lines[1]                              # MHz
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
            struct.AoD_Xfactor = lines[13]                      # nm/MHz
            struct.AoD_Yfactor = lines[14]                      # nm/MHz
            file.close()
        elif v_num == np.float32(0.3):
            #   An older version of the info file
            #   I have bee chasing a phantom bug with the existence of
            #   The particle diameter. 
            struct.Xmin = lines[0]                          # MHz
            struct.Xmax = lines[1]                          # MHz
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
            struct.AoD_Xfactor = lines[13]                  # nm/MHz
            struct.AoD_Yfactor = lines[14]                  # nm/MHz
            file.close()
        elif v_num == np.float32(0.5):
            #v 0.5 has element #12 (particle diam) not being written
            struct.Xmin = lines[0]                              # MHz
            struct.Xmax = lines[1]                              # MHz
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
            struct.AoD_Xfactor = lines[12]                      # nm/MHz
            struct.AoD_Yfactor = lines[13]                      # nm/MHz
            file.close()
        elif v_num == np.float32(0.6):
            # V 0.6 is my brownian-motion only system. Info file is the same as 0.45
            # 0.6 specifically is a temp-debug set that I don't expect to be used much
            struct.Xmin = lines[0]                              # MHz
            struct.Xmax = lines[1]                              # MHz
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
            struct.AoD_Xfactor = lines[13]                      # nm/MHz
            struct.AoD_Yfactor = lines[14]                      # nm/MHz
            file.close()
    # Read in the raw pin file
    if raw_loc:
        if v_num <= np.float(0.45):
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
        else:
            raw = np.loadtxt(raw_loc)
            exp_data.Pin1 = raw[0, :]
            exp_data.Pin2 = raw[1, :]
            exp_data.Pin3 = raw[2, :]
            exp_data.Pin4 = raw[3, :]
            exp_data.IPin = raw[4, :]
    # Get the experiment data
    if exp_loc:
        file = open(exp_loc, "r")
        data = np.loadtxt(file)
        if v_num < 0.6:  # Added a version 0.6 that's just for brownian motion
            # Remove rows that are zero-rows (from a default value issue in LabView)
            stop_index = np.where(~data[:, 0:3].any(axis=1))[0]
            if stop_index:
                X_positions = data[0:stop_index[0], 0]
                Y_positions = data[0:stop_index[0], 1]
                Intensities = data[0:stop_index[0], 2]
                Z_positions = data[0:stop_index[0], 3]
                X_AOD_position = data[0:stop_index[0], 4]
                Y_AOD_position = data[0:stop_index[0], 5]
            else:
                X_positions = data[:, 0]
                Y_positions = data[:, 1]
                Intensities = data[:, 2]
                Z_positions = data[:, 3]
                X_AOD_position = data[:, 4]
                Y_AOD_position = data[:, 5]
                if len(data[0,:]) > 6:
                    Write_Index = data[:,6]
            # Do we shift it so the mean position is zero?
            if mean_shift:
                struct.X_positions = X_positions - np.mean(X_positions)
                struct.Y_positions = Y_positions - np.mean(Y_positions)
            else:
                struct.X_positions = X_positions
                struct.Y_positions = Y_positions
            struct.X_AoD_position = X_AOD_position
            struct.Y_AoD_position = Y_AOD_position
            struct.Z_positions = Z_positions
            struct.intensities = Intensities
            if len(data[0,:]) > 6:
                struct.write_index = Write_Index
            file.close()
        else:
            # This is for brownian motion modes only
            X_positions = data[:, 0]
            Y_positions = data[:, 1]
            Z_positions = data[:, 2]
            Trap_positions = data[:, 3]
            struct.X_positions = X_positions
            struct.Y_positions = Y_positions
            struct.Z_positions = Z_positions
            struct.Trap_positions = Trap_positions
            file.close()

    return struct   