"""
A set of functions to do some brownian motion analysis
    All units should be in metres, seconds
    Display for graphs will be in relevant axes
"""
import numpy as np
import scipy as sp
import scipy.signal as sig
import matplotlib
from matplotlib import pyplot as plt

def yield_correlation(trap_a, trap_b, traj_size, num_traj):
    trap_a = trap_a-np.mean(trap_a)
    trap_b = trap_b-np.mean(trap_b)
    correlation = np.zeros(traj_size-1)
    for i in range(num_traj-1):
        small_correlation = np.zeros(traj_size-1)
        for k in range(traj_size-1):
            for j in range(traj_size-k):
                small_correlation[k] = small_correlation[k] + trap_a[j+(i*traj_size)+k]*trap_b[j+(i*traj_size)]
            small_correlation[k] = small_correlation[k]/(traj_size-k)
            #small_correlation[k] = np.corrcoef(trap_a[i*traj_size+k:(i+1)*traj_size], trap_b[i*traj_size:(i+1)*traj_size-k])[0,1]
        correlation = correlation + small_correlation
    return correlation/num_traj
        
def exp_wrap(estimate, **data):
    # A wrapper for returning the fitness of a curve estimate 
    x = data.get('x')
    y = data.get('y')*10**18 # Convert to nanometres for an actual fit!!
    return np.linalg.norm(y - estimate[0]*np.exp(estimate[1]*x))

def autocorr_fit(x, T, timestep=5*10**-6, dnvis=0.001, radius=10**-6, t_scale=6*10**-3):
    """
    Fits an exponential decay to the autocorrelation function, then return
    the stiffness. 
        x_m is one dimensional position array
        T is temperature in kelvin
        dnvis is dynamic viscocity
        radius is raius of the (assumed spherical) particle (meters)
        t_scale is the length (in time) of our subtrajectory
    """
    kb = 1.3806*10**-23 # boltzmann constant
    stokes_drag = 6*np.pi*dnvis*radius
    #Fix to handle weather or not x is a numpy thing or a array
    try:
        x[0]
    except KeyError:
        x = x.values
    
    # Calculate the length of a subteajectory in elements, and how many
    # sub trajectories we can extract from our system.
    sub_traj_length = int(t_scale//timestep)
    num_traj = len(x)//sub_traj_length
    correlation = yield_correlation(x,x,sub_traj_length,num_traj)


    """
    small_correlation = np.zeros(sub_traj_length*2-3)
    
    # Generate the autocorrelation:
    successes = num_traj
    for j in range(num_traj):
        start = j*sub_traj_length
        end = (j+1)*sub_traj_length-1
        try:
            small_correlation += np.correlate(x[start:end]-np.mean(x), x[start:end]-np.mean(x),'full')
        except ValueError:
            successes = successes-1
    
    correlation = small_correlation[np.size(small_correlation)//2:]/successes/(sub_traj_length*2)
    """
    time_base = np.arange(sub_traj_length-1)*timestep
    
    # Plot the autocorrelation
    fig, ax = plt.subplots()
    ax.plot(time_base, correlation,'o')
    ax.set_xlabel("Time, seconds")
    ax.set_ylabel("Autocorrelation, m^2")
    
    # Fit to the autocorrelation
    data = {'x':time_base,
            'y':correlation}
    guess = [0, 0]
    coeffs = sp.optimize.least_squares(exp_wrap, x0=guess, kwargs=data).x
    
    # Plot the fit, converting our coefficient back into metres.
    ax.plot(time_base, coeffs[0]/(10**18)*np.exp(coeffs[1]*time_base))
    ax.legend(["Autocorrelation","fit to Autocorrelation"])
    
    stiff = -1*stokes_drag*coeffs[1]
    ax.set_title('Axis Stiffness: %.3e N/m, Scale = %.3e N/m' % (stiff, coeffs[0]/(kb*T/stiff)/(10**18) ))
    
    return stiff
    
    

def stiffness_MSD(x, T):
    """
    Calculate the trap stiffness along every x using the
    MSD / Equipartition Method.
        x is a one dimensional position array
        T is the temperature in kelvin
    """
    k_b = 1.3806*10**-23  # Boltzman's constant, SI units
    MSD = np.mean((x-np.mean(x))**2)
    stiffness = k_b*T/MSD
    return stiffness

def MSD_gen(x, num_sub_traj=1000):
    """
    Generate an averaged mean-squared-displacement 
    """

    buff = 10
    # A number-of-points buffer to ensure we don't overlap sub trajectories
    sub_traj_points = np.size(x)//num_sub_traj

    MSD = np.zeros((num_sub_traj, sub_traj_points-buff))
    for i in range(num_sub_traj):
        MSD_Start = i*sub_traj_points
        for j in range(sub_traj_points-buff):
            MSD[i,j] = (x[MSD_Start] - x[MSD_Start+j])**2

    # Return the average MSD 
    MSD_avg = np.mean(MSD, axis=0)

    return MSD_avg


def gaussian_fit(x, T, use_filter=True):
    """
    Performs a gaussian fit to a histogram of x.
    Also returns the stiffness according to this fit.
        x is a one dimensional position array
    """
    from scipy.stats import norm
    from matplotlib import mlab
    from scipy.optimize import curve_fit


    if use_filter:
        # Generate a high pass filter (# This is 100Hz on a normal 5 us sample rate)
        a, b = sig.butter(1,0.001,'high')
        x_proc = sig.lfilter(a, b, x-np.mean(x))
    else:
        x_proc = x

    k_b = 1.3806*10**-23  # Boltzman's constant, SI units
    f,ax = plt.subplots()
    n, bins, patches = plt.hist(x_proc, bins=100, normed=True)
    mu, sigma = norm.fit(x_proc)
    # add a 'best fit' line
    y = norm.pdf(bins, mu, sigma)
    ax.plot(bins, y, '--', linewidth=2)
    ax.set_xlabel('position deviation metres')
    print(mu,sigma)
    stiff =  k_b*T/((sigma)**2)
    ax.legend(['Gaussian fit', 'Position Histogram'])
    ax.set_title('Axis Stiffness: %.3e N/m' % stiff)
    ax.set_ylabel('Normalised Frequency')
    return stiff, f

def log_blocking(x, n_blocks,f_start):
    """
    Perform block-averaging of a frequency-space signal, in such a way that blocks at higher frequencies
        contain MORE samples - i.e. the spacing of blocks is logarythmic, not uniform.
    """

    index_locs = np.linspace(f_start,np.size(x),n_blocks)
    block = np.zeros((n_blocks-1))
    for i in range(np.size(index_locs)-1):
        block[i] = np.mean(x[int(index_locs[i]):int(index_locs[i+1])])

    return block, index_locs[1:]

def lorentz_func(f, fc, num):
    """
        Defines a lorentzian with corner frequency fc, numerator num.
    """
    return num/(fc**2+f**2)

def lorentz_wrapper(guess, **kwargs):
    # kywargs contains x and y DATA, which we use to evaluate the dit.
    x = kwargs.get('x')
    y = kwargs.get('y')

    residuals = (lorentz_func(x, fc=guess[0], num=guess[1]) - y)
    return residuals

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def lorentzian_fit(x, timestep=5*10**-6, n_blocks=10000, f_start=10, f_end=20000):
    """
    Takes the power spectrum of x and fits a lorentzian
    Returns the corner frequency according to this fit, and the figure
        x is a one dimensional positional array
        timestep is the timestep of sampling
        n_blocks is the number of blocks to concatonate the sample to
        f_start is the start cutoff frequency
        f_end is the ending cutoff frequency

    Analysis is performed in accordance with 
    Berg-Sorensen, K. & Flyvbjerg, H. 
    Power spectrum analysis for optical tweezers. Rev. Sci. Instrum. 75, 594-612
    https://www.researchgate.net/publication/224481725_Berg-Sorensen_K_Flyvbjerg_H_Power_spectrum_analysis_for_optical_tweezers_Rev_Sci_Instrum_75_594-612
    
    """

    import scipy.fftpack as fp
    import scipy.optimize


    # Perform the fourier transform of x, and square it's norm to get the power spectrum;
    X = np.fft.rfft(x)
    freq_bins = (0.5/timestep)*np.array(range(np.size(X)))/np.size(X)

    f_start_id = find_nearest(freq_bins, f_start)
    f_end_id = find_nearest(freq_bins, f_end)

    X_cut = X[f_start_id:f_end_id]

    power, positions = log_blocking(np.abs(X_cut)**2, n_blocks, f_start)
    f,ax = plt.subplots()
    ax.loglog(freq_bins[(positions.astype(int)-1)], power)

    kwargs = {
    "x": freq_bins[(positions.astype(int)-1)],
    "y": power
    }
    guesses = scipy.optimize.least_squares(lorentz_wrapper, x0 = [f_start, 0], kwargs=kwargs, method='lm')
    fc = guesses.x[0] # The corner freuency.
    ax.loglog(freq_bins[(positions.astype(int)-1)], lorentz_func(freq_bins[(positions.astype(int)-1)], fc=guesses.x[0], num=guesses.x[1]))
    plt.xlabel('Frequency, Hz')
    plt.ylabel('Power, nm^2/Hz')
    ax.set_title('Corner Frequency: %i Hz' % fc)

    return guesses.x[0], f
        

def fc_stiff(fc, radius=False, stokes=False, dnvis=False):
    """
    A function to evaluate the trap stiffness of a particle given it's corner frequency
    You can provide either the radius (assumed a spherical particle), or the stokes drag coeff.
            I also need the dynamic viscoscity of the fluid (dynvis)
    """
    import numpy as np

    if radius is not False:
        stokes_drag = 6*np.pi*dnvis*radius
    elif stokes is not False:
        stokes_drag = stokes
    else:
        print("Radius or Stokes Drag not Specified!") 
        return 0

    stiffness = fc*2*np.pi*stokes_drag
    return stiffness

def general_check(x, T=300, timestep=5*10**-6, n_blocks=10000, f_start=10, f_end=20000, radius=0.5*10**-6, dnvis=0.001, plots=False, use_filter=True):
    """
    A function to compute the trap stiffness via lorentzian and fitting a gaussian.
        Default parameters are for a 1um sphere in water
    """

    stiff_gauss, fig_gauss = gaussian_fit(x, T)
    fc, fig_lorentzian = lorentzian_fit(x, timestep, n_blocks, f_start, f_end)
    stiff_lorentzian = fc_stiff(fc, radius=radius, dnvis=dnvis)
    stiff_auto= autocorr_fit(x, T, timestep=timestep, dnvis=dnvis, radius=radius, t_scale=10*10**-3)

    print('Gaussian Stiffness: %.3e N/m' % stiff_gauss)
    print('Lorentzian Stiffness: %.3e N/m' % stiff_lorentzian)
    print('Autocorrelation Stiffness: %.3e N/m' % stiff_auto)
    if plots:
        plt.show()

    return stiff_gauss, stiff_lorentzian, stiff_auto