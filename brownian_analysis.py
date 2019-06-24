"""
A set of functions to do some brownian motion analysis
    All units should be in metres, seconds
    Display for graphs will be in relevant axes
"""
import numpy as np
import scipy as sp
import matplotlib
from matplotlib import pyplot as plt

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


def gaussian_fit(x, T):
    """
    Performs a gaussian fit to a histogram of x.
    Also returns the stiffness according to this fit.
        x is a one dimensional position array
    """
    from scipy.stats import norm
    from matplotlib import mlab
    from scipy.optimize import curve_fit


    k_b = 1.3806*10**-23  # Boltzman's constant, SI units
    f,ax = plt.subplots()
    n, bins, patches = plt.hist(x, bins=100, normed=True)
    mu, sigma = norm.fit(x)
    # add a 'best fit' line
    y = mlab.normpdf(bins, mu, sigma)
    ax.plot(bins, y, 'r--', linewidth=2)
    ax.set_xlabel('position deviation metres')
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

def general_check(x, T=300, timestep=5*10**-6, n_blocks=10000, f_start=10, f_end=20000, radius=0.5*10**-6, dnvis=0.001):
    """
    A function to compute the trap stiffness via lorentzian and fitting a gaussian.
        Default parameters are for a 1um sphere in water
    """

    stiff_gauss, fig_gauss = gaussian_fit(x, T)
    fc, fig_lorentzian = lorentzian_fit(x, timestep, n_blocks, f_start, f_end)
    stiff_lorentzian = fc_stiff(fc, radius=radius, dnvis=dnvis)

    print('Gaussian Stiffness: %.3e N/m' % stiff_gauss)
    print('Lorentzian Stiffness: %.3e N/m' % stiff_lorentzian)
    plt.show()

    return 0