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
    ax.set_xlabel('position deviation, nm')
    stiff =  k_b*T/((sigma)**2)
    ax.legend(['Gaussian fit', 'Position Histogram'])
    ax.set_title('Axis Stiffness: %.3e N/m' % stiff)
    ax.set_ylabel('Normalised Frequency')
    return stiff, f

def log_blocking(x, n_blocks):
    """
    Perform block-averaging of a frequency-space signal, in such a way that blocks at higher frequencies
        contain MORE samples - i.e. the spacing of blocks is logarythmic, not uniform.
    """

    index_locs = np.linspace(10,np.size(x),n_blocks)
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
    print(guess)
    print(np.linalg.norm(residuals))
    return residuals

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def lorentzian_fit(x, timestep=5*10**-6, n_blocks=10000, f_start=10, f_end=50000):
    """
    Takes the power spectrum of x and fits a lorentzian
    Returns the stiffness according to this fit, and the figure
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
    freq_bins = (1/timestep)*np.array(range(np.size(X)))/np.size(X)

    f_start_id = find_nearest(freq_bins, f_start)
    f_end_id = find_nearest(freq_bins, f_end)

    X_cut = X[f_start_id:f_end_id]

    power, positions = log_blocking(np.abs(X_cut)**2, n_blocks)
    f,ax = plt.subplots()
    ax.loglog(freq_bins[(positions.astype(int)-1)], power)

    kwargs = {
    "x": positions,
    "y": power
    }
    guesses = scipy.optimize.least_squares(lorentz_wrapper, x0 = [1000, 30], kwargs=kwargs, method='lm')
    print(guesses.nfev)
    ax.loglog(positions, lorentz_func(positions, fc=guesses.x[0], num=guesses.x[1]))

        

