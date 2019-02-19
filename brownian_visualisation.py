"""
A script to generate some trajectory behaviour and
then print some graphs for my thesis - basically a big demo.
"""

import pandas as pd 
import numpy as np
import otpm_read as ot

from matplotlib import pyplot as plt
from matplotlib import mlab
from scipy.stats import norm
from scipy.optimize import curve_fit

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size//2:]



def exponenial_func(x, a, b, c):
    return a*np.exp(-b*x)+c    

## DATA INPUTS:
k_b = 1.3806*10**-23  # Boltzman's constant, SI units
T = 300     # Kelvin


## USER INPUTS
traj_loc = 'ExpT14.dat'
experimental = True
new = False
# Reading the trajectory into a pandas data frame. 

if experimental:
    # Read in the experimental data
    exp_data = ot.param_read(
        param_loc='INFO-v0-Exp4.dat',
        exp_loc=traj_loc,
        name='test',
        mean_shift=True,
        )
    # Generating a time vector
    time_vector = range(np.size(exp_data.Y_positions))\
        *exp_data.time_between_position_samples*10**-6    # s from us
    # Stack it into a single array to prep for pandas data frame
    if new:
        stacked_data = np.vstack([exp_data.X_positions, exp_data.Y_positions, exp_data.intensities,
                   exp_data.Z_positions,
                   (exp_data.X_AoD_position - exp_data.AoD_midpoint_X)*exp_data.AoD_Xfactor,
                   (exp_data.Y_AoD_position - exp_data.AoD_midpoint_Y)*exp_data.AoD_Yfactor,
                   time_vector])
        data = pd.DataFrame(np.transpose(stacked_data))
        data.columns = ['x_pos', 'y_pos', 'intensity', 'z_pos', 'x_centre', 'y_centre', 'time']
    else: 
        stacked_data = np.vstack([exp_data.X_positions, exp_data.Y_positions, time_vector])
        data = pd.DataFrame(np.transpose(stacked_data))
        data.columns = ['x_pos', 'y_pos', 'time']
else:
    data = pd.read_table(sim_file_path, names=['time', 'x_pos', 'y_pos',
                         'intensity', 'x_centre', 'y_centre'])
    # Convert our metres into nanometres
    data.x_pos = data.x_pos*10**9
    data.y_pos = data.y_pos*10**9
    data.x_centre = data.x_centre*10**9
    data.y_centre = data.y_centre*10**9
#%%

## Generating a position histogram and gaussian fit.
n, bins, patches = plt.hist(data.x_pos, bins=100, normed=True)
mu, sigma = norm.fit(data.x_pos)
# add a 'best fit' line
y = mlab.normpdf( bins, mu, sigma)
l = plt.plot(bins, y, 'r--', linewidth=2)
plt.xlabel('X position deviation, nm')
print(sigma**2)
print(np.mean((data.x_pos-data.x_pos.mean())**2))
stiff =  k_b*T/((sigma*10**-9)**2)
plt.legend(['Gaussian fit, variance (MSD) = %.2f' % sigma**2, 'Position Histogram'])
plt.title('Axis Stiffness: %.3e N/m' % stiff)
plt.ylabel('Normalised Frequency')
plt.show()
"""
plt.figure()
plt.hist(data.y_pos, bins=100, normed=True)
mu, sigma = norm.fit(data.y_pos)
# add a 'best fit' line
y = mlab.normpdf( bins, mu, sigma)
l = plt.plot(bins, y, 'r--', linewidth=2)
plt.xlabel('Y position deviation, nm')
plt.show()
"""

## Generating a power spectrum.
ps = np.abs(np.fft.fft(data.x_pos))**2
freqs = np.fft.fftfreq(ps.size, exp_data.time_between_position_samples*10**-6)
plt.loglog(freqs, ps)
plt.xlabel('Power (a.u)')
plt.ylabel('Frequency (Hz)')
plt.show()
#%%
## Generating an autocorellation function
traj_size = 200;
plot_width = 200;
auto_x = np.ndarray((data.x_pos.size//traj_size,traj_size))
for i in np.array(range(data.x_pos.size//traj_size-1))*traj_size:
    auto_x[i//traj_size,:] = autocorr(data.x_pos[i:i+traj_size])
auto_x_mean = np.mean(auto_x,axis=0)
#Fit an exponential
popt, pcov = curve_fit(exponenial_func, data.time[:plot_width], auto_x_mean[:plot_width], p0=(1, 1e-6, 1))
plt.plot(data.time[:plot_width]*1000,auto_x_mean[:plot_width],'.')
plt.plot(data.time[:plot_width]*1000, exponenial_func(data.time[:plot_width], *popt))
# Calculate the trap stiffness;
drag = 9.42447*10**-9 # For 1um sphere
stiffness = popt[1]*drag
plt.xlabel('Lag time, ms')
plt.ylabel('Autocorellation, arbitrary units')
plt.legend(['Autocorellation Values, Avg of 1000 Traj.', 'Exponential fit.'])
plt.title('Stiffness estimate = %.6e N/m' % stiffness)
plt.show()