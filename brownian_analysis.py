"""
A set of functions to do some brownian motion analysis
    All units should be in metres, seconds
    Display for graphs will be in relevant axes
"""
import numpy as np

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