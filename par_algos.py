"""
Script to run algorithms for the calculation of modeled PAR and a series of constants for use in selecting cloudless days.

Author: Thomas Armstrong (tomarmstro)
Created: 1/03/2024
"""

import numpy as np
from statistics import stdev
import math

def get_model_par(z):
    """
    Calculate the modeled Photosynthetically Active Radiation (PAR) value based on the zenith angle.

    This function computes a modeled PAR value (in units specific to the model coefficients) based on the cosine of the zenith angle.
    The model formula is a polynomial function that adjusts PAR based on the zenith angle of the sun.

    Args:
        z (float): The zenith angle in degrees (0-90 degrees), representing the angle between the sun and the vertical direction.

    Returns:
        float: Modeled PAR value adjusted by the zenith angle, using polynomial regression.
    """
    pi = 3.1415926
    z0 = z * pi / 180.0
    # Get cosine of zenith angle
    cz = np.cos(z0)
    modpar = -7.1165 + 768.894 * cz + 4023.167 * cz ** 2 - 4180.1969 * cz ** 3 + 1575.0067 * cz ** 4
    return modpar


def statsxy(x, y):
    """
    Calculate statistical constants representing deviations between observed and modeled PAR values.

    This function generates two lists of constants that quantify the differences between `x` (observed PAR values) and scaled versions of `y` (modeled PAR values). For each scaling factor, it computes the standard deviation of the difference between `x` and the scaled `y` values, as well as the standard deviation normalized by the maximum of `y`.

    Args:
        x (list or np.array): Observed PAR values.
        y (list or np.array): Modeled PAR values.

    Returns:
        tuple: A tuple containing:
            - const (list): Standard deviations of differences between `x` and `y` scaled by a decreasing factor.
            - const1 (list): Normalized standard deviations of differences between `x` and scaled `y`, where each standard deviation is divided by the maximum value of `y`.
    """

    const = []
    const1 = []
    for j in range(19):
        dely = 2.0 - j * 0.1
        y1 = y * dely
        const.append(stdev(x - y1))
        const1.append(stdev(x - y1) / np.max(y))
    # np.std gives slightly different constant values
        # const.append(np.std(x - y1))
        # const1.append(np.std(x - y1) / np.max(y))
    return const, const1


# Zenith angle
def zen(lat0, long0, dn, hr0, min0):
    """
    Calculate the solar zenith angle and associated parameters based on geographic location and time.

    This function determines the zenith angle (z), radius vector (rv), equation of time (et), and solar declination (dec) for a given latitude, longitude, day of the year, hour, and minute. These values are useful for solar irradiance and solar position calculations.

    Args:
        lat0 (float): Latitude in radians.
        long0 (float): Longitude in degrees.
        dn (int): Day of the year (1 to 365).
        hr0 (int): Hour of the day (24-hour format).
        min0 (int): Minute of the hour.

    Returns:
        tuple: A tuple containing:
            - z (float): Solar zenith angle in degrees.
            - rv (float): Radius vector, representing Earth's distance from the Sun in astronomical units.
            - et (float): Equation of time in hours, which accounts for Earth's elliptical orbit and axial tilt.
            - dec (float): Solar declination in radians, indicating the angle between Earth's equatorial plane and the Sun's rays.
    """

    # xl is a yearly timescale extending from 0 to 2 pi.
    xl = 2.0 * math.pi * (dn - 1) / 365.0

    dec = (
            0.006918 - 0.399912 * np.cos(xl) + 0.07257 * np.sin(xl) -
            0.006758 * np.cos(2.0 * xl) + 0.000907 * np.sin(2.0 * xl) -
            0.002697 * np.cos(3.0 * xl) + 0.00148 * np.sin(3.0 * xl)
    )

    rv = (
            1.000110 + 0.034221 * np.cos(xl) + 0.001280 * np.sin(xl) +
            0.000719 * np.cos(2.0 * xl) + 0.000077 * np.sin(2.0 * xl)
    )

    et = (
             0.000075 + 0.001868 * np.cos(xl) - 0.032077 * np.sin(xl) -
             0.014615 * np.cos(2.0 * xl) - 0.04089 * np.sin(2.0 * xl)
     ) * 229.18 / 60.0

    # zenith angle estimated (z)
    tst = hr0 + ((min0 + 5.0) / 60.0) - (4.0 / 60.0) * np.abs(150.0 - long0) + et
    hrang = (12.0 - tst) * 15.0 * math.pi / 180.0
    # Cosine zenith angle
    cz = np.sin(lat0) * np.sin(dec) + np.cos(lat0) * np.cos(dec) * np.cos(hrang)
    # zenith angle
    z = (180.0 / math.pi) * np.arccos(cz)
    
    return z, rv, et, dec