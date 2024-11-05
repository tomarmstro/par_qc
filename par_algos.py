import numpy as np
from statistics import stdev
import math

def get_model_par(z):
    '''
    Calculates the model PAR value based on the zenith angle

        Parameters:
            z:  
    '''
    pi = 3.1415926
    z0 = z * pi / 180.0
    # Get cosine of zenith angle
    cz = np.cos(z0)
    modpar = -7.1165 + 768.894 * cz + 4023.167 * cz ** 2 - 4180.1969 * cz ** 3 + 1575.0067 * cz ** 4
    return modpar


def statsxy(x, y):
    """Generates constants based on x, y

    Args:
        x (list): raw par values
        y (list): model par values

    Returns:
        const (list of lists): _description_
        const1 (list of lists)
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
    """Calculates the zenith angle and associated variables based on the latitude and time.

    Args:
        lat0 (_type_): _description_
        long0 (_type_): _description_
        dn (_type_): _description_
        hr0 (_type_): _description_
        min0 (_type_): _description_

    Returns:
        z (_type_): Zenith angle
        rv (_type_): Raduis vector
        et (_type_): Equation of time
        dec (_type_): Declination
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