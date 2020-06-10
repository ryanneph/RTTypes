from copy import deepcopy

import numpy as np
from scipy.ndimage.morphology import binary_dilation

from .volume import Volume

def generate_selem_sphere(shape, radius, center):
    """generate binary structuring element to be used in morphological operations,
    All measures given in pixels
    Borrowed from https://stackoverflow.com/a/46626448/6347151"""
    semisizes = radius
    grid = [slice(-x0, dim-x0) for x0, dim in zip(center, shape)]
    center = np.ogrid[grid]
    arr = np.zeros(shape, dtype=float)
    for x_i, semisize in zip(center, semisizes):
        arr += (x_i / semisize)**2
    return arr <= 1.0

def binary_expansion(vol: Volume, radius=1, inplace=False):
    """Expand binary mask volume by radius in physical units of millimeters"""
    # generate structuring element
    radius_pixels = np.array((radius,)*3)/vol.frame.spacing
    size = np.ceil(radius_pixels*2).astype(int)
    size += np.mod(size+1,2)
    selem = generate_selem_sphere(size[::-1], radius_pixels[::-1], size[::-1]//2)

    if inplace:
        copyvol = vol
    else:
        copyvol = deepcopy(vol)
    copyvol.data = binary_dilation(vol.data, selem).astype(vol.data.dtype)
    return copyvol
