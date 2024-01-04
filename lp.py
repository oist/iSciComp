"""L^p norm module"""

import numpy as np

def norm(x, p=2):
    """The L^p norm of a vector."""
    y = abs(x) ** p
    return np.sum(y) ** (1/p)

def normalize(x, p=2):
    """L^p normalization"""
    return x/norm(x, p)
