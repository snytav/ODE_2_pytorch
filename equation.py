import numpy as np
import torch


def LEFT(x):
    '''
        Left part of initial equation
    '''
    return 0.0 #x + (1. + 3.*x**2) / (1. + x + x**3)


def RIGHT(x):
    '''
        Right part of initial equation
    '''
    return torch.exp(x) #x**3 + 2.*x + x**2 * ((1. + 3.*x**2) / (1. + x + x**3))

def psy_analytic(x):
    '''
        Analytical solution of current problem
    '''
    return np.exp(x) # (np.exp((-x**2)/2.)) / (1. + x + x**3) + x**2
