import torch
from torch.autograd.functional import jacobian

from equation import LEFT,RIGHT

def f(x, psy):
    '''
        d(psy)/dx = f(x, psy)
        This is f() function on the right
    '''
    return RIGHT(x) - psy * LEFT(x)

