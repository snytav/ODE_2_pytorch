import torch
from torch.autograd.functional import jacobian

from equation import LEFT,RIGHT

def f(x, psy):
    '''
        d(psy)/dx = f(x, psy)
        This is f() function on the right
    '''
    return RIGHT(x) - psy * LEFT(x)

def loss_function(pde, x):
    loss_sum = 0.
    for xi in x:
        xi = xi.reshape(1)
        input_point = xi
        net_out = pde.forward(xi)
        psy_t = pde.trial(xi)
        func = f(xi, pde.trial(xi))
        trial_jacobian = jacobian(pde.trial, input_point, create_graph=True)
        err_sqr = (trial_jacobian - func)**2
        loss_sum += err_sqr
    return loss_sum
