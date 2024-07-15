import torch
from torch.autograd.functional import jacobian

def loss_function(pde, x):
    loss_sum = 0.

    f = pde.rightHandSide
    for xi in x:
        xi = xi.reshape(1)
        input_point = xi
        xi.requires_grad = True
        net_out = pde.forward(xi)
        psy_t = 1. + xi * net_out
        net_out.backward()
        d_net_out = xi.grad     #needs checking !!!!!!!!!!!!!!!!!!!!!!!
        d_psy_t = net_out + xi * d_net_out
        func = f(xi, psy_t)
        err_sqr = (d_psy_t - func) ** 2

        # psy_t = pde.trial(xi)
        # func = f(xi, pde.trial(xi))
        # trial_jacobian = jacobian(pde.trial, input_point, create_graph=True)
        # err_sqr = (trial_jacobian - func)**2
        loss_sum += err_sqr
    return loss_sum
