import torch
import torch.nn as nn
import numpy as np
from sympy import *
from sympy.parsing.sympy_parser import parse_expr

import shutil

class ODEnet(nn.Module):
    def __init__(self,N):
        super(ODEnet,self).__init__()
        #self.ode_numpy_path = 'c:\\Users\\snyta\\PycharmProjects\\ode_numpy'
        self.ode_numpy_path = 'c:\\Users\\Snytnikov.A\\PycharmProjects\\ode_numpy\ode_numpy'
        self.follow_numpy = True
        if self.follow_numpy:
            shutil.copy(self.ode_numpy_path+'\W00.txt','W00.txt')
            W00 = np.loadtxt('W00.txt')
            W00 = torch.from_numpy(W00).to(torch.float)
            self.N = W00.shape[0]
        else:
            self.N = N

        fc1 = nn.Linear(1,self.N)
        if self.follow_numpy:
           W00 = W00.reshape(10,1)
          # W00 = torch.from_numpy(W00)
           fc1.weight = torch.nn.Parameter(W00)
           fc1.bias = torch.nn.Parameter(torch.zeros(self.N))

        y_test = fc1(torch.ones(1))
        if self.follow_numpy:
           fc1.bias = torch.nn.Parameter(torch.zeros_like(fc1.bias))

        self.fc1 = fc1
        x = torch.ones(1)
        # x = x.to(torch.double)
        y = self.fc1(x)
        act_fc1 = torch.sigmoid
        y = act_fc1(y)
        self.act_fc1 = act_fc1


        #W01 = torch.from_numpy(W01).to(torch.float)

        fc2 = nn.Linear(self.N, 1)
        if self.follow_numpy:
            shutil.copy(self.ode_numpy_path + '\W01.txt', 'W01.txt')
            W01 = torch.from_numpy(np.loadtxt('W01.txt'))
            fc2.weight = torch.nn.Parameter(W01.reshape(1, self.N).float())
            fc2.bias = torch.nn.Parameter(torch.zeros((1)))
        act_fc2 = torch.sigmoid
       # test2_torch = fc2(act_fc1.reshape(1, self.N))
        y= fc2(y)
        self.fc2 = fc2
        # self.fc1 = fc1
        # self.act1 = torch.nn.Sigmoid
        self.act_fc2 = act_fc2
        qq = 0

    def forward(self,x):
        x = self.fc1(x)
        x = self.act_fc1(x)
        x = self.fc2(x)
        # x = self.act_fc2(x)
        return x

    def rightHandSide(self,x,psy):
        from auxiliary_functions import f
        return f(x,psy)

    def getExpressionFromLinear(self,f):
        y_str = ""
        for j,w in enumerate(f.weight):
            x_str = "x_" + str(j)
            w_str = "w_" + str(j)
            x_j = Symbol(x_str)
            w_j = Symbol(w_str)
            y_str += w_str + '*' +x_str
            if j < f.weight.shape[0] - 1:
                y_str += '+'

        ex = parse_expr(y_str)
        return ex  # x**2

    def boundary(self,x):
        return 1.0

    def trial(self,x):
        y = x*self.forward(x) + self.boundary(x)
        return y


if __name__ == '__main__':
    ode = ODEnet(10)
    #ex = ode.getExpressionFromLinear(ode.fc1)
    x = torch.ones(1)
    x.requires_grad = True
    y = ode(x)
    y.backward()
    dx = x.grad

    from loss import loss_function
    x_space = torch.linspace(0,1.0,ode.N)
    lf = loss_function(ode,x_space)

    qq  = 0