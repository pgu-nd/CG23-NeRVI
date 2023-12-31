import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import init
import time
import math

class ConditionedInstanceNorm3d(nn.Module):
    def __init__(self, num_feature_maps, tf_desc_sz, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.var = nn.Linear(tf_desc_sz, num_feature_maps)
        self.mu  = nn.Linear(tf_desc_sz, num_feature_maps)
        nn.init.normal_(self.var.weight, std=1e-2)
        nn.init.constant_(self.var.bias, 1.0)
        nn.init.normal_(self.mu.weight,  std=1e-2)
        nn.init.constant_(self.mu.bias,  0.0)
        self.n_feature_maps = num_feature_maps

    def forward(self, x, tf_desc):
        dtype = x.dtype
        bs = x.size(0)

        mu, var = self.mu(tf_desc).view(bs, -1, 1,1,1).float(), self.var(tf_desc).view(bs, -1, 1,1,1).float()
        # xmean = x.mean(dim=(2,3,4)).view(bs, self.n_feature_maps, 1,1,1)
        # xvar  = x.var( dim=(2,3,4)).view(bs, self.n_feature_maps, 1,1,1)
        # return (var * (  ( (x - xmean) / (xvar + self.eps) ) + mu  )).to(dtype)
        return ((x - mu) / (var + self.eps)).to(dtype)


class Sine(nn.Module):
    def __init(self):
        super(Sine,self).__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)

class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        output =  torch.sin(self.omega_0 * self.linear(input))
        return output

class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()

    
    def init_weights(self):
        with torch.no_grad():
            self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features) 
            #self.linear.weight.normal_(0,0.05) 
        
    def forward(self, input):
        return self.linear(input)

class ResBlock(nn.Module):
    def __init__(self,in_features,out_features,nonlinearity='relu'):
        super(ResBlock,self).__init__()
        nls_and_inits = {'sine':Sine(),
                         'relu':nn.ReLU(inplace=True),
                         'sigmoid':nn.Sigmoid(),
                         'tanh':nn.Tanh(),
                         'selu':nn.SELU(inplace=True),
                         'softplus':nn.Softplus(),
                         'elu':nn.ELU(inplace=True)}

        self.nl = nls_and_inits[nonlinearity]

        self.net = []

        self.net.append(SineLayer(in_features,out_features))

        self.net.append(SineLayer(out_features,out_features))

        self.flag = (in_features!=out_features)

        if self.flag:
            self.transform = SineLayer(in_features,out_features)

        self.net = nn.Sequential(*self.net)
    
    def forward(self,features):
        outputs = self.net(features)
        if self.flag:
            features = self.transform(features)
        return 0.5*(outputs+features)


class CoordNet(nn.Module):
    #A fully connected neural network that also allows swapping out the weights when used with a hypernetwork. Can be used just as a normal neural network though, as well.

    def __init__(self, in_features, out_features, init_features=64,num_res=10):
        super(CoordNet,self).__init__()
        self.pe_coord = PosEncoding(in_features[0],10)
        self.pe_view = PosEncoding(in_features[1],4)
        self.num_res = num_res
        '''
        self.sine1 = SineLayer(self.pe_coord.out_dim+self.pe_view.out_dim,num_features)
        self.sine2 = SineLayer(num_features,num_features)
        self.sine3 =  SineLayer(num_features,num_features)
        self.sine4 =  SineLayer(num_features,num_features)
        self.sine5 =  SineLayer(num_features,num_features)
        self.sine6 =  SineLayer(num_features,num_features)
        self.sine7 =  SineLayer(num_features,num_features)
        self.sine8 =  SineLayer(num_features,out_features)
        '''
        self.net = []

        #self.net.append(PositionEncoding(in_features,16))

        self.net.append(ResBlock(self.pe_coord.out_dim+self.pe_view.out_dim,init_features))
        #self.net.append(nl)
        self.net.append(ResBlock(init_features,2*init_features))
        #self.net.append(nl)
        self.net.append(ResBlock(2*init_features,4*init_features))
        #self.net.append(nl)

        for i in range(self.num_res):
            self.net.append(ResBlock(4*init_features,4*init_features))

        self.net.append(ResBlock(4*init_features, out_features))

        self.net = nn.Sequential(*self.net)



    def forward(self, coords):
        pe_coord = self.pe_coord(coords[:,0:2])
        pe_view = self.pe_view(coords[:,2:4])
        pe_feature = torch.cat((pe_coord,pe_view),dim=1)
        return self.net(pe_feature)



class PosEncoding(nn.Module):
    '''Module to add positional encoding as in NeRF [Mildenhall et al. 2020].'''
    def __init__(self, in_features, num_frequencies):
        super().__init__()
        self.in_features = in_features
        self.num_frequencies = num_frequencies
        self.out_dim = self.in_features + 2 * in_features * self.num_frequencies

    def forward(self, coords):
        coords = coords.view(coords.shape[0], -1, self.in_features)

        coords_pos_enc = coords
        for i in range(self.num_frequencies):
            for j in range(self.in_features):
                c = coords[..., j]

                sin = torch.unsqueeze(torch.sin((2 ** i) * np.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((2 ** i) * np.pi * c), -1)

                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)

        return coords_pos_enc.reshape(coords.shape[0], self.out_dim)


def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad



