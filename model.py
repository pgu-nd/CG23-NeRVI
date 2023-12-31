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
from torch.nn import init
from siren import *


class NeRVBlock(nn.Module):
    def __init__(self, in_channel, out_channel, up_scale):
        super(NeRVBlock,self).__init__()
        self.conv = nn.Sequential(
          nn.Conv2d(in_channel, out_channel*up_scale*up_scale, 3, 1, 1),
          nn.PixelShuffle(up_scale))
        self.norm = nn.BatchNorm2d(out_channel)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

			
# 400 x 400
class NeRV_Net_400(nn.Module):

  def __init__(self, in_features, out_channels, init_features, nerv_init_features,opt):
      super(NeRV_Net_400,self).__init__()

      self.pos = opt.pos

      ### Postioal Encoding for capturing high frequency regions

      if self.pos == 1:
        print('Use Positional Encoding')

        self.pe_t = PosEncoding(in_features[0],opt.lt)
        self.pe_v = PosEncoding(in_features[1],opt.lv)

        self.init_dim = self.pe_t.out_dim+self.pe_v.out_dim

      else:
        self.init_dim = np.sum(in_features)

      ### MLP for upscaling input dimensions

      self.rb1 = ResBlock(self.init_dim,init_features)
      self.rb2 = ResBlock(init_features+self.init_dim,4*init_features)


      ### Upscale 1D vector to 2D image (i.e., 256 X 256)

      self.NeRVBlock1 = NeRVBlock(4*init_features//4,nerv_init_features,5)
      self.NeRVBlock2 = NeRVBlock(nerv_init_features,nerv_init_features//2,5)
      self.NeRVBlock3 = NeRVBlock(nerv_init_features//2,nerv_init_features//4,2)
      self.NeRVBlock4 = NeRVBlock(nerv_init_features//4,nerv_init_features//8,2)
      self.NeRVBlock5 = NeRVBlock(nerv_init_features//8,nerv_init_features//16,2)
      #self.NeRVBlock6 = NeRVBlock(nerv_init_features//16,nerv_init_features//32,2)

      self.final = nn.Conv2d(nerv_init_features//16, out_channels, 1, 1)

      self.fc_dim = 4*init_features
        

  def forward(self, coords):

    if self.pos == 1:
      pe_t = self.pe_t(coords[:,0:1])
      pt_v = self.pe_v(coords[:,1:3])

      pe = torch.cat((pe_t,pt_v),dim=1)

    else:
      pe = coords

    #print('pe shape', pe.shape)

    rb1 = self.rb1(pe)
    rb2 = self.rb2(rb1)

    #print('rb2 shape', rb2.shape)

    rb2_reshape = rb2.view(rb2.size(0), self.fc_dim//4, 2, 2)

    #print('rb2_reshape shape', rb2_reshape.shape) # torch.Size([2, 128, 2, 2])

    NeRVBlock1 = self.NeRVBlock1(rb2_reshape)

    NeRVBlock2 = self.NeRVBlock2(NeRVBlock1)

    NeRVBlock3 = self.NeRVBlock3(NeRVBlock2)

    NeRVBlock4 = self.NeRVBlock4(NeRVBlock3)

    NeRVBlock5 = self.NeRVBlock5(NeRVBlock4)

    #NeRVBlock6 = self.NeRVBlock6(NeRVBlock5)

    final = self.final(NeRVBlock5)

    img_out = torch.sigmoid(final)
    return img_out


# ### 600 x 600
class NeRV_Net_600(nn.Module):

  def __init__(self, in_features, out_channels, init_features, nerv_init_features,opt):
      super(NeRV_Net_600,self).__init__()

      self.pos = opt.pos

      ### Postioal Encoding for capturing high frequency regions

      if self.pos == 1:
        print('Use Positional Encoding')

        self.pe_t = PosEncoding(in_features[0],opt.lt)
        self.pe_v = PosEncoding(in_features[1],opt.lv)

        self.init_dim = self.pe_t.out_dim+self.pe_v.out_dim

      else:
        self.init_dim = np.sum(in_features)

      ### MLP for upscaling input dimensions

      self.rb1 = ResBlock(self.init_dim,init_features)
      self.rb2 = ResBlock(init_features,4*init_features)


      ### Upscale 1D vector to 2D image (i.e., 256 X 256)

      self.NeRVBlock1 = NeRVBlock(4*init_features//1,nerv_init_features,5)
      self.NeRVBlock2 = NeRVBlock(nerv_init_features,nerv_init_features//2,5)
      self.NeRVBlock3 = NeRVBlock(nerv_init_features//2,nerv_init_features//4,3)
      self.NeRVBlock4 = NeRVBlock(nerv_init_features//4,nerv_init_features//8,2)
      self.NeRVBlock5 = NeRVBlock(nerv_init_features//8,nerv_init_features//16,2)
      self.NeRVBlock6 = NeRVBlock(nerv_init_features//16,nerv_init_features//32,2)

      self.final = nn.Conv2d(nerv_init_features//32, out_channels, 1, 1)

      self.fc_dim = 4*init_features
        

  def forward(self, coords):

    if self.pos == 1:
      pe_t = self.pe_t(coords[:,0:1])
      pt_v = self.pe_v(coords[:,1:3])

      pe = torch.cat((pe_t,pt_v),dim=1)

    else:
      pe = coords

    #print('pe shape', pe.shape)

    rb1 = self.rb1(pe)

    rb1 = torch.cat((pe,rb1),dim=1)
    rb2 = self.rb2(rb1)

    #print('rb2 shape', rb2.shape)

    rb2_reshape = rb2.view(rb2.size(0), self.fc_dim//1, 1, 1)

    #print('rb2_reshape shape', rb2_reshape.shape) # torch.Size([2, 128, 2, 2])

    NeRVBlock1 = self.NeRVBlock1(rb2_reshape)

    NeRVBlock2 = self.NeRVBlock2(NeRVBlock1)

    NeRVBlock3 = self.NeRVBlock3(NeRVBlock2)

    NeRVBlock4 = self.NeRVBlock4(NeRVBlock3)

    NeRVBlock5 = self.NeRVBlock5(NeRVBlock4)

    NeRVBlock6 = self.NeRVBlock6(NeRVBlock5)

    final = self.final(NeRVBlock6)

    img_out = torch.sigmoid(final)
    return img_out


### 800 x 800
class NeRV_Net_800(nn.Module):

  def __init__(self, in_features, out_channels, init_features, nerv_init_features,opt):
      super(NeRV_Net_800,self).__init__()

      self.pos = opt.pos

      ### Postioal Encoding for capturing high frequency regions

      if self.pos == 1:
        print('Use Positional Encoding')

        self.pe_t = PosEncoding(in_features[0],opt.lt)
        self.pe_v = PosEncoding(in_features[1],opt.lv)

        self.init_dim = self.pe_t.out_dim+self.pe_v.out_dim

      else:
        self.init_dim = np.sum(in_features)

      ### MLP for upscaling input dimensions

      self.rb1 = ResBlock(self.init_dim,init_features)
      self.rb2 = ResBlock(init_features,4*init_features)


      ### Upscale 1D vector to 2D image (i.e., 256 X 256)

      self.NeRVBlock1 = NeRVBlock(4*init_features//4,nerv_init_features,5)
      self.NeRVBlock2 = NeRVBlock(nerv_init_features,nerv_init_features//2,5)
      self.NeRVBlock3 = NeRVBlock(nerv_init_features//2,nerv_init_features//4,2)
      self.NeRVBlock4 = NeRVBlock(nerv_init_features//4,nerv_init_features//8,2)
      self.NeRVBlock5 = NeRVBlock(nerv_init_features//8,nerv_init_features//16,2)
      self.NeRVBlock6 = NeRVBlock(nerv_init_features//16,nerv_init_features//32,2)
      #self.NeRVBlock7 = NeRVBlock(nerv_init_features//32,nerv_init_features//64,2)

      self.final = nn.Conv2d(nerv_init_features//32, out_channels, 1, 1)

      self.fc_dim = 4*init_features
        

  def forward(self, coords):

    if self.pos == 1:
      pe_t = self.pe_t(coords[:,0:1])
      pt_v = self.pe_v(coords[:,1:3])

      pe = torch.cat((pe_t,pt_v),dim=1)

    else:
      pe = coords

    #print('pe shape', pe.shape)

    rb1 = self.rb1(pe)
    rb2 = self.rb2(rb1)

    #print('rb2 shape', rb2.shape)

    rb2_reshape = rb2.view(rb2.size(0), self.fc_dim//4, 2, 2)

    #print('rb2_reshape shape', rb2_reshape.shape) # torch.Size([2, 128, 2, 2])

    NeRVBlock1 = self.NeRVBlock1(rb2_reshape)

    NeRVBlock2 = self.NeRVBlock2(NeRVBlock1)

    NeRVBlock3 = self.NeRVBlock3(NeRVBlock2)

    NeRVBlock4 = self.NeRVBlock4(NeRVBlock3)

    NeRVBlock5 = self.NeRVBlock5(NeRVBlock4)

    NeRVBlock6 = self.NeRVBlock6(NeRVBlock5)

    #NeRVBlock7 = self.NeRVBlock7(NeRVBlock6)

    final = self.final(NeRVBlock6)

    img_out = torch.sigmoid(final)
    return img_out


### 1024 x 1024
class NeRV_Net_1024(nn.Module):

  def __init__(self, in_features, out_channels, init_features, nerv_init_features,opt):
      super(NeRV_Net_1024,self).__init__()

      self.pos = opt.pos

      ### Postioal Encoding for capturing high frequency regions

      if self.pos == 1:
        print('Use Positional Encoding')

        self.pe_t = PosEncoding(in_features[0],opt.lt)
        self.pe_v = PosEncoding(in_features[1],opt.lv)

        self.init_dim = self.pe_t.out_dim+self.pe_v.out_dim

      else:
        self.init_dim = np.sum(in_features)

      ### MLP for upscaling input dimensions

      self.rb1 = ResBlock(self.init_dim,init_features)
      self.rb2 = ResBlock(init_features,4*init_features)


      ### Upscale 1D vector to 2D image (i.e., 256 X 256)

      self.NeRVBlock1 = NeRVBlock(4*init_features//4,nerv_init_features,4)
      self.NeRVBlock2 = NeRVBlock(nerv_init_features,nerv_init_features//2,4)
      self.NeRVBlock3 = NeRVBlock(nerv_init_features//2,nerv_init_features//4,2)
      self.NeRVBlock4 = NeRVBlock(nerv_init_features//4,nerv_init_features//8,2)
      self.NeRVBlock5 = NeRVBlock(nerv_init_features//8,nerv_init_features//16,2)
      self.NeRVBlock6 = NeRVBlock(nerv_init_features//16,nerv_init_features//32,2)
      self.NeRVBlock7 = NeRVBlock(nerv_init_features//32,nerv_init_features//64,2)
      #self.NeRVBlock8 = NeRVBlock(nerv_init_features//64,nerv_init_features//128,2)

      self.final = nn.Conv2d(nerv_init_features//64, out_channels, 1, 1)

      self.fc_dim = 4*init_features
        

  def forward(self, coords):

    if self.pos == 1:
      pe_t = self.pe_t(coords[:,0:1])
      pt_v = self.pe_v(coords[:,1:3])

      pe = torch.cat((pe_t,pt_v),dim=1)

    else:
      pe = coords

    #print('pe shape', pe.shape)

    rb1 = self.rb1(pe)
    rb2 = self.rb2(rb1)

    #print('rb2 shape', rb2.shape)

    rb2_reshape = rb2.view(rb2.size(0), self.fc_dim//4, 2, 2)

    #print('rb2_reshape shape', rb2_reshape.shape) # torch.Size([2, 128, 2, 2])

    NeRVBlock1 = self.NeRVBlock1(rb2_reshape)

    NeRVBlock2 = self.NeRVBlock2(NeRVBlock1)

    NeRVBlock3 = self.NeRVBlock3(NeRVBlock2)

    NeRVBlock4 = self.NeRVBlock4(NeRVBlock3)

    NeRVBlock5 = self.NeRVBlock5(NeRVBlock4)

    NeRVBlock6 = self.NeRVBlock6(NeRVBlock5)

    NeRVBlock7 = self.NeRVBlock7(NeRVBlock6)

    #NeRVBlock8 = self.NeRVBlock8(NeRVBlock7)

    final = self.final(NeRVBlock7)

    img_out = torch.sigmoid(final)
    return img_out





