from dataio import *
import sys
import os
from torch.utils.data import DataLoader
import argparse
import torch
from train import *
from siren import *
from model import *


p = argparse.ArgumentParser()

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
# General training options
p.add_argument('--batch_size', type=int, default=36)
p.add_argument('--lr', type=float, default=2e-5, help='learning rate. default=2e-5')
p.add_argument("--perc-loss", type=str, default="relu1_2",
                help="layer that perceptual loss is computed on (default: relu1_2)")
p.add_argument('--num_epochs', type=int, default=2001,
               help='Number of epochs to train for.')
p.add_argument('--checkpoint', type=int, default=50,
               help='checkpoint is saved.')
p.add_argument('--train', type=str, default='inf', metavar='N',
                    help='train or inference the network')
p.add_argument('--factor', type=int, default=1, metavar='N',
                    help='randomly sample factor*batch_size, only used for SIREN')
p.add_argument('--init', type=int, default=128, metavar='N',
                    help='init features')
p.add_argument('--approach', type=str, default='CNN', metavar='N',
                    help='CNN or SIREN model')
p.add_argument('--lt', type=int, default=4, help='frequency of time')
p.add_argument('--liso', type=int, default=4, help='frequency of view')
p.add_argument('--lv', type=int, default=4, help='frequency of view')
p.add_argument('--pos', type=int, default=1, help='whether use postional encoding')

opt = p.parse_args()
opt.cuda = not opt.no_cuda and torch.cuda.is_available()

def main():
  if opt.train == 'train':
    if opt.approach == 'SIREN':
      Data = ViewSynthesis_SIREN(opt)
      # Model = CoordNet([2,2],3,opt.init)
      # Data.ReadData()
      # Model.cuda()
      # trainSIREN(Model,opt,Data)
    elif opt.approach == 'CNN':
      Data = ViewSynthesis_CNN(opt)
      Model = NeRV_Net_1024([1,2],4,opt.init,1024,opt)
      #Model = NeRV_Net_800([1,2],4,opt.init,1024,opt)
      

      # Data.ReadData()
      # Model.cuda()
      # trainCNN(Model,opt,Data)

      Model.cuda()
      trainCNN(Model,opt)
  elif opt.train == 'inf':
    if opt.approach  == 'SIREN':
      Data = ViewSynthesis_SIREN(opt)
      # inf_SIREN(Data,opt)
    elif opt.approach == 'CNN':
      #Data = ViewSynthesis_CNN(opt)
      inf_CNN(opt)

if __name__== "__main__":
    main()





