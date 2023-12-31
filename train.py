import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import torch.optim as optim
import time
from model import *
from utils import *
from skimage.io import imsave
from skimage.io import imread
from skimage import data,img_as_float,img_as_int
#import lpips
from vgg19 import *
from siren import *
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

res = 1024

train_data_dir1 = '/afs/crc.nd.edu/group/vis/pgu/../'
train_data_dir2 = '/afs/crc.nd.edu/group/vis/pgu/../'

#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

class TrainingDataSet(Dataset):
    def __init__(self, data_path, data_path2):
        self.data_path = data_path
        self.data_path2 = data_path2

        self.times = [3,6,9,12,15,18,21,24,27,30,
        33,36,39,42,45,48,51,54,57,60,
        63,66,69,72,75,78,81,84,87,90]
        
        

        self.view_parms = []
        self.imgs_path = []
        self.view_parms_original = []

        
       
        for t in self.times:
            print('time step', t)
            t_ = t-self.times[0]/(self.times[-1]-self.times[0])
            t_ -= 0.5
            t_ *= 2.0
            for theta in range(0,180,9):
                theta_ = theta/179.0
                theta_ -= 0.5
                theta_ *= 2.0 ### normaize theta parameter
                for phi in range(0,360,9):
                    phi_ = phi/359.0
                    phi_ -= 0.5
                    phi_ *= 2.0 ### normaize theta parameter
                    self.view_parms.append([t_,phi_,theta_])
                    self.view_parms_original.append([t, phi, theta])
                    #self.imgs_path.append(self.data_path+'/save-iso-'+'{:4f}'.format(iso)+'-theta-'+'{:6f}'.format(theta)+'-phi-'+'{:6f}'.format(phi)+'.png') ### read image
                    if t<50:
                        self.imgs_path.append(self.data_path+'/save-timestep-'+str(t)+'-iso-'+'{:4f}'.format(0.0)+'-theta-'+'{:6f}'.format(theta)+'-phi-'+'{:6f}'.format(phi)+'.png') ### read img_name
                    else:
                        self.imgs_path.append(self.data_path2+'/save-timestep-'+str(t)+'-iso-'+'{:4f}'.format(0.0)+'-theta-'+'{:6f}'.format(theta)+'-phi-'+'{:6f}'.format(phi)+'.png') ### read img_name

            

        
        #print('self.imgs_path',self.imgs_path)
        #print('self.view_parms',self.view_parms)
        #print('self.view_parms_original',self.view_parms_original)

    def __len__(self):
        return len(self.imgs_path)

    def GetMask(self,image):
        mask = np.sum(image,axis=0,keepdims=True)/3.0
        mask[mask!=1.0] = 0
        mask = 1-mask
        return mask

    def __getitem__(self, idx):
        
        valid_idx = idx 
        #print('------------')
        #print('valid_idx', valid_idx)
        
        ### get the images
        img_id = self.imgs_path[valid_idx]
        img_name = os.path.join(self.data_path, img_id)
        #print('img_name', img_name)

        view_parms = self.view_parms[valid_idx]
        #print('view_parms', view_parms)
        view_parms_original = self.view_parms_original[valid_idx]
        #print('view_parms_original', view_parms_original)
        
        img = img_as_float(imread(img_name)) ### read image
        img = img.transpose(2,0,1)
        mask = self.GetMask(img)
        

        img_np = np.asarray(img)
        mask_np = np.asarray(mask)
        view_parms_np = np.asarray(view_parms)

        training_data_img = torch.FloatTensor(img_np)
        training_data_mask = torch.FloatTensor(mask_np)
        training_data_param = torch.FloatTensor(view_parms_np)


       
        return training_data_param, training_data_img, training_data_mask

# def trainSIREN(model,args,dataset):
#     t = 0
#     with open('../Exp/'+'loss-siren.txt','a') as loss:
#         optimizer = optim.Adam(model.parameters(), lr=args.lr,betas=(0.9,0.999),weight_decay=1e-6)
#         criterion = nn.L1Loss()
#         for itera in range(1,args.num_epochs+1):
#             train_loader = dataset.GetTrainingData()
#             x = time.time()
#             print('======='+str(itera)+'========')
#             loss_l1 = 0
#             for batch_idx, (coord,v) in enumerate(train_loader):
#                 t1 = time.time()
#                 if args.cuda:
#                     coord = coord.cuda()
#                     v = v.cuda()
#                 optimizer.zero_grad()
#                 v_pred = model(coord)
#                 l1 = criterion(v_pred.view(-1),v.view(-1))
#                 l1.backward()
#                 loss_l1 += l1.mean().item()
#                 optimizer.step()
            
#             y = time.time()
#             t += y-x
#             print("Epochs "+str(itera)+": loss = "+str(loss_l1))
#             loss.write("Epochs "+str(itera)+": loss = "+str(loss_l1))
#             loss.write('\n')

#             if itera%args.checkpoint == 0 or itera==1:
#                 torch.save(model,'../Exp/SIREN-'+str(itera)+'.pth')

#         loss.write("Time = "+str(t))
#         loss.write('\n')
#         loss.close()

def trainCNN(model,args):
    #device = torch.device("cuda:3" if args.cuda else "cpu")

    with open('../'+'loss.txt','a') as w:
        optimizer = optim.Adam(model.parameters(), lr=args.lr,betas=(0.9,0.999),weight_decay=1e-6)
        #train_loader = dataset.GetTrainingData()
        x1 = time.time()
        train_dataset = TrainingDataSet(train_data_dir1,train_data_dir2)
        data_size = len(train_dataset)
        print('train data size', len(train_dataset))      #args.batch_size
        train_loader = torch.utils.data.DataLoader(dataset =train_dataset, batch_size = args.batch_size, shuffle=True)
        print('num of train batches', len(train_loader))
        y1 = time.time()
        print("Data load time = "+str(y1-x1))


        total_params_original = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        print(f'Params original another computation: {total_params_original}M')

        t = 0
        for itera in range(1,args.num_epochs+1):
            
            x = time.time()
            print('======='+str(itera)+'========')
            loss_l1 = 0
            loss_m = 0
            loss_i = 0

            loss_ssim_ =  0
            
            for batch_idx, (parms,imgs,masks) in enumerate(train_loader):

                parms = parms.cuda()
                imgs = imgs.cuda()
                masks = masks.cuda()

                #print('imgs shape', imgs.shape)
                #print('masks shape', masks.shape)

                optimizer.zero_grad()
                pred = model(parms)

                loss_mask =  torch.mean(torch.abs(pred[:,3:4:,:,]-masks))
                loss_imgs = torch.sum(torch.abs(masks*(pred[:,0:3:,:,]-imgs)))/torch.sum(masks)

                loss_ssim=  (1 - ssim(pred[:,0:3:,:,], imgs, data_range=1, size_average=True))

                l1 = 0.7*(loss_mask+loss_imgs) + 0.3*loss_ssim
                l1.backward()

                loss_l1 += l1.mean().item()
                loss_m += loss_mask.mean().item()
                loss_i += loss_imgs.mean().item()
                loss_ssim_ += loss_ssim.mean().item()
                optimizer.step()
            
            y = time.time()
            print("Per Epoch Time = "+str(y-x))
            t += y-x
            print("Epochs "+str(itera)+": loss image = "+str(loss_i))
            print("Epochs "+str(itera)+": loss mask = "+str(loss_m))
            print("Epochs "+str(itera)+": loss ssim = "+str(loss_ssim_))
            w.write("Epochs "+str(itera)+": loss = "+str(loss_l1))
            w.write('\n')

            if itera%args.checkpoint == 0 or itera==5:
                torch.save(model.state_dict(),'../CNN-Pos-Encode-'+str(args.pos)+'-epoch-'+str(itera)+'.pth')
            # if itera%args.checkpoint == 0 or itera==5:
            #     model.eval()
            #     ### testing 
            #     count = 1
                
                        
            #     times = [3,6,9,12,15,18,21,24,27,30,
            #             33,36,39,42,45,48,51,54,57,60,
            #             63,66,69,72,75,78,81,84,87,90]
                
            #     for t in times:
            #         print('time step', t)
            #         t_ = t-times[0]/(times[-1]-times[0])
            #         t_ -= 0.5
            #         t_ *= 2.0
            #         for theta in range(0,180,9):
            #             theta_ = theta/179.0
            #             theta_ -= 0.5
            #             theta_ *= 2.0 ### normaize theta parameter
            #             for phi in range(0,360,9):
            #                 phi_ = phi/359.0
            #                 phi_ -= 0.5
            #                 phi_ *= 2.0 ### normaize theta parameter
            #                 parms = [t_,phi_,theta_]
            #                 parms = np.asarray(parms)
            #                 parms = torch.FloatTensor(parms)
            #                 parms = torch.unsqueeze(parms,0)
            #                 #print(parms.size())
            #                 parms = parms.cuda()
            #                 with torch.no_grad():
            #                     results = model(parms)[0].detach().cpu().numpy()
            #                     img = results[0:3:,:,]
            #                     mask = results[3:4:,:,]
            #                     mask[mask>=0.5] = 1
            #                     mask[mask<0.5] = 0
            #                     img = mask*img+(1-mask)*np.ones((3,res,res))
            #                 img *= 255
            #                 img = img.transpose(1,2,0)
            #                 img = img.astype(np.uint8)
            #                 if t<50:
            #                         direc = '../Result' + '/test_iter' + str(itera)   
            #                         directory = os.path.join(direc) 
            #                         if not os.path.exists(directory):
            #                             os.makedirs(directory)
            #                         imsave(direc+'/save-timestep-'+str(t)+'-iso-'+'{:4f}'.format(0.0)+'-theta-'+'{:6f}'.format(theta)+'-phi-'+'{:6f}'.format(phi)+'.png',img)
            #                         count += 1
            #                 else:
            #                     direc = '../Result' + '/test_iter' + str(itera) + '_2'
            #                     directory = os.path.join(direc) 
            #                     if not os.path.exists(directory):
            #                         os.makedirs(directory)
            #                     imsave(direc+'/save-timestep-'+str(t)+'-iso-'+'{:4f}'.format(0.0)+'-theta-'+'{:6f}'.format(theta)+'-phi-'+'{:6f}'.format(phi)+'.png',img)
            #                     count += 1
            #         #idx = idx +1
                model.train()
        w.write("Time = "+str(t))
        w.write('\n')
        w.close()


def adjust_lr(args, optimizer, epoch):
    lr = args.lr * (0.5 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# def inf_SIREN(dataset,args):
#     model =  CoordNet(4,3,args.init)
#     model.load_state_dict(torch.load('../Exp/'+str(args.num_epochs)+'.pth'))
#     model.cuda()
#     coords = get_mgrid([res,res],dim=2)
#     theta = np.zeros((res*res,1))
#     phi = np.zeros((res*res,1))
#     count = 1
#     for p in range(0,180,9):
#         phi_ = p/179.0
#         phi_ -= 0.5
#         phi_ *= 2.0
#         phi.fill(phi_)
#         for t in range(0,360,9):
#             theta_ = t/359.0
#             theta_ -= 0.5
#             theta_ *= 2.0
#             theta.fill(theta_)
#             print([theta_,phi_])
#             train_loader = DataLoader(dataset=torch.FloatTensor(np.concatenate((coords,theta,phi),axis=1)), batch_size=args.batch_size, shuffle=False)
#             r = []
#             g = []
#             b = []
#             for batch_idx, coord in enumerate(train_loader):
#                 coord = coord.cuda()
#                 with torch.no_grad():
#                     v_pred = model(coord).permute(1,0)
#                     r += list(v_pred[0].view(-1).detach().cpu().numpy())
#                     g += list(v_pred[1].view(-1).detach().cpu().numpy())
#                     b += list(v_pred[2].view(-1).detach().cpu().numpy())
#             r = np.asarray(r)
#             g = np.asarray(g)
#             b = np.asarray(b)
#             r = r.reshape(res,res).transpose()
#             g = g.reshape(res,res).transpose()
#             b = b.reshape(res,res).transpose()
#             img = np.asarray([r,g,b])
#             img /= 2.0
#             img += 0.5
#             img *= 255
#             img = img.transpose(1,2,0)
#             img = img.astype(np.uint8)
#             imsave('../Result/'+'{:05d}'.format(count)+'.png',img)
#             count += 1


def inf_CNN(args):
    for itera in [50]:
    #itera = 400
        model =  NeRV_Net_1024([1,2],4,args.init,1024,args) 
        model.load_state_dict(torch.load('/afs/crc.nd.edu/user/p/pgu/Research/../CNN-Pos-Encode-1-epoch-500.pth'))
        model.cuda()

        quant_bit = 9
        cur_ckt = model.state_dict()
        from dahuffman import HuffmanCodec
        quant_weitht_list = []
        for k,v in cur_ckt.items():
            large_tf = (v.dim() in {2,4} and 'bias' not in k)
            #print('large_tf', large_tf)
            quant_v, new_v = quantize_per_tensor(v, quant_bit, quant_axis if large_tf else -1)
            valid_quant_v = quant_v[v!=0] # only include non-zero weights
            quant_weitht_list.append(valid_quant_v.flatten())
            cur_ckt[k] = new_v
        cat_param = torch.cat(quant_weitht_list)
        input_code_list = cat_param.tolist()
        unique, counts = np.unique(input_code_list, return_counts=True)
        num_freq = dict(zip(unique, counts))

        # generating HuffmanCoding table
        codec = HuffmanCodec.from_data(input_code_list)
        sym_bit_dict = {}
        for k, v in codec.get_code_table().items():
            sym_bit_dict[k] = v[0]
        total_bits = 0
        for num, freq in num_freq.items():
            total_bits += freq * sym_bit_dict[num]
        avg_bits = total_bits / len(input_code_list)    
        # import pdb; pdb.set_trace; from IPython import embed; embed()       
        encoding_efficiency = avg_bits / quant_bit
        print_str = f'Entropy encoding efficiency for bit {quant_bit}: {encoding_efficiency}'
        print(print_str)      
        model.load_state_dict(cur_ckt)


        count = 1
                
                        
        times = [3,6,9,12,15,18,21,24,27,30,
        33,36,39,42,45,48,51,54,57,60,
        63,66,69,72,75,78,81,84,87,90]

        x1 = time.time()
        for t in [4]:
            print('time step', t)
            t_ = t-times[0]/(times[-1]-times[0])
            t_ -= 0.5
            t_ *= 2.0
            for theta in [4]:#range(0,20,1):
                theta_ = theta/179.0
                theta_ -= 0.5
                theta_ *= 2.0 ### normaize theta parameter
                for phi in [4]:#range(0,40,1):
                    phi_ = phi/359.0
                    phi_ -= 0.5
                    phi_ *= 2.0 ### normaize theta parameter
                    parms = [t_,phi_,theta_]
                    parms = np.asarray(parms)
                    parms = torch.FloatTensor(parms)
                    parms = torch.unsqueeze(parms,0)
                    #print(parms.size())
                    parms = parms.cuda()
                    with torch.no_grad():
                        results = model(parms)[0].detach().cpu().numpy()
                        img = results[0:3:,:,]
                        mask = results[3:4:,:,]
                        mask[mask>=0.5] = 1
                        mask[mask<0.5] = 0
                        img = mask*img+(1-mask)*np.ones((3,res,res))
                    img *= 255
                    img = img.transpose(1,2,0)
                    img = img.astype(np.uint8)
                    
                    direc = '../results/' #+ '/change_time_theta_phi' 
                    directory = os.path.join(direc) 
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    imsave(direc+'/save-timestep-'+str(t)+'-iso-'+'{:4f}'.format(0.0)+'-theta-'+'{:6f}'.format(theta)+'-phi-'+'{:6f}'.format(phi)+'.png',img)
                    count += 1
                   
        y1 = time.time()
        print("Infer Time = "+str(y1-x1))






