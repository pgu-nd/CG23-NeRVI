import os
import numpy as np
from skimage.transform import resize

def get_mgrid(sidelen, dim=2, s=1,t=0):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[1]:s, :sidelen[0]:s], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / (sidelen[1] - 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[0] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[2]:s, :sidelen[1]:s, :sidelen[0]:s], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / (sidelen[2] - 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[0] - 1)
    elif dim == 4:
        pixel_coords = np.stack(np.mgrid[:sidelen[0]:(t+1), :sidelen[3]:s, :sidelen[2]:s, :sidelen[1]:s], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[3] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
        pixel_coords[..., 3] = pixel_coords[..., 3] / (sidelen[1] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)
    pixel_coords -= 0.5
    pixel_coords *= 2.
    print(pixel_coords.shape)
    pixel_coords = np.reshape(pixel_coords,(-1,dim))
    return pixel_coords

def vorticity_3d(dx,dy,dz,u,v,w):

    dFx_dy = np.gradient(u, dy, axis = 1)
    dFx_dz = np.gradient(u, dz, axis = 2)
    dFy_dx = np.gradient(v, dx, axis = 0)
    dFy_dz = np.gradient(v, dz, axis = 2)
    dFz_dx = np.gradient(w, dx, axis = 0)
    dFz_dy = np.gradient(w, dy, axis = 1)

    rot_x = dFz_dy - dFy_dz
    rot_y = dFx_dz - dFz_dx
    rot_z = dFy_dx - dFx_dy


    vorticity = [rot_x,rot_y,rot_z]

    av = np.sqrt(rot_x**2+rot_y**2+rot_z**2)

    return av


def acceleration(u,v,w,dx,dy,dz,dt):
    ### ref: https://www.youtube.com/watch?v=PpY9k6QEqo0
    dFx_dx = np.gradient(u, dx, axis = 1)
    dFx_dy = np.gradient(u, dy, axis = 2)
    dFx_dz = np.gradient(u, dz, axis = 3)
    dFx_dt = np.gradient(u, dt, axis = 0)

    dFy_dx = np.gradient(v, dx, axis = 1)
    dFy_dy = np.gradient(v, dy, axis = 2)
    dFy_dz = np.gradient(v, dz, axis = 3)
    dFy_dt = np.gradient(v, dt, axis = 0)

    dFz_dx = np.gradient(w, dx, axis = 1)
    dFz_dy = np.gradient(w, dy, axis = 2)
    dFz_dz = np.gradient(w, dz, axis = 3)
    dFz_dt = np.gradient(w, dt, axis = 0)

    acc_x = u*dFx_dx+v*dFx_dy+w*dFx_dz+dFx_dt
    acc_y = u*dFy_dx+v*dFy_dy+w*dFy_dz+dFy_dt
    acc_z = u*dFz_dx+v*dFz_dy+w*dFz_dz+dFz_dt

    av = np.sqrt(np.power(acc_x,2.0) + np.power(acc_y,2.0) + np.power(acc_z,2.0))

    return  av


'''
for n in ['640']:
    for k in range(1,101):
        gt = np.fromfile('/Volumes/Data/temp/half-cylinder-640-magnitude-'+"{:04d}".format(k)+'.dat',dtype='<f')
        gt = 2*(gt-np.min(gt))/(np.max(gt)-np.min(gt))-1
        gt.tofile('/Volumes/Data/temp/half-cylinder-640-magnitude-'+'{:04d}'.format(k)+'.dat',format='<f')
        gt = gt.reshape(80,240,640).transpose()
        gt = resize(gt,(160,60,20),order=3)
        #gt = np.sqrt(np.sum(gt**2,axis=0))
        #print(gt.shape)
        gt = gt.flatten('F')
        gt = np.asarray(gt,dtype='<f')
        gt.tofile('/Volumes/Data/temp/half-cylinder-640-magnitude-low-'+'{:04d}'.format(k)+'.dat',format='<f')
'''


'''

for i in range(99,300):
    gt = np.fromfile('/Volumes/Data/Data-VIS/ScalarData/Earthquake/earthquake/amp'+"{:04d}".format(i)+'.dat',dtype='<f')
    print("==============="+str(i)+'===================')
    print(np.max(gt))
    print(np.min(gt))
    gt.tofile('/Volumes/Data/temp/Earthquake'+"-"+'{:04d}'.format(i-98)+'.dat',format='<f')
'''
