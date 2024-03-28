# -*- coding: utf-8 -*-

import tifffile
import numpy as np
import h5py
import argparse
import os
import torch
from torch import Tensor

parser = argparse.ArgumentParser()
parser.add_argument('--input_direct', default='', help='directory where hdf5 files are saved')
parser.add_argument('--output_direct', default='', help='directory to output tiff')
parser.add_argument('--indx_data', default=0, help='index of data to be used')
parser.add_argument('--sample_step', default=1, help='step between samples')
parser.add_argument('--name', default='', help='file name')

opt = parser.parse_args()

opt.input_direct = './data/LSCF_750_750_750_20nm_'
opt.output_direct = opt.input_direct  + '/images_tiff'
opt.name = 'result'

os.makedirs(opt.output_direct, exist_ok=True)
indx_data = opt.indx_data
step = opt.sample_step

for a in range(0, 20):
    fake_files = str(opt.input_direct)+'/'+'LSCF_750_750_750_20nm_'+str(indx_data)+'.hdf5'  
    f = h5py.File(fake_files, 'r')
    data = f['data'][()]
    fake_data = Tensor(data)

    b_size = 1

    W = fake_data.shape[1]
    H = fake_data.shape[2]
    L = fake_data.shape[3]
    output_data = fake_data #.argmax(dim=1)
    # output_data will have dimensions of [b_size, imsize, imsize] since the channels
    # are eliminated by the argmax function▲
    output_img = torch.zeros([b_size, 1, W, H, L])
    for m in range(0, b_size):
        for n in range(0, W):
            for l in range(0, H):
                for o in range(0, L):
                    if output_data[m, n, l, o] == 0:
                        output_img[m, 0, n, l, o] = 0.0
                    elif output_data[m, n, l, o] == 1:
                        output_img[m, 0, n, l, o] = 255.0
    
    output = output_img.numpy()
    output = output.astype(np.uint8) 
    print(output.shape)
    
    """
    MAKE LOOP IN BATCH SIZE OF OUTPUTS
    """
    
    # For 3D just first sample of batch
    image = output[0, 0, :, :, :]
    tifffile.imsave(str(opt.output_direct)+'/'+str(opt.name)+'_'+str(indx_data)+'.tif', image)
    
    batch_size = output.shape[0]
    
    indx_data += step
    
    