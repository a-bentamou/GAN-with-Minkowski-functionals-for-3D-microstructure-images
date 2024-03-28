# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import tifffile
import torch.nn.parallel
import torch.utils.data
import os
import numpy as np
from dcgan_test import Generator
import torch.backends.cudnn as cudnn
from quantimpy import minkowski as mk
import pandas as pd
from scipy.ndimage import median_filter

def minkowski_functionals(img):
    
    #img_mk = get_img(img)
    img_mk = img.astype(dtype=bool)
    mk_metrics = mk.functionals(img_mk,  norm=True)

    return mk_metrics

params = {
    'imsize' : 64,# Spatial size of training images. All images will be resized to this size during preprocessing.
    'nc' : 2,# Number of channles in the training images. For coloured images this is 3.
    'nz' : 100,# Size of the Z latent vector
    'ngf' : 64,# Size of feature maps in the generator. The filtes will be multiples of this.
    'ndf' : 16, # Size of features maps in the discriminator. The filters will be multiples of this.
    'ngpu': 1, # Number of GPUs to be used
    'alpha' : 1,# Size of z space
    'stride' : 0,# Stride on image to crop
    'num_samples' : 20000}# Save step.

cudnn.benchmark = True

# Use GPU is available else use CPU.
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
#device = torch.device('cpu')
print(device, " will be used.\n")

out_dir = 'Results/' 
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

model_path = 'mod_out_new/Generator/netG_best.pth'
checkpoint = torch.load(model_path, map_location=device) 

netG = Generator(params['nz'], params['nc'], params['ngf'], params['ngpu']).to(device)
netG.load_state_dict(checkpoint)
netG = nn.DataParallel(netG)

netG.eval()


vol, surf, leng, eul = 0, 0, 0, 0
# Create an empty DataFrame
df_synthetic = pd.DataFrame(columns=['Volume', 'Surface', 'Length','Euler'])

for i in range(0, params['num_samples']):
    print('Image generation: ', i, '/', params['num_samples'])
    # Create the generator.
    
                    
    noise = torch.FloatTensor(1, params['nz'], params['alpha'], params['alpha'], params['alpha']).normal_(0, 1)
    noise = noise.to(device)

    with torch.no_grad():
        fake = netG(noise)
        #print(fake.shape)
    
    img = fake.cpu()
    img = img.detach().numpy()
    

    W = img.shape[2]
    H = img.shape[3]
    L = img.shape[4]
    edge = params['stride']/2
    edge = int(edge)
    
    phase2 = np.zeros([W, H, L])
    p1 = np.array(img[0][0])
    p2 = np.array(img[0][1])
    phase2[(p2 > p1)] = 255  # binder, white
      
    output_img = np.int_(phase2)
    
    ### Crop edges ###
    nW = W-params['stride']
    nH = H-params['stride']
    nL = L-params['stride']
    #output_image = np.zeros([1, 1, nW, nH, nL])
    output_image = output_img[0+edge:W-edge, 0+edge:H-edge, 0+edge:L-edge]
    
    ### Save cropped image as tiff ###
    new_output = output_image
    new_output = new_output.astype(np.uint8)

    #tifffile.imwrite(str(out_dir)+'/Sample_before_{0}.tif'.format(i), new_output)

    new_output = median_filter(new_output, size=(3, 3, 3))

    mk_metrics = minkowski_functionals(new_output)

    # Create a temporary DataFrame for the current iteration
    temp_df = pd.DataFrame({'Volume': [mk_metrics[0]], 'Surface': [mk_metrics[1]], 'Length': [mk_metrics[2]], 'Euler': [mk_metrics[3]]})

    # Concatenate the temporary DataFrame to the main DataFrame
    df_synthetic = pd.concat([df_synthetic, temp_df], ignore_index=True)

    vol += mk_metrics[0]
    surf += mk_metrics[1]
    leng += mk_metrics[2]
    eul += mk_metrics[3]

    tifffile.imwrite(str(out_dir)+'/Sample_{0}.tif'.format(i), new_output)

print("Average Porosity = ", vol/params['num_samples'])
print("Average Specific Surface Area = ", surf/params['num_samples'])
print("Average Length = ", leng/params['num_samples'])
print("Average Specific Euler Characteristic  = ", eul/params['num_samples'])

# Export the DataFrame to a Tableau data source file (.hyper or .tde)

#df_synthetic.to_csv('Minkowski_functionals_synthetic.csv', index=False)

#df_synthetic["Type"] = 1

df_org = pd.read_csv('Minkowski_functionals_org.csv')
#df_synthetic = pd.read_csv('Minkowski_functionals_synthetic.csv')

from matplotlib.ticker import ScalarFormatter
from pylab import setp
import matplotlib.pyplot as plt

def setBoxColors(bp):
    width = 3
    setp(bp['boxes'][0], color='black', linewidth=width)
    setp(bp['boxes'][1], color='black', linewidth=width)
    
    for i in range(4):
        setp(bp['caps'][i], color='black', linewidth=width*2)
        setp(bp['whiskers'][i], color='black', linewidth=width)
        
    setp(bp['fliers'][0], color='black', linewidth=width)
    setp(bp['fliers'][1], color='black', linewidth=width)
    setp(bp['medians'][0], color='black', linewidth=width, linestyle="--")
    setp(bp['medians'][1], color='black', linewidth=width, linestyle="--")
    
fig, ax = plt.subplots(1, 4, figsize=(48, 12))
for i, prop in enumerate(["Volume", "Surface", "Euler", "Length"]):
    # if prop == "perm":
    #     data = [[perms_sample.flatten()], [perms.flatten()]]
    #     bp = ax[i].boxplot(data)
    #     setBoxColors(bp)    
    # else:
        data = [df_org[prop].values, df_synthetic[prop].values]
        bp = ax[i].boxplot(data)
        setBoxColors(bp)
        
properties = [
    r"Porosity \ [-]",
    r"Specific \ Surface  \ Area \ [\frac{1}{voxel}]",
    r"Specific \ Euler \ Characteristic \ [\frac{1}{voxel^3}]",
    r"Length \ [voxel]"
]
 
        
for j, prop in enumerate(properties):
    ax[j].set_title(r"$"+prop+r"$", fontsize=42, y=1.06)  
    
fig.canvas.draw()

for i in range(4):
    labels = [item.get_text() for item in ax[i].get_xticklabels()]
    labels[0] = r'$Original$'
    labels[1] = r'$Synthetic$'
    ax[i].set_xticklabels(labels, fontsize=36)  
    
    labels_y = [item.get_text() for item in ax[i].get_yticklabels()]
    ax[i].set_yticklabels(labels_y, fontsize=26)
    ax[i].grid()
    #ax[i].xaxis.set_major_formatter(ScalarFormatter())
    ax[i].yaxis.set_major_formatter(ScalarFormatter())

    if i > 0:
        ax[i].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax[i].get_yaxis().get_offset_text().set(va='bottom', ha='left')
        ax[i].yaxis.get_offset_text().set_fontsize(26)

for i, s in enumerate([r'$1\times 10^{-2}$', r'$1\times 10^{-5}$', r'$1\times 10^{-11}$']):
    t = ax[i+1].text(0.01, 1.016, s, transform=ax[i+1].transAxes, fontsize=30)
    t.set_bbox(dict(color='white', alpha=1.0, edgecolor=None))  
    
fig.savefig("minkowski_functionals.png", bbox_extra_artists=None, bbox_inches='tight', dpi=72)