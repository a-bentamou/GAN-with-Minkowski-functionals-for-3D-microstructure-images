# -*- coding: utf-8 -*-

import argparse    
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import random
import os
import h5py
from dataset_test import HDF5Dataset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from quantimpy import minkowski as mk
import tifffile
from dcgan_test import Generator, Discriminator

# Set random seed for reproducibility.
seed = 500
random.seed(seed)
torch.manual_seed(seed)
print("Random Seed: ", seed)

def get_img(img):
  L = img.shape[1]
  H = img.shape[2]
  W = img.shape[3]

  phase2 = np.zeros([L, H, W])
  p1 = np.array(img[0])
  p2 = np.array(img[1])
  phase2[(p2 > p1)] = 255  # binder, white

  output_img = np.int_(phase2)

  new_output = output_img.astype(np.uint8)
 
  return new_output

def minkowski_functionals(img, device):
    img = img.cpu().detach().numpy() #.astype(dtype=bool) #[0,0,:,:,:]
    
    b_size = img.shape[0]
    mk_metrics = []
    for m in range(b_size):
        img_mk = get_img(img[m])
        img_mk = img_mk.astype(dtype=bool)
        mk_metrics.append(mk.functionals(img_mk,  norm=True))
    return torch.tensor(mk_metrics, device=device)

def hdf5totiff(img, filename, nphases=2):
  img = img.detach().numpy()

  L = img.shape[2]
  H = img.shape[3]
  W = img.shape[4]

  phase2 = np.zeros([L, H, W])
  p1 = np.array(img[0][0])
  p2 = np.array(img[0][1])
  phase2[(p2 > p1)] = 255  # binder, white

  output_img = np.int_(phase2)

  new_output = output_img.astype(np.uint8)
  tifffile.imwrite(filename, new_output)

  return torch.tensor(new_output)

def filesave(filename, img, nphases=2):
    img = img.detach().numpy()
    img = img.astype(np.uint8)
    _, _, Length, Height, Width = img.shape
    img_mat = np.zeros([Length, Height, Width], dtype= np.uint8)

    if nphases ==2:
      material0 = 0 # corresponding to layer 0
      material1 = 255 # corresponding to layer 1
      for i in range(0, Length):
        for j in range(0, Height):
          for k in range(0, Width):
              if img[0, 0, i, j, k] > 127:
                img_mat[i, j, k] = material1
              else:
                img_mat[i, j, k] = material0
    
    elif nphases==3:
      material0 = 0 # corresponding to layer 0
      material1 = 127 # corresponding to layer 1
      material2 = 255 # corresponding to layer 2
      for i in range(0, Length):
        for j in range(0, Height):
          for k in range(0, Width):
              if img[0, 0, i, j, k] < 85:
                img_mat[i, j, k] = material0
              elif img[0, 0, i, j, k] > 170:
                img_mat[i, j, k] = material2
              else:
                img_mat[i, j, k] = material1

    tifffile.imwrite(filename, img_mat)
    return 0

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default='', help='input dataset file')
parser.add_argument('--out_dir_hdf5', default='', help= 'output file for generated images')
parser.add_argument('--out_dir_model', default='', help= 'output file for model')
parser.add_argument('--workers', type=int, default=0, help='number of workers')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--bsize', default=32, help='batch size during training')
parser.add_argument('--imsize', default=64, help='size of training images')
parser.add_argument('--nc', default=2, help='number of channels')
parser.add_argument('--nz', default=100, help='size of z latent vector')
parser.add_argument('--ngf', default=64, help='size of feature maps in generator')
parser.add_argument('--ndf', default=16, help='size of feature maps in discriminator')
parser.add_argument('--nepochs', default=200, help='number of training epochs')
parser.add_argument('--lr', default=0.0002, help='learning rate for optimisers')
parser.add_argument('--beta1', default=0.5, help='beta1 hyperparameter for Adam optimiser')
parser.add_argument('--save_epoch', default=2, help='step for saving paths')
parser.add_argument('--sample_interval', default=50, help='output image step')

opt = parser.parse_args()
cudnn.benchmark = True

nphases = 2
electrode = 'O2'

opt.dataroot = '../data/LSCF_750_750_750_20nm_Pores0_stride_64/HDF5'
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = int(opt.nc)
workers = int(opt.workers)

vol_density_target = 0.4585 # average porosity of training images
surf_density_target = 0.0144 # average surface density of training images

# Use GPU is available else use CPU.
device = torch.device("cuda:0" if(torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device, " will be used.\n")

# Get the data.
dataset = HDF5Dataset(opt.dataroot,
                          input_transform=transforms.Compose([
                          transforms.ToTensor()
                          ]))

dataloader = torch.utils.data.DataLoader(dataset,
        batch_size=opt.bsize,
        shuffle=True, num_workers=workers)

opt.out_dir_hdf5 = 'img_out_new/' 
opt.out_dir_model = 'mod_out_new/'
log_dir = 'runs/'
if not os.path.exists(opt.out_dir_hdf5):
    os.makedirs(opt.out_dir_hdf5)
if not os.path.exists(opt.out_dir_model):
    os.makedirs(opt.out_dir_model + '/Generator')
    os.makedirs(opt.out_dir_model + '/Discriminator')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

###############################################
# Functions to be used:
###############################################
# weights initialisation
def weights_init(w):
    """
    Initializes the weights of the layer, w.
    """
    classname = w.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(w.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(w.weight.data, 1.0, 0.02)
        nn.init.constant_(w.bias.data, 0)

# save tensor into hdf5 format
def save_hdf5(tensor, filename):

    tensor = tensor.cpu()
    ndarr = tensor.mul(255).byte().numpy()
    with h5py.File(filename, 'w') as f:
        f.create_dataset('data', data=ndarr, dtype="i8", compression="gzip")

###############################################

fixed_noise = torch.randn((1, nz, 1, 1, 1)).cuda()

# Create the generator
netG = Generator(nz, nc, ngf, ngpu).to(device)

# images = next(iter(dataloader))
# print(images.shape)
# grid = torchvision.utils.make_grid(images[:,:,:,0])
tb = SummaryWriter(log_dir)
# tb.add_image("images", grid)
    
# z = torch.randn(1, CNN.nz, 1, 1)
# tb.add_graph(netG, z)

if('cuda' in str(device)) and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

netG.apply(weights_init)
print(netG)

# Create the discriminator
netD = Discriminator(nz, nc, ndf, ngpu)
#tb.add_graph(netD, images)

netD = Discriminator(nz, nc, ndf, ngpu).to(device)

if('cuda' in str(device)) and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

netD.apply(weights_init)
print(netD)

# Binary Cross Entropy loss function.
criterion = nn.BCELoss()
mse = nn.MSELoss()

if(device.type == 'cuda'):
    netD.cuda()
    netG.cuda()
    criterion.cuda()
    mse.cuda

real_label = 0.9 # lable smoothing epsilon = 0.1
fake_label = 0

# Optimizer for the discriminator.
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
# Optimizer for the generator.
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

schedulerD = optim.lr_scheduler.MultiStepLR(optimizerD, milestones=[10,50,100,150], gamma=0.1)
schedulerG = optim.lr_scheduler.MultiStepLR(optimizerG, milestones=[10,50,100,150], gamma=0.1)

# schedulerD = optim.lr_scheduler.StepLR(optimizerD, step_size=10, gamma=0.1)
# schedulerG = optim.lr_scheduler.StepLR(optimizerG, step_size=10, gamma=0.1)

# Stores generated images as training progresses.
img_list = []
# Stores generator losses during training.
G_losses = []
# Stores discriminator losses during training.
D_losses = []

iters = 0
W = opt.imsize
H = opt.imsize

print("Starting Training Loop...")
#print("-"*25)
best_loss = np.inf
early_stop_count = 0
patience = 200
for epoch in range(opt.nepochs):
    print("----------------------------- Epoch : %d -----------------------------------------" % epoch)
    D_loss_epoch = []
    G_loss_epoch = []
    real_acc_epoch = []
    fake_acc_epoch = []

    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximise log(D(x)) + log(1 - D(G(z)))
        ###########################
        netD.zero_grad()
        
        #real_data = data[0].to(device)
        real_data = data.to(device)
        #print('real', real_data.shape)
        grid_real = vutils.make_grid(data[:,:,0,:,:])
        tb.add_image("real image", grid_real, iters)

        b_size = real_data.size(0)

        mk_real = torch.tensor([[vol_density_target, surf_density_target]] * b_size, device=device)
        
        label = torch.full((b_size,), real_label, device=device)
        
        output = netD(real_data).view(-1)
        #output from D will be of size (b_size, 1, 1, 1, 1), with view(-1) we
        #reshape the output to have size (b_size)
        #print(output.shape)
        errD_real = criterion(output, label) # log(D(x))
        errD_real.backward()
        D_x = output.mean().item()
        
        noise = torch.randn(b_size, nz, 1, 1, 1, device=device)
        fake_data = netG(noise)
        label.data.fill_(fake_label)
        output = netD(fake_data.detach()).view(-1) # detach() no need for gradients
        errD_fake = criterion(output, label) # log(1 - D(G(z)))
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()
        
        ############################
        # (2) Update G network: maximise log(D(G(z)))
        ###########################

        gen_it = 1
        while gen_it != 0:
            netG.zero_grad()
            label.data.fill_(real_label)
            noise = torch.randn(b_size, nz, 1, 1, 1, device=device)
            fake_data = netG(noise)
            output = netD(fake_data).view(-1)
            errC = criterion(output,label) # log(D(G(z)))
            mk_fake = minkowski_functionals(fake_data, device) 
            errM = mse(mk_fake[:,:2], mk_real)   #only porosity and surface area density
            errG = errC + errM
            errG.backward()
            D_G_z2 = output.data.mean().item()
            optimizerG.step()
            gen_it -= 1

        
        tb.add_scalar("Generator Loss per iteration", errG.item(), iters)

        grid_fake = vutils.make_grid(fake_data.data[:,:,0,:,:])
        tb.add_image("fake image", grid_fake, iters)
        
        D_loss_epoch.append(errD)
        G_loss_epoch.append(errG)

        real_acc_epoch.append(D_x)
        fake_acc_epoch.append(D_G_z1)

        iters += 1
        
        # Check progress of training.
        #if i%50 == 0:
        print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                % (epoch, opt.nepochs, i, len(dataloader),
                    errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save the losses for plotting.
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        
        # batches_done = epoch * len(dataloader) + i
        # if batches_done % opt.sample_interval == 0:
        #     #fake = netG(noise)
        #     save_hdf5(fake_data.data, str(opt.out_dir_hdf5)+'/fake_{0}.hdf5'.format(batches_done)) 

    print("----------------------------- End of Epoch : %d -----------------------------------------" % epoch)
    print(
        "[Epoch %d/%d] [Discriminator mean loss: %f] [Generator mean loss: %f] [Realimage mean accuracy: %.4f] [Generated image mean accuracy: %.4f]"
            % (epoch, opt.nepochs, sum(D_loss_epoch)/float(len(D_loss_epoch)), sum(G_loss_epoch)/float(len(G_loss_epoch)), 
            sum(real_acc_epoch)/float(len(real_acc_epoch)), sum(fake_acc_epoch)/float(len(fake_acc_epoch)))
        )

    tb.add_scalar("Discriminator mean Loss", sum(D_loss_epoch)/float(len(D_loss_epoch)), epoch)
    tb.add_scalar("Generator mean Loss", sum(G_loss_epoch)/float(len(G_loss_epoch)), epoch)
    tb.add_scalar("Real image mean accuracy", sum(real_acc_epoch)/float(len(real_acc_epoch)), epoch)
    tb.add_scalar("Generated image mean accuracy", sum(fake_acc_epoch)/float(len(fake_acc_epoch)), epoch)

    netG.eval()
    with torch.no_grad():
        fixed_img_output = netG(fixed_noise).detach().cpu()
        filesave(str(opt.out_dir_hdf5)+'/fake_{0}.tif'.format(epoch), fixed_img_output.data, nphases)
        save_hdf5(fixed_img_output.data, str(opt.out_dir_hdf5)+'/fake_{0}.hdf5'.format(epoch)) 
        new_output = hdf5totiff(fixed_img_output.data, str(opt.out_dir_hdf5)+'/fake_h5_{0}.tif'.format(epoch))

        grid = vutils.make_grid(fixed_img_output.data[:,:,0,:,:])
        tb.add_image("fake images per epoch", grid, epoch)

        grid = vutils.make_grid(new_output.data[0,:,:])
        tb.add_image("fake images hdf5", grid, epoch)
    netG.train()
    # plt.figure()
    # plt.axis("off")
    # plt.title("Epoch {}".format(epoch))
    # plt.imshow(
    # np.transpose(torchvision.utils.make_grid(fixed_img_output, padding=5, normalize=True).cpu(), (1, 2, 0)))
    # plt.savefig('./GAN_output/' + str(epoch).zfill(3) + '.png')
    # plt.close()

        # torchvision.utils.save_image(fixed_img_output.cpu().data,
        # './GAN_output/' + str(epoch + 1) + '.png', nrow=8)

        # Save model every epoch
    # save_model(G, 'models/Generator/model_{}.pth.tar'.format(int(epoch)))
    # save_model(G, 'models/Discriminator/model_{}.pth.tar'.format(int(epoch)))
###############################################################################            
# This section can be included for saving the images produced at each timestep
# It increases the processing time more than three times, but is useful to view
# if the algorithm is producing reasonable images
#            
    # if batches_done % opt.sample_interval == 0:
    #     output_data = fake_data.argmax(dim=1)
    #        # output_data will have dimensions of [b_size, imsize, imsize] since the channels are already eliminated by the 
    #        # argmax function
    #     output_img = torch.zeros([b_size, 1, W, H])
    #     for m in range(0, b_size):
    #         for n in range(0, W):
    #             for l in range(0, H):
    #                 if output_data[m, n, l] == 0:
    #                     output_img[m, 0, n, l] = 0.0
    #                 elif output_data[m, n, l] == 1:
    #                     output_img[m, 0, n, l] = 127.0 # 127.0 for three phase data, 255.0 for two phase
    #                 elif output_data[m, n, l] == 2:
    #                     output_img[m, 0, n, l] = 255.0
    #     save_image(output_img.data[:25],  str(opt.out_dir_hdf5)+'/%d.png' % batches_done, nrow=5, normalize=True) #save_image
    #     print(output_img.shape)

###############################################################################
           
    if epoch % opt.save_epoch == 0:    
        # Save checkpoints
        torch.save(netG.state_dict(), str(opt.out_dir_model)+'/Generator/netG_epoch_{}.pth'.format(epoch))
        torch.save(netD.state_dict(), str(opt.out_dir_model)+'/Discriminator/netD_epoch_{}.pth'.format(epoch))
        torch.save(optimizerG.state_dict(), str(opt.out_dir_model)+'/Generator/optimG_epoch_{}.pth'.format(epoch))
        torch.save(optimizerD.state_dict(), str(opt.out_dir_model)+'/Discriminator/optimD_epoch_{}.pth'.format(epoch))
    
    schedulerD.step()
    schedulerG.step()

    G_mean_loss = sum(G_loss_epoch)/float(len(G_loss_epoch))
    if G_mean_loss < best_loss:
      best_loss = G_mean_loss
      torch.save(netG.state_dict(), str(opt.out_dir_model)+'/Generator/netG_best.pth')
      torch.save(netD.state_dict(), str(opt.out_dir_model)+'/Discriminator/netD_best.pth')
      torch.save(optimizerG.state_dict(), str(opt.out_dir_model)+'/Generator/optimG_best.pth')
      torch.save(optimizerD.state_dict(), str(opt.out_dir_model)+'/Discriminator/optimD_best.pth')
      early_stop_count = 0
    else:
      early_stop_count += 1
      if early_stop_count >= patience:
        print(f"Training loss has not improved for {patience} epochs. Stopping training.")
        break
tb.close()

# Save the final trained model
# torch.save(netG.state_dict(), str(opt.out_dir_model)+'/netG_final.pth'.format(epoch))
# torch.save(netD.state_dict(), str(opt.out_dir_model)+'/netD_final.pth'.format(epoch))
# torch.save(optimizerG.state_dict(), str(opt.out_dir_model)+'/optimG_final.pth'.format(epoch))
# torch.save(optimizerD.state_dict(), str(opt.out_dir_model)+'/optimD_final.pth'.format(epoch))
