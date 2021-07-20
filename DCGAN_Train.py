# -*- coding: utf-8 -*-
from __future__ import print_function
from Utilities.SaveAnimation import Video
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
from IPython.display import HTML
import time
import pandas as pd
import pickle

#Get GPU Information
print("CUDA is available: {}".format(torch.cuda.is_available()))
print("CUDA Device Count: {}".format(torch.cuda.device_count()))
print("CUDA Device Name: {}".format(torch.cuda.get_device_name(0)))

#Location of Training Data
spectra_path = '/home/ramanlab/Documents/MachineLearning/GAN/_TrainingData/Data/absorptionData_HybridGAN.csv'

#Location to Save Models (Generators and Discriminators)
save_dir = '/home/ramanlab/Documents/MachineLearning/GAN/_TrainingData/Data/'

#Root directory for dataset (images must be in a subdirectory within this folder)
img_path = '/home/ramanlab/Documents/MachineLearning/GAN/_TrainingData/Images_HybridGAN_Color'

def Excel_Tensor(spectra_path):
    # Location of excel data
    excelData = pd.read_csv(spectra_path, header = 0, index_col = 0)    
    excelDataSpectra = excelData.iloc[:,:800] #index until the last point of the spectra in the Excel file
    excelDataTensor = torch.tensor(excelDataSpectra.values).type(torch.FloatTensor)
    return excelData, excelDataSpectra, excelDataTensor

excelData, excelDataSpectra, excelDataTensor = Excel_Tensor(spectra_path)

f = open('training_log.txt','w')
start_time = time.time()
local_time = time.ctime(start_time)
print('Start Time = %s' % local_time)
print('Start Time = %s' % local_time, file=f)

#Does not truncate tensor contents (Can set "Default")
torch.set_printoptions(profile="full")

#Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

#Number of workers for dataloader (for Windows workers must = 0, for reference: https://github.com/pytorch/pytorch/issues/2341)
workers = 1 

#Batch size during training
batch_size = 16

#Spatial size of training images. All images will be resized to this size using a transformer.
image_size = 64

#Number of channels in the training images. For color images this is 3
nc = 3 

#Size of z latent vector (i.e. size of generator input)
latent = 400
gan_input = excelDataTensor.size()[1] + latent

#Size of feature maps in generator
ngf = 128

#Size of feature maps in discriminator
ndf = 64

#Number of training epochs
num_epochs = 1

#Learning rate for optimizers
lr = 0.0001

#Beta1 hyperparam for Adam optimizers
beta1 = 0.5

#Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

#Create the dataset. Use "dataset.imgs" to show filenames
dataset = dset.ImageFolder(root=img_path,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize([0.5],[0.5]) 
                           ]))
#Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=False, num_workers=workers)

#Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

#Custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

#Generator Code
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu            
        self.conv1 = nn.ConvTranspose2d(gan_input, ngf * 8, 6, 1, 0, bias=False)
        self.conv2 = nn.BatchNorm2d(ngf * 8)
        self.conv3 = nn.ReLU(True)
        # state size. (ngf*8) x 6 x 6
        self.conv4 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 6, 2, 2, bias=False)
        self.conv5 = nn.BatchNorm2d(ngf * 4)
        self.conv6 = nn.ReLU(True)
        # state s7ze. (ngf*4) x 12 x 12
        self.conv7 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 6, 2, 4, bias=False)
        self.conv8 = nn.BatchNorm2d(ngf * 2)
        self.conv9 = nn.ReLU(True)
        # state size. (ngf*2) x 20 x 20
        self.conv10 = nn.ConvTranspose2d(ngf * 2, ngf, 6, 2, 5, bias=False)
        self.conv11 = nn.BatchNorm2d(ngf)
        self.conv12 = nn.ReLU(True)
        # state size. (ngf) x 36 x 36
        self.conv13 = nn.ConvTranspose2d(ngf, nc, 6, 2, 4, bias=False)
        self.conv14 = nn.Tanh()
        # state size. (nc) x 68 x 68

    def forward(self, input):
        imageOut = input
        imageOut = self.conv1(imageOut)
        imageOut = self.conv2(imageOut)
        imageOut = self.conv3(imageOut)
        imageOut = self.conv4(imageOut)
        imageOut = self.conv5(imageOut)
        imageOut = self.conv6(imageOut)
        imageOut = self.conv7(imageOut)
        imageOut = self.conv8(imageOut)
        imageOut = self.conv9(imageOut)
        imageOut = self.conv10(imageOut)
        imageOut = self.conv11(imageOut)
        imageOut = self.conv12(imageOut)
        imageOut = self.conv13(imageOut)
        imageOut = self.conv14(imageOut)               
        return imageOut

#Create the generator
netG = Generator(ngpu).to(device)

#Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

#Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
netG.apply(weights_init)

#Print the model
print(netG)

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.l1 = nn.Linear(800, image_size*image_size*nc, bias=False)           
        self.conv1 = nn.Conv2d(2*nc, ndf, 6, 2, 4, bias=False) 
        self.conv2 = nn.LeakyReLU(0.2, inplace=True)
        # state size. (ndf) x 36 x 36
        self.conv3 = nn.Conv2d(ndf, ndf * 2, 6, 2, 5, bias=False)
        self.conv4 = nn.BatchNorm2d(ndf * 2)
        self.conv5 = nn.LeakyReLU(0.2, inplace=True)
        # state size. (ndf*2) x 20 x 20
        self.conv6 = nn.Conv2d(ndf * 2, ndf * 4, 6, 2, 4, bias=False)
        self.conv7 = nn.BatchNorm2d(ndf * 4)
        self.conv8 = nn.LeakyReLU(0.2, inplace=True)
        # state size. (ndf*4) x 12 x 12
        self.conv9 = nn.Conv2d(ndf * 4, ndf * 8, 6, 2, 2, bias=False)
        self.conv10 = nn.BatchNorm2d(ndf * 8)
        self.conv11 = nn.LeakyReLU(0.2, inplace=True)
        # state size. (ndf*8) x 6 x 6
        self.conv12 = nn.Conv2d(ndf * 8, 1, 6, 1, 0, bias=False)
        self.conv13 = nn.Sigmoid()

    def forward(self, input, label):
        x1 = input
        x2 = self.l1(label)
        x2 = x2.reshape(int(b_size/ngpu),nc,image_size,image_size) 
        combine = torch.cat((x1,x2),1)
        combine = self.conv1(combine)
        combine = self.conv2(combine)
        combine = self.conv3(combine)
        combine = self.conv4(combine)
        combine = self.conv5(combine)
        combine = self.conv6(combine)
        combine = self.conv7(combine)
        combine = self.conv8(combine)
        combine = self.conv9(combine)
        combine = self.conv10(combine)
        combine = self.conv11(combine)
        combine = self.conv12(combine)
        combine = self.conv13(combine)
        return combine

#Create the Discriminator
netD = Discriminator(ngpu).to(device)

#Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

#Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
netD.apply(weights_init)

#Print the model
print(netD)

#Initialize BCELoss function
criterion = nn.BCELoss()

#Create batch of latent vectors that we will use to visualize the progression of the generator
testTensor = torch.Tensor()
for i in range (100):
    fixed_noise1 = torch.cat((excelDataTensor[i*int(np.floor(len(excelDataSpectra)/100))],torch.rand(latent)))
    fixed_noise2 = fixed_noise1.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    fixed_noise = fixed_noise2.permute(1,0,2,3)
    testTensor = torch.cat((testTensor,fixed_noise),0)
testTensor = testTensor.to(device)

#Establish convention for real and fake labels during training
real_label = random.uniform(0.9,1.0)
fake_label = 0

#Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

##Training Loop
#Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0
noise = torch.Tensor()
noise2 = torch.Tensor()
print("Starting Training Loop...")
#For each epoch
x=0
for epoch in range(num_epochs):
    x=0
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, device=device)

        # Generate batch of Spectra,  latent vectors, and Properties     
        for j in range(batch_size):
            excelIndex = x*batch_size+j
            try:
                gotdata = excelDataTensor[excelIndex]
            except IndexError:
                break
            tensorA = excelDataTensor[excelIndex].view(1,800)
            noise2 = torch.cat((noise2,tensorA),0)      
            
            tensor1 = torch.cat((excelDataTensor[excelIndex],torch.rand(latent)))
            tensor2 = tensor1.unsqueeze(1).unsqueeze(1).unsqueeze(1)         
            tensor3 = tensor2.permute(1,0,2,3)
            noise = torch.cat((noise,tensor3),0)         
                              
        noise = noise.to(device)            
        noise2 = noise2.to(device)                
        
         # Forward pass real batch through D
        output = netD.forward(real_cpu,noise2).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()
              
        ## Train with all-fake batch                
        # Generate fake image batch with G
        fake = netG.forward(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD.forward(fake.detach(),noise2).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD.forward(fake,noise2).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2), file=f)

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

       #  Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(testTensor).detach().cpu()
            img_list.append(vutils.make_grid(fake, nrow=10, padding=2, normalize=True))

        iters += 1
        noise = torch.Tensor()
        noise2 = torch.Tensor()     
        x += 1
    if epoch % 50 == 0:
        ##Update folder location
        torch.save(netG, save_dir + 'netG' + str(epoch) + '.pt')
        torch.save(netD, save_dir + 'netD' + str(epoch) + '.pt')

local_time = time.ctime(time.time())
print('End Time = %s' % local_time)
print('End Time = %s' % local_time, file=f)
run_time = (time.time()-start_time)/3600
print('Total Time Lapsed = %s Hours' % run_time)
print('Total Time Lapsed = %s Hours' % run_time, file=f)
f.close()


#Save training progress video
ims, ani = Video.save_video(save_dir, img_list, G_losses, D_losses)


#Plot and save G and D Training Losses
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="Generator Loss")
plt.plot(D_losses,label="Discriminator Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig('losses.png')
plt.show()

