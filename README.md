# Multiclass_Metasurface_InverseDesign


## Introduction
Welcome to the Raman Lab GitHub! This repo will walk you through the code used in the following publication: https://onlinelibrary.wiley.com/doi/10.1002/adom.202100548

Here, we use a conditional deep convolutional generative adversarial network (cDCGAN) to inverse design across multiple classes of metasurfaces.

## Requirements
The following software is required to run the provided scripts. As of this writing, the versions below have been tested and verified. Training on GPU is recommended due to lengthy training times with GANs. 

-Python 3.7

-Pytorch 1.9.0

-CUDA 10.2 (Recommended for training on GPU)

-OpenCV 3.4.2 (Depends on Python 3.7, Python 3.8 is not supported as of this writing)
-Scipy 1.6.2

-Matplotlib

-Pandas

-Spyder 

Installation instructions for Pytorch (with CUDA) are at: https://pytorch.org/. For convenience, here are installation commands for the Conda distribution (after installing Anaconda: https://www.anaconda.com/products/individual).

```
conda create -n myenv python=3.7
conda activate myenv
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install -c anaconda opencv
conda install -c anaconda scipy
conda install matplotlib
conda install pandas
conda install spyder
```

## Steps
### 0) Setup ffmpeg:
Go to the 'Utilities/SaveAnimation.py' file and update the following line to setup 'ffmpeg':
```python
plt.rcParams['animation.ffmpeg_path'] = '/home/ramanlab/anaconda3/pkgs/ffmpeg-3.1.3-0/bin/ffmpeg'
```
Refer to here for more information: https://stackoverflow.com/questions/23856990/cant-save-matplotlib-animation
Optionally, comment out the save_video line in 'DCGAN_Train.py'.

### 1) Train the cDCGAN (DCGAN_Train.py)
Download the files in the 'Training Data' folder and update the following lines in the 'DCGAN_Train.py' file:
```python
#Location of Training Data
spectra_path = 'C:/.../absorptionData_HybridGAN.csv'

#Location to Save Models (Generators and Discriminators)
save_dir = 'C:/.../'

#Root directory for dataset (images must be in a subdirectory within this folder)
img_path = 'C:/.../Images'
```
Running this file will train the cDCGAN and save the models in the specified location (every 50 epochs). Depending on the available hardware, the training process can take up to a few hours. After training, the following will also be produced:

## 1.1) Log file showing losses and total training time:

## 1.2) Video showing generator outputs per epoch:

## 1.3) Plots of Generator and Discriminator losses:

### 2) Load cDCGAN & Predict by Inputting Target Spectrum (DCGAN_Predict.py)

### 3) Decode Image & Convert to Binary (DCGAN_Decode.py)

### 4) Generate Simulation Model - Lumerical (DCGAN_FDTD.py)
