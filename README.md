# Multiclass_Metasurface_InverseDesign


## Introduction
Welcome to the Raman Lab GitHub! This repo will walk you through the code used in the following publication: ___

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
### 1) Train the cDCGAN (DCGAN_Train.py)
Download the files in the 'Training Data' folder and update lines 38, 41, and 44 in the 'DCGAN_Train.py' file:
```python
## Define File Locations (Images, Spectra, and CNN Model Save)
spectra_path = 'C:/.../Spectra.csv'
save_dir = 'C:/.../model.h5'
img_path = 'C:/.../*.png'
```
Running this file will train the cDCGAN and save the model in the specified location. Depending on the available hardware, the training process can take up to a few hours.

### 2) Load cDCGAN & Predict by Inputting Target Spectrum (DCGAN_Predict.py)

### 3) Decode Image & Convert to Binary (DCGAN_Decode.py)

### 4) Generate Simulation Model - Lumerical (DCGAN_FDTD.py)
