from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from statistics import median, mean
from matplotlib import pyplot as plt
import numpy as np
import json
from tqdm import tqdm
from glob import glob
import os
import sys
sys.path.insert(0,"/study/mrphys/skunkworks/kk/mriUnet")
import unet
from torchvision import transforms
from torch.utils.data import Dataset
from sklearn.model_selection import KFold as kf
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import h5py
import shutil

allImages = sorted(glob('/study/mrphys/skunkworks/kk/pred/denoised_*.npy', recursive=True))
T1path = sorted(glob('/study/mrphys/skunkworks/training_data/mover01/*/processed_data/T1_3_tv.nii'))
samples = [p.split('/')[6] for p in T1path]
noisyPath = [f'/study/mrphys/skunkworks/training_data/mover01/{p}/processed_data/acc_2min/C.h5' for p in samples]

def getComplexSlices(path):

    with h5py.File(path,'r') as hf:
        prefix = 'C_000_0'
        imagestackReal = []
        imagestackImag = []
        for i in range(10):
            n = prefix + str(i).zfill(2)
            image = hf['Images'][n]
            imagestackReal.append(np.array(image['real']))
            imagestackImag.append(np.array(image['imag']))
            if i==0:
                normScale = np.abs(np.array(image['real']+image['real']*1j)).max()
        imagestackReal = np.array(imagestackReal)/normScale
        imagestackImag = np.array(imagestackImag)/normScale
        
    return imagestackReal+imagestackImag*1j, normScale

try:
    os.mkdir('/scratch/mrphys/denoised')
except:
    pass

for i, imgIndex in tqdm(enumerate(range(len(allImages)))):
    
    name = sorted(glob("/study/mrphys/skunkworks/training_data//mover01/*/", recursive=True))[imgIndex].split('/')[-2]
    noisypath = noisyPath[i]
    
    if os.path.exists(f'/scratch/mrphys/denoised/{name}/C.h5'):
        file_stats = os.stat(f'/scratch/mrphys/denoised/{name}/C.h5')
        file_size_gb = file_stats.st_size / (1024 * 1024 * 1024)
        if file_size_gb>=2: #finished file
            continue
    
    try:
        os.mkdir(f'/scratch/mrphys/denoised/{name}')
    except:
        pass
    
    for file in ['basis.arma','gmpnrage_parameters.txt','mpnrage_t1fitting_parameters.txt']:
                
        shutil.copy(f'/study/mrphys/skunkworks/kk/modelfitting/{file}', f'/scratch/mrphys/denoised/{name}/{file}')
        
    noisy, normscale = getComplexSlices(noisypath)

    with h5py.File(f'/scratch/mrphys/denoised/{name}/C.h5','w') as f:
        pred = np.load(f'pred/denoised_{i}.npy')
        
        #norming
        pred[noisy==0]*=0
        pred /= np.max(np.abs(pred[0]))
        pred *= normscale
        
        #pad the remaining pca
        zero = np.zeros([6,256,256,256]).astype(pred.dtype)
        pred = np.concatenate([pred, zero])
        
        temp = pred.astype(np.dtype([('real','f'),('imag','f')]))
        temp['imag'] = pred.imag
        pred = temp
        grp = f.create_group('Images')
        for n in range(16):     
            grp.create_dataset('C_000_0'+ str(n).zfill(2), data=pred[n])
            
    del noisy, pred