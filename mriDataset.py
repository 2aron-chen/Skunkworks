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
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

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
        
    return imagestackReal+imagestackImag*1j

class mriSliceDataset(Dataset):
    def __init__(self, sample):
        self.originalPath = []
        self.accelPath = [] 

        allImages = sorted(glob("/study/mrphys/skunkworks/training_data//mover01/*/", recursive=True))
        folderName  = allImages[sample]
        self.originalPath = folderName + 'processed_data/C.h5'
        self.accelPath = folderName +'processed_data/acc_2min/C.h5'
        
        self.originalFile = getComplexSlices(self.originalPath)
        self.accelFile = getComplexSlices(self.accelPath)

    def __getitem__(self, index):
        if index<256:
            return self.accelFile[:,index,:,:], self.originalFile[:,index,:,:]
        elif index<512:
            index = index-256
            return self.accelFile[:,:,index,:], self.originalFile[:,:,index,:]
        else:
            index = index-512
            return self.accelFile[:,:,:,index], self.originalFile[:,:,:,index]
        
    def __len__(self):
        return 768