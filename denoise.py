from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from torchmetrics import StructuralSimilarityIndexMeasure
from statistics import median, mean
from matplotlib import pyplot as plt
import numpy as np
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from torchsummary import summary
import json
from tqdm import tqdm
from glob import glob
import sys
path = "/study3/mrphys/skunkworks/kk/mriUnet"
sys.path.insert(0,path)
import unet
from torchvision import transforms
from torch.utils.data import Dataset
import h5py
from sklearn.model_selection import KFold as kf

allImages = sorted(glob("/study/mrphys/skunkworks/training_data//mover01/*/", recursive=True))

def slice2d(array, discardZero=False):
    '''
    slice a 4d array of shape (c-channel, n, n, n) where n in the cube length
    into 3d arrays slices of shape (c, n, n) per each 2d plane
    '''
    result = []
    c, w, h, d = array.shape
    assert (w==h)and(h==d)and(d==w), f"Array must be cubic, got: {w}x{h}x{d}"
    for i in range(w):
        result.append(array[:,i,:,:])
        result.append(array[:,:,i,:])
        result.append(array[:,:,:,i])
    return np.array(result)

def getComplexSlices(path, return_scale=False):

    with h5py.File(path,'r') as hf:
        prefix = 'C_000_0'
        imagestackReal = []
        imagestackImag = []
        for i in range(6):
            n = prefix + str(i).zfill(2)
            image = hf['Images'][n]
            imagestackReal.append(np.array(image['real']))
            imagestackImag.append(np.array(image['imag']))
            if i==0:
                normScale = np.max([np.abs(np.array(image['real'])).max(), np.abs(np.array(image['imag'])).max()])
        imagestackReal = np.array(imagestackReal)/normScale
        imagestackImag = np.array(imagestackImag)/normScale
        imagesliceReal = slice2d(imagestackReal)
        imagesliceImag = slice2d(imagestackImag)
        
    if return_scale:
        return imagesliceReal+imagesliceImag*1j, normScale
    else:
        return imagesliceReal+imagesliceImag*1j

class noisyDataset(Dataset):
    def __init__(self, sample):
        self.accelPathList = []
        self.accelFileList = []

        allImages = sorted(glob("/study/mrphys/skunkworks/training_data//mover01/*/", recursive=True))
        folderName  = allImages[sample]
        self.accelPathList.append(folderName +'processed_data/acc_2min/C.h5')
        
        for accelPath in self.accelPathList:
            slices, scale = getComplexSlices(accelPath, return_scale=True)
            self.accelFileList+= list(slices)
            self.scale = scale
            print('Image ' + accelPath + ' loaded')

    def __getitem__(self, index):
        return self.accelFileList[index]

    def __len__(self):
        return len(self.accelFileList)
    
def predict(model, dataset, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    model.eval()
    model.to(device)
    X = []
    Y = []
    Z = []
    for i, noisy in tqdm(enumerate(dataset)):
        noisy = torch.tensor(noisy).to(device).unsqueeze(0)
        with torch.no_grad():
            p = model(noisy).cpu().numpy() * dataset.scale
            if i%3==0:
                X.append(p)
            elif i%3==1:
                Y.append(p)
            else:
                Z.append(p)
                
    return np.vstack(X).transpose(1,0,2,3), np.vstack(Y).transpose(1,2,0,3), np.vstack(Z).transpose(1,2,3,0)

def get_prediction(idx, fold):
    
    model = unet.UNet(6,
    6,
    f_maps=32,
    layer_order=['separable convolution', 'relu'],
    depth=3,
    layer_growth=2.0,
    residual=True,
    complex_input=True,
    complex_kernel=True,
    ndims=2,
    padding=1)

    name = f'slice_kfold_{fold}'

    model.load_state_dict(torch.load(f'/study/mrphys/skunkworks/kk/outputs/{name}/weights/{name}_LATEST.pth'))

    dataset = noisyDataset(idx)
    X, Y, Z = predict(model, dataset)
    
    return (X+Y+Z)/3

kfsplitter = kf(n_splits=5, shuffle=True, random_state=69420)
for i, (train_index, test_index) in enumerate(kfsplitter.split(np.arange(65))):
    for idx in test_index:
        print(f'Fold = {i+1}')
        pred = get_prediction(idx, i+1)
        np.save(f'pred/denoised_{idx}.npy', pred)