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
        for i in range(16):
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

class mriSliceDataset(Dataset):
    def __init__(self, sample):
        self.originalPathList = []
        self.accelPathList = []
        self.originalFileList = []
        self.accelFileList = []

        allImages = sorted(glob("/study/mrphys/skunkworks/training_data//mover01/*/", recursive=True))
        folderName  = allImages[sample]
        self.originalPathList.append(folderName + 'processed_data/C.h5')
        self.accelPathList.append(folderName +'processed_data/acc_2min/C.h5')
        
        for originalPath, accelPath in zip(self.originalPathList, self.accelPathList):
            self.originalFileList+= list(getComplexSlices(originalPath))
            self.accelFileList+= list(getComplexSlices(accelPath))
            print('Image ' + originalPath + ' loaded')

    def __getitem__(self, index):
        return self.accelFileList[index], self.originalFileList[index]

    def __len__(self):
        return len(self.accelFileList)

traintestData  = []

pbar = tqdm(range(len(allImages[:10])), desc="loading datasets")

for i in pbar:
    with open(f'/scratch/mrphys/pickled/dataset_{i}.pickle', 'rb') as f:
        data = pickle.load(f)
        traintestData.append(data)
        del data

class Trainer:
    
    def __init__(self, 
                 model, 
                 learningRate,
                 train_data, 
                 test_data,
                 norm_scale = 1,
                 model_name = 'mriUnet',
                 gpu_id = 0,
                 parent_dir = '/study/mrphys/skunkworks/kk',
                ):
        
        self.lossCounter = {
            'train':[],
            'test':[],
        } #can unpack into pandas dataFrame later'

        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.model.to(self.device)
        self.name = model_name
        self.trainLoader = train_data
        self.testLoader = test_data
        self.norm_scale = norm_scale
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learningRate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.5)
        self.min_lr = learningRate/1000
        
        self.parent_dir = parent_dir
        
        #make directories for checkpoint
        os.makedirs(f'{self.parent_dir}/outputs/{self.name}', exist_ok=True)
        os.makedirs(f'{self.parent_dir}/outputs/{self.name}/logs', exist_ok=True)
        os.makedirs(f'{self.parent_dir}/outputs/{self.name}/weights', exist_ok=True)
        os.makedirs(f'{self.parent_dir}/outputs/{self.name}/lossPlot', exist_ok=True)
        os.makedirs(f'{self.parent_dir}/outputs/{self.name}/preds', exist_ok=True)

    def trainOneEpoch(self, curr_ep):
        
        self.model.to(self.device)
        
        self.model.train()
        
        pbar = tqdm(enumerate(self.trainLoader))
        total_batch = len(self.trainLoader)
        pbar.set_description(f"Training Epoch : {curr_ep}")
        
        meanLoss = 0
        counter = 0
        mse = nn.MSELoss()
        
        for batch, (X, y) in pbar:
            
            batch_size = X.size()[0]
            
            X, y = X*self.norm_scale, y*self.norm_scale
            X, y = X.to(self.device), y.to(self.device)
            pred = self.model(X)
            l2_loss = mse(pred.real, y.real) + mse(pred.imag, y.imag)
            pred = torch.sigmoid(pred)
            y = torch.sigmoid(y)
            ssim_loss = (1-ms_ssim(pred.real, y.real, data_range=self.norm_scale, size_average=False)).mean() + (1-ms_ssim(pred.imag, y.imag, data_range=self.norm_scale, size_average=False)).mean()
            
            ## update loss counter
            loss = ssim_loss #+l2_loss
            counter += X.shape[0]
            meanLoss += loss.item()*X.shape[0]
            
            ## Backpropagation
            self.optimizer.zero_grad()
            loss.backward() 
            self.optimizer.step()
            
            ## Memory clear
            del X, y, pred
            
            pbar.set_description(f"Training Epoch : {curr_ep} [batch {batch+1}/{total_batch}] - loss = {round(meanLoss/counter,6)}")
            
        self.lossCounter['train'].append(meanLoss/counter)
        
        return meanLoss/counter  

    def testOneEpoch(self, curr_ep):
        
        self.model.to(self.device)
        
        self.model.eval()
        
        pbar = tqdm(enumerate(self.testLoader))
        total_batch = len(self.testLoader)
        pbar.set_description(f"Testing Epoch : {curr_ep}")
        
        meanLoss = 0
        counter = 0
        mse = nn.MSELoss()
        
        for batch, (X, y) in pbar:
            
            with torch.no_grad():
                
                batch_size = X.size()[0]
            
                X, y = X*self.norm_scale, y*self.norm_scale
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
  
                l2_loss = mse(pred.real, y.real) + mse(pred.imag, y.imag)
                pred = torch.sigmoid(pred)
                y = torch.sigmoid(y)
                ssim_loss = (1-ms_ssim(pred.real, y.real, data_range=self.norm_scale, size_average=False)).mean() + (1-ms_ssim(pred.imag, y.imag, data_range=self.norm_scale, size_average=False)).mean()
            
                #update loss counter
                loss = ssim_loss+l2_loss
                counter += X.shape[0]
                meanLoss += loss.item()*X.shape[0]

                pbar.set_description(f"Testing Epoch : {curr_ep} [batch {batch+1}/{total_batch}] - loss = {round(meanLoss/counter,6)}")
                
                ## Memory clear
                del X, y, pred
            
        self.lossCounter['test'].append(meanLoss/counter)
        
        return meanLoss/counter
    
    def saveLossPLot(self):
        tr_loss = self.lossCounter['train']
        te_loss = self.lossCounter['test']
        plt.plot(tr_loss, label='train loss')
        plt.plot(te_loss, label='test loss')
        plt.legend()
        plt.savefig(f'{self.parent_dir}/outputs/{self.name}/lossPlot/{self.name}_LossPlot.png')
        plt.close()
        
    def savepred(self, curr_ep):
        
        fixedX, fixedY = testData[5][200]
        fixedX = torch.unsqueeze(torch.tensor(fixedX),0)
        fixedY = torch.unsqueeze(torch.tensor(fixedY),0)

        model.eval()
        fixedX = fixedX.to(self.device)
        with torch.no_grad():
            pred = model(fixedX)
        fixedX = fixedX.cpu()
        pred = pred.cpu()
            
        plt.gray()
        fig, ax = plt.subplots(3, 6, figsize=(20,6)) 
        for i in range(6):
            ax[0,i].imshow(torch.abs(fixedX[0,i]))
            ax[0,i].axis('off')
        for i in range(6):
            ax[1,i].imshow(torch.abs(pred[0,i]))
            ax[1,i].axis('off')
        for i in range(6):
            ax[2,i].imshow(torch.abs(fixedY[0,i]))
            ax[2,i].axis('off')
        plt.savefig(f'{self.parent_dir}/outputs/{self.name}/preds/{self.name}_pred_{curr_ep}.png')
        plt.close()

    def trainLoop(self, 
                  epochs,
                  es_patience = 1e9, #set to absurdly high meaning we don't care about es
                  lr_patience = 10,
                  fromCheckpoint = False,
                 ):
        patienceCounter = 0,
        bestLoss = 1e9 #REALLY LARGE
        start_ep = 0

        if fromCheckpoint:

            try:

                # load model's weight (Latest)
                self.model.load_state_dict(torch.load(f'{self.parent_dir}/outputs/{self.name}/weights/{self.name}_LATEST.pth', map_location=self.device))

                # load training log
                with open(f'{self.parent_dir}/outputs/{self.name}/logs/{self.name}_logs_LATEST.json', 'r') as f:
                    self.lossCounter = json.load(f)
                    bestLoss = np.min(self.lossCounter['test'])
                    patienceCounter = len(self.lossCounter['test'])-1-np.argmin(self.lossCounter['test'])

                start_ep = len(self.lossCounter['test'])

            except:

                # load model's weight (Best)
                self.model.load_state_dict(torch.load(f'{self.parent_dir}/outputs/{self.name}/weights/{self.name}_BEST.pth', map_location=self.device))

                # load training log
                with open(f'{self.parent_dir}/outputs/{self.name}/logs/{self.name}_logs.json', 'r') as f:
                    self.lossCounter = json.load(f)
                    bestLoss = np.min(self.lossCounter['test'])
                    patienceCounter = len(self.lossCounter['test'])-1-np.argmin(self.lossCounter['test'])

                start_ep = len(self.lossCounter['test'])

        
        for curr_ep in range(epochs):
            curr_ep = curr_ep + start_ep
            meanTrainLoss = self.trainOneEpoch(curr_ep)
            meanTestLoss = self.testOneEpoch(curr_ep)
            self.saveLossPLot()

            #EARLYSTOPPING
            if bestLoss > meanTestLoss:
                bestLoss = meanTestLoss
                patienceCounter = 0
                torch.save(self.model.state_dict(), f'{self.parent_dir}/outputs/{self.name}/weights/{self.name}_BEST.pth')
                with open(f'{self.parent_dir}/outputs/{self.name}/logs/{self.name}_logs.json', 'w') as f:
                    json.dump(self.lossCounter, f)
            else:
                patienceCounter += 1
                
            #scheduled save
            if curr_ep%10==0:
                torch.save(self.model.state_dict(), f'{self.parent_dir}/outputs/{self.name}/weights/{self.name}_ep{curr_ep}.pth')


            torch.save(self.model.state_dict(), f'{self.parent_dir}/outputs/{self.name}/weights/{self.name}_LATEST.pth')
            with open(f'{self.parent_dir}/outputs/{self.name}/logs/{self.name}_logs_LATEST.json', 'w') as f:
                json.dump(self.lossCounter, f)
            
            #savefig every epoch
            self.savepred(curr_ep)
                
            print(f'Early Stopping Counter = {patienceCounter}/20')

            if patienceCounter>=lr_patience and self.scheduler.get_last_lr()[-1]>=self.min_lr:
                print('Loss stops improving for 10 epochs -> LR step')
                self.scheduler.step()
                
            if patienceCounter>=es_patience:
                print('Loss stops improving -> EARLY STOPPING')
                break

if __name__=="__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description='train script')
    parser.add_argument('fold', type=int, help='which fold of 1-5')
    # args = parser.parse_args()
    
    kfsplitter = kf(n_splits=1, shuffle=True, random_state=69420)

    for i, (train_index, test_index) in enumerate(kfsplitter.split(traintestData)):
        
        fold = i+1
        if fold!=1:
        # if fold!=args.fold:
            continue
        
        trainData = [traintestData[i] for i in train_index]
        testData = [traintestData[i] for i in test_index]
        BATCHSIZE = 32
        trainDataset = torch.utils.data.ConcatDataset(trainData)
        testDataset = torch.utils.data.ConcatDataset(testData)
        print(len(trainDataset), len(testDataset))

        trainDataloader = DataLoader(dataset=trainDataset, batch_size=BATCHSIZE, shuffle=False)
        testDataloader = DataLoader(dataset=testDataset, batch_size=BATCHSIZE, shuffle=False)
        
        model = unet.UNet(
            10,
            10,
            f_maps=32,
            layer_order=['separable convolution', 'relu'],
            depth=4,
            layer_growth=2.0,
            residual=True,
            complex_input=True,
            complex_kernel=True,
            ndims=2,
            padding=1
        )
    
        name = f'fullDenoiser_{fold}'
        print(name)
        trainer = Trainer(
            model, 
            1e-3,
            trainDataloader, 
            testDataloader,
            norm_scale = 1,
            model_name = name,
            gpu_id = i
        )
        trainer.trainLoop(100, fromCheckpoint = False)
        
        fold += 1