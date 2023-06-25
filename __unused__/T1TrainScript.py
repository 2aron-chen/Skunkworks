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
import os
import sys
path = "/study3/mrphys/skunkworks/kk/mriUnet"
sys.path.insert(0,path)
import unet
from torchvision import transforms
from torch.utils.data import Dataset
from sklearn.model_selection import KFold as kf
import nibabel as nib

T1path = sorted(glob('/study3/mrphys/skunkworks/training_data/mover01/*/processed_data/T1_3_tv.nii'))
xPath = sorted(glob('/scratch/mrphys/skunkworks/denoised/denoised_*.h5'))
gtPath = sorted(glob('/study3/mrphys/skunkworks/training_data/mover01/*/processed_data/C.h5'))

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

class T1Dataset(Dataset):
    
    def __init__(self, index, gt=False):
    
        if gt:
            self.x_path = gtPath[index]
        else:
            self.x_path = xPath[index]  
        self.y_path = T1path[index]
        
        self.x = list(getComplexSlices(self.x_path))
        self.y = list(slice2d(np.transpose(nib.load(self.y_path).get_fdata()).reshape(1,256,256,256)))

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

traintestData  = []

pbar = tqdm(range(len(T1path)), desc="loading datasets")

for i in pbar:
    with open(f'/scratch/mrphys/pickled/T1dataset2_{i}.pickle', 'rb') as f:
        data = pickle.load(f)
        traintestData.append(data)
        del data

transformIdentity = lambda x : x

class Trainer:
    
    def __init__(self, 
                 model, 
                 learningRate,
                 train_data, 
                 test_data,
                 model_name = 'PCAtoT1',
                 discriminator = None,
                 device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu"),
                 unsupervised = False,
                ):
        
        self.lossCounter = {
            'train':[],
            'test':[],
        } #can unpack into pandas dataFrame later
        
        self.unsupervised = unsupervised
        
        self.model = model.to(device)
        if discriminator is not None:
            self.gan = True
            self.dis = discriminator.to(device)
            self.optim_d = torch.optim.Adam(self.dis.parameters(), lr=learningRate)
            self.lossCounter['dis_loss'] = []
            self.bce = nn.BCELoss()
        else:
            self.gan = False
        self.name = model_name
        self.trainLoader = train_data
        self.testLoader = test_data
        self.device = device
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learningRate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.5)
        self.min_lr = learningRate/1000
        
        self.device = device
        self.loss = torch.nn.MSELoss()
        
        #make directories for saving
        os.makedirs(f'/study/mrphys/skunkworks/kk/outputs/{self.name}', exist_ok=True)
        os.makedirs(f'/study/mrphys/skunkworks/kk/outputs/{self.name}/logs', exist_ok=True)
        os.makedirs(f'/study/mrphys/skunkworks/kk/outputs/{self.name}/weights', exist_ok=True)
        os.makedirs(f'/study/mrphys/skunkworks/kk/outputs/{self.name}/lossPlot', exist_ok=True)
        os.makedirs(f'/study/mrphys/skunkworks/kk/outputs/{self.name}/preds', exist_ok=True)

    def trainOneEpoch(self, curr_ep):
        
        self.model.to(self.device)
        
        self.model.train()
        
        pbar = tqdm(enumerate(self.trainLoader))
        total_batch = len(self.trainLoader)
        pbar.set_description(f"Training Epoch : {curr_ep}")
        
        meanLoss = 0
        disLoss = 0
        counter = 0
        
        for batch, (X, y) in pbar:
            
            X, y = X.to(self.device), y.to(self.device).float()
            pred = torch.abs(self.model(X))
            
            if self.gan: # train discrim first
                self.optim_d.zero_grad()
                fake_pred = torch.sigmoid(self.dis(pred.detach(), torch.abs(X)))
                real_pred = torch.sigmoid(self.dis(y, torch.abs(X)))
                fake_lbl = torch.zeros_like(fake_pred)
                real_lbl = torch.ones_like(real_pred)
                fake_loss = self.bce(fake_pred, fake_lbl)
                real_loss = self.bce(real_pred, real_lbl)
                dis_loss = fake_loss + real_loss
                dis_loss.backward()
                self.optim_d.step()
                disLoss += dis_loss.item()*X.shape[0]
            
            pred = torch.abs(self.model(X))
            
            mseLoss = self.loss(pred, y)
            
            ## update loss counter
            counter += X.shape[0]
            meanLoss += mseLoss.item()*X.shape[0]
            
            if self.gan:
                fake_pred = torch.sigmoid(self.dis(pred, torch.abs(X)))
                fake_lbl = torch.ones_like(fake_pred)
                if not self.unsupervised:
                    loss = mseLoss + self.bce(fake_pred, fake_lbl)
                else:
                    loss = self.bce(fake_pred, fake_lbl)
                pbar.set_description(f"Training Epoch : {curr_ep} [batch {batch+1}/{total_batch}] - loss = {round(meanLoss/counter,6)} - discriminator loss = {round(disLoss/counter,6)}")
            else:
                loss = mseLoss
                pbar.set_description(f"Training Epoch : {curr_ep} [batch {batch+1}/{total_batch}] - loss = {round(meanLoss/counter,6)}")
            
            ## Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            ## Memory clear
            del X, y, pred
            
        self.lossCounter['train'].append(meanLoss/counter)
        if self.gan:
            self.lossCounter['dis_loss'].append(disLoss/counter)
        
        return meanLoss/counter  

    def testOneEpoch(self, curr_ep):
        
        self.model.to(self.device)
        
        self.model.eval()
        
        pbar = tqdm(enumerate(self.testLoader))
        total_batch = len(self.testLoader)
        pbar.set_description(f"Testing Epoch : {curr_ep}")
        
        meanLoss = 0
        counter = 0
        
        for batch, (X, y) in pbar:
            
            with torch.no_grad():
            
                X, y = X.to(self.device), y.to(self.device).float()
                pred = torch.abs(self.model(X))

                loss = self.loss(pred, y)
                # ssim_loss = (1-ms_ssim(torch.sigmoid(pred), torch.sigmoid(y), data_range=1.0, size_average=False)).mean()
                # loss = loss + ssim_loss
            
                #update loss counter
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
        plt.savefig(f'/study/mrphys/skunkworks/kk/outputs/{self.name}/lossPlot/{self.name}_LossPlot.png')
        plt.close()
        
        if self.gan:
            dis_loss = self.lossCounter['dis_loss']
            plt.plot(dis_loss, label='train loss')
            plt.legend()
            plt.savefig(f'/study/mrphys/skunkworks/kk/outputs/{self.name}/lossPlot/{self.name}_GANLossPlot.png')
            plt.close()
        
    def savepred(self, curr_ep):
        
        plt.gray()
        self.model.eval()
        fig, ax = plt.subplots(3, 10, figsize=(20,6)) 
        for i in range(10):
            fixedX, fixedY = testData[5][200+i]
            fixedX = torch.unsqueeze(torch.tensor(fixedX),0)
            fixedY = torch.unsqueeze(torch.tensor(fixedY),0)
            fixedX = fixedX.to(self.device)
            with torch.no_grad():
                pred = torch.abs(model(fixedX))
            fixedX = torch.abs(fixedX.cpu())
            pred = pred.cpu()
            ax[0,i].imshow(fixedX[0,0])
            ax[0,i].axis('off')
            ax[1,i].imshow(pred[0,0])
            ax[1,i].axis('off')
            ax[2,i].imshow(fixedY[0,0])
            ax[2,i].axis('off')
        plt.savefig(f'/study/mrphys/skunkworks/kk/outputs/{self.name}/preds/{self.name}_pred_{curr_ep}.png')
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
                self.model.load_state_dict(torch.load(f'/study/mrphys/skunkworks/kk/outputs/{self.name}/weights/{self.name}_LATEST.pth', map_location=self.device))
                if self.gan:
                    self.dis.load_state_dict(torch.load(f'/study/mrphys/skunkworks/kk/outputs/{self.name}/weights/{self.name}_dis_LATEST.pth'))

                # load training log
                with open(f'/study/mrphys/skunkworks/kk/outputs/{self.name}/logs/{self.name}_logs_LATEST.json', 'r') as f:
                    self.lossCounter = json.load(f)
                    bestLoss = np.min(self.lossCounter['test'])
                    patienceCounter = len(self.lossCounter['test'])-1-np.argmin(self.lossCounter['test'])

                start_ep = len(self.lossCounter['test'])

            except:

                # load model's weight (Best)
                self.model.load_state_dict(torch.load(f'/study/mrphys/skunkworks/kk/outputs/{self.name}/weights/{self.name}_BEST.pth', map_location=self.device))

                # load training log
                with open(f'/study/mrphys/skunkworks/kk/outputs/{self.name}/logs/{self.name}_logs.json', 'r') as f:
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
                torch.save(self.model.state_dict(), f'/study/mrphys/skunkworks/kk/outputs/{self.name}/weights/{self.name}_BEST.pth')
                if self.gan:
                    torch.save(self.dis.state_dict(), f'/study/mrphys/skunkworks/kk/outputs/{self.name}/weights/{self.name}_dis_BEST.pth')
                with open(f'/study/mrphys/skunkworks/kk/outputs/{self.name}/logs/{self.name}_logs.json', 'w') as f:
                    json.dump(self.lossCounter, f)
            else:
                patienceCounter += 1
                
            #scheduled save
            if curr_ep%10==0:
                torch.save(self.model.state_dict(), f'/study/mrphys/skunkworks/kk/outputs/{self.name}/weights/{self.name}_ep{curr_ep}.pth')
                if self.gan:
                    torch.save(self.dis.state_dict(), f'/study/mrphys/skunkworks/kk/outputs/{self.name}/weights/{self.name}_dis_ep{curr_ep}.pth')

            torch.save(self.model.state_dict(), f'/study/mrphys/skunkworks/kk/outputs/{self.name}/weights/{self.name}_LATEST.pth')
            if self.gan:
                torch.save(self.dis.state_dict(), f'/study/mrphys/skunkworks/kk/outputs/{self.name}/weights/{self.name}_dis_LATEST.pth')
            with open(f'/study/mrphys/skunkworks/kk/outputs/{self.name}/logs/{self.name}_logs_LATEST.json', 'w') as f:
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
    
    kfsplitter = kf(n_splits=5, shuffle=True, random_state=69420)
    
    fold = 1

    for train_index, test_index in kfsplitter.split(traintestData):
        trainData = [traintestData[i] for i in train_index]
        testData = [traintestData[i] for i in test_index]
        BATCHSIZE = 32
        trainDataset = torch.utils.data.ConcatDataset(trainData)
        testDataset = torch.utils.data.ConcatDataset(testData)
        print(len(trainDataset), len(testDataset))

        trainDataloader = DataLoader(dataset=trainDataset, batch_size=BATCHSIZE, shuffle=True)
        testDataloader = DataLoader(dataset=testDataset, batch_size=BATCHSIZE, shuffle=False)
        
        model = unet.UNet(6,
            1,
            f_maps=32,
            layer_order=['separable convolution', 'relu','batch norm'],
            depth=3,
            layer_growth=2.0,
            residual=True,
            complex_input=True,
            complex_kernel=True,
            ndims=2,
            padding=1)
        
        discriminator = unet.PatchGAN(
            7,
            f_maps=32,
            layer_order=['separable convolution', 'relu', 'batch norm'],
            depth=3,
            layer_growth=2.0,
            residual=True,
            complex_input=False,
            complex_kernel=False,
            ndims=2,
            padding=1
        )
    
        name = f'T1_denoised_GAN_unsupervised_{fold}'
        print(name)
        trainer = Trainer(
            model, 
            1e-3,
            trainDataloader, 
            testDataloader,
            model_name = name, 
            discriminator = discriminator,
            unsupervised = True,
        )
        trainer.trainLoop(100, fromCheckpoint = False)
        
        fold += 1
        break #train for 1 fold!