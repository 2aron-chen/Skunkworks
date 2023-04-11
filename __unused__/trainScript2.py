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

allImages = sorted(glob("/study/mrphys/skunkworks/training_data//mover01/*/", recursive=True))

class h5DatasetIndividual(Dataset):
    def __init__(self, sample):
        self.orginalPathList = []
        self.accelPathList = []
        self.orginalFileList = []
        self.accelFileList = []
        # self.mid = int(256/2) - 3  ## minus three because we are taking the middle 8 slices

        folderName = allImages[sample]
        self.orginalPathList.append(folderName + 'processed_data/C.h5')
        self.accelPathList.append(folderName +'processed_data/acc_2min/C.h5')
        
        for orginalPath, accelPath in zip(self.orginalPathList, self.accelPathList):
            prefix = 'C_000_0'
            orginalImageNumpy_Stack = None
            accelImageNumpy_Stack = None
            with h5py.File(orginalPath,'r') as hf:
                for i in range(16):
                    n = prefix + str(i).zfill(2)
                    image = hf['Images'][n]
                
                    imageNumpy = image['real']
                    imageNumpy = imageNumpy-imageNumpy.min()
                    imageNumpy = imageNumpy * (1/(imageNumpy.max()))
                    orginalImageNumpy = np.array(imageNumpy + 0j*image['imag'])
                    if i == 0:
                        orginalImageNumpy_Stack = np.expand_dims(np.copy(orginalImageNumpy), axis=0)
                    else:
                        orginalImageNumpy_Stack = np.concatenate((orginalImageNumpy_Stack, np.expand_dims(orginalImageNumpy, axis=0)), axis=0)

            
            with h5py.File(accelPath,'r') as hf:
                for i in range(16):
                    n = prefix + str(i).zfill(2)
                    image = hf['Images'][n]
                
                    imageNumpy = image['real']
                    imageNumpy = imageNumpy-imageNumpy.min()
                    imageNumpy = imageNumpy * (1/(imageNumpy.max()))
                    accelImageNumpy = np.array(imageNumpy + 0j*image['imag'])
                    if i == 0:
                        accelImageNumpy_Stack = np.expand_dims(np.copy(accelImageNumpy), axis=0)
                    else:
                        accelImageNumpy_Stack = np.concatenate((accelImageNumpy_Stack, np.expand_dims(accelImageNumpy, axis=0)), axis=0)

            for i in range(256): ## train each slice for each subject
                for j in range(16):
                    if j == 0:
                        orginalStack = np.expand_dims(np.copy(orginalImageNumpy_Stack[j][i][32:224]), axis=0)
                        accelStack = np.expand_dims(np.copy(accelImageNumpy_Stack[j][i][32:224]), axis=0)
                    else:
                        orginalStack = np.concatenate((orginalStack, np.expand_dims(orginalImageNumpy_Stack[j][i][32:224], axis=0)), axis=0)
                        accelStack = np.concatenate((accelStack, np.expand_dims(accelImageNumpy_Stack[j][i][32:224], axis=0)), axis=0)
                self.orginalFileList.append(orginalStack)
                self.accelFileList.append(accelStack)
            
            print('Image ' + orginalPath + ' loaded')

    def __getitem__(self, index):
        return self.accelFileList[index], self.orginalFileList[index]

    def __len__(self):
        return len(self.accelFileList)

trainData  = []
testData = []

pbar = tqdm(range(len(allImages)), desc="loading datasets")

for i in pbar:
    with open(f'/scratch/mrphys/pickled/dataset_{i}.pickle', 'rb') as f:
        data = pickle.load(f)
        if i>= 55: # test
            testData.append(data)
        else: # train
            trainData.append(data)
        del data

transformIdentity = lambda x : x

class Trainer:
    
    def __init__(self, 
                 model, 
                 learningRate,
                 train_data, 
                 test_data,
                 norm_scale = 1,
                 model_name = 'mriUnet_features',
                 device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 transforms = transformIdentity
                ):
        
        self.lossCounter = {
            'train':[],
            'test':[],
        } #can unpack into pandas dataFrame later
        
        self.model = model.to(device)
        self.name = model_name
        self.trainLoader = train_data
        self.testLoader = test_data
        self.norm_scale = norm_scale
        self.device = device
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learningRate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.5)
        
        self.transforms = transforms
        self.device = device

    def trainOneEpoch(self, curr_ep):
        
        self.model.to(self.device)
        
        self.model.train()
        
        pbar = tqdm(enumerate(self.trainLoader))
        total_batch = len(self.trainLoader)
        pbar.set_description(f"Training Epoch : {curr_ep}")
        
        meanLoss = 0
        counter = 0
        
        for batch, (X, y) in pbar:
            
            batch_size = X.size()[0]
            Batch = torch.cat([X, y], axis=0)
            Batch = self.transforms(Batch)
            X = Batch[:batch_size]
            y = Batch[batch_size:]
            
            X, y = X*self.norm_scale, y*self.norm_scale
            X, y = X.to(self.device), y.to(self.device)
            pred = torch.sigmoid(self.model(X))
            ssim_loss = (1-ms_ssim(pred.real, y.real, data_range=self.norm_scale, size_average=False)).mean()
            
            ## update loss counter
            counter += X.shape[0]
            meanLoss += ssim_loss.item()*X.shape[0]
            
            ## Backpropagation
            self.optimizer.zero_grad()
            ssim_loss.backward() 
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
        
        for batch, (X, y) in pbar:
            
            with torch.no_grad():
                
                batch_size = X.size()[0]
                Batch = torch.cat([X, y], axis=0)
                Batch = self.transforms(Batch)
                X = Batch[:batch_size]
                y = Batch[batch_size:]
            
                X, y = X*self.norm_scale, y*self.norm_scale
                X, y = X.to(self.device), y.to(self.device)
                pred = torch.sigmoid(self.model(X))
                ssim_loss = (1-ms_ssim(pred.real, y.real, data_range=self.norm_scale, size_average=False)).mean()
            
                #update loss counter
                counter += X.shape[0]
                meanLoss += ssim_loss.item()*X.shape[0]

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
        plt.savefig(f'/study/mrphys/skunkworks/kk/lossPlot/{self.name}_LossPlot.png')
        plt.close()
    
    def trainLoop(self, 
                  epochs,
                  es_patience = 20,
                  lr_patience = 10,
                  fromCheckpoint = False,
                 ):
        patienceCounter = 0,
        bestLoss = 1e9 #REALLY LARGE

        if fromCheckpoint:

            # load model's weight
            self.model.load_state_dict(torch.load(f'/study/mrphys/skunkworks/kk/weights/{self.name}_BEST.pth', map_location=self.device))

            # load training log
            with open(f'/study/mrphys/skunkworks/kk/logs/{self.name}_logs.json', 'r') as f:
                self.lossCounter = json.load(f)
                bestLoss = np.min(self.lossCounter['test'])
                patienceCounter = len(self.lossCounter['test'])-1-np.argmin(self.lossCounter['test'])

        
        for curr_ep in range(epochs):
            meanTrainLoss = self.trainOneEpoch(curr_ep)
            meanTestLoss = self.testOneEpoch(curr_ep)
            self.saveLossPLot()

            #EARLYSTOPPING
            if bestLoss > meanTestLoss:
                bestLoss = meanTestLoss
                patienceCounter = 0
                torch.save(self.model.state_dict(), f'/study/mrphys/skunkworks/kk/weights/{self.name}_BEST.pth')
                with open(f'/study/mrphys/skunkworks/kk/logs/{self.name}_logs.json', 'w') as f:
                    json.dump(self.lossCounter, f)
            else:
                patienceCounter += 1
                
            print(f'Early Stopping Counter = {patienceCounter}/20')

            if patienceCounter>=lr_patience:
                print('Loss stops improving for 10 epochs -> LR step by 0.5')
                self.scheduler.step()
                
            if patienceCounter>=es_patience:
                print('Loss stops improving for 20 epochs -> EARLY STOPPING')
                break

transformSet = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ])

BATCHSIZE = 16

trainDataset = torch.utils.data.ConcatDataset(trainData)
testDataset = torch.utils.data.ConcatDataset(testData)

trainDataloader = DataLoader(dataset=trainDataset, batch_size=BATCHSIZE, shuffle=True)
testDataloader = DataLoader(dataset=testDataset, batch_size=BATCHSIZE, shuffle=False)

model = unet.UNet(16,
        16,
        f_maps=32,
        layer_order=['separable convolution', 'relu'],
        depth=3,
        layer_growth=2.0,
        residual=True,
        complex_input=True,
        complex_kernel=True,
        ndims=2,
        padding=1)

if __name__=="__main__":
    name = 'mriUnet_features_desp_16'
    print(name)
    trainer = Trainer(model, 
        1e-3,
        trainDataloader, 
        testDataloader,
        norm_scale = 1,
        transforms=transformIdentity,
        model_name = name
    )
    trainer.trainLoop(1000, fromCheckpoint = False)