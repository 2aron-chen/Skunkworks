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

# dataset 
T1path = sorted(glob('/study/mrphys/skunkworks/training_data/mover01/*/processed_data/T1_3_tv.nii'))[0:40]
xPath = sorted(glob('/scratch/mrphys/denoised/denoised_*.h5'))[0:40]
gtPath = sorted(glob('/study/mrphys/skunkworks/training_data/mover01/*/processed_data/C.h5'))[0:40]
def getComplexSlices(path, return_scale=False):
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
                normScale = np.max(np.abs(np.array(image['real'])+np.array(image['imag'])*1J))
        imagestackReal = np.array(imagestackReal)/normScale
        imagestackImag = np.array(imagestackImag)/normScale
        
    if return_scale:
        return imagestackReal+imagestackImag*1j, normScale
    else:
        return imagestackReal+imagestackImag*1j    
class T1Dataset(Dataset):  
    def __init__(self, index, gt=False, norm_factor=1000):
        if gt:
            self.x_path = gtPath[index]
        else:
            self.x_path = xPath[index]  
        self.y_path = T1path[index]
        
        self.x = getComplexSlices(self.x_path)
        self.y = np.transpose(nib.load(self.y_path).get_fdata()).reshape(1,256,256,256)/norm_factor

    def __getitem__(self, index):
        if index<256:
            return self.x[:,index,:,:], self.y[:,index,:,:]
        elif index<512:
            index = index-256
            return self.x[:,:,index,:], self.y[:,:,index,:]
        else:
            index = index-512
            return self.x[:,:,:,index], self.y[:,:,:,index]
    def __len__(self):
        return 768
        
# DDP SETUP

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12360"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

class Trainer:
    def __init__(
        self,
        model,
        loaders,
        name,
        fixed_data,
        parent_dir='/study/mrphys/skunkworks/kk',
        gpu_id=0,
        norm_scale = 1,
        lr=1e-3,
    ):
        self.lossCounter = {
            'Training':{},
            'Testing':{},
        }
        self.trainLoader, self.testLoader = loaders
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.5)
        self.model = DDP(model, device_ids=[gpu_id])
        self.min_lr = lr/1000
        self.fixedX, self.fixedY = fixed_data
        self.norm_scale = norm_scale
        self.parent_dir = parent_dir
        self.name = name
        self.loss = torch.nn.MSELoss()
        #make directories for checkpoint
        os.makedirs(f'{self.parent_dir}/outputs/{self.name}', exist_ok=True)
        os.makedirs(f'{self.parent_dir}/outputs/{self.name}/logs', exist_ok=True)
        os.makedirs(f'{self.parent_dir}/outputs/{self.name}/weights', exist_ok=True)
        os.makedirs(f'{self.parent_dir}/outputs/{self.name}/lossPlot', exist_ok=True)
        os.makedirs(f'{self.parent_dir}/outputs/{self.name}/preds', exist_ok=True)
        
    def _batch_run(self, source, target, training=True):
        if training:
            self.model.train()
            self.optimizer.zero_grad()
            output = self.model(source)
            output = torch.abs(output)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()
        else:
            with torch.no_grad():
                self.model.eval()
                output = self.model(source)
                output = self.model(source)
                output = torch.abs(output)
                loss = self.loss(output, target)
        return loss.item()
    
    def _epoch_run(self, epoch, training=True):
        word = 'Training' if training else 'Testing'
        loader = self.trainLoader if training else self.testLoader
        loader.sampler.set_epoch(epoch)
        if not epoch in self.lossCounter[word].keys():
            self.lossCounter[word][epoch] = {'loss':0.0,'counter':1e-7,}   
        pbar = tqdm(enumerate(loader))
        for i, (source, target) in pbar:
            source = source.to(self.gpu_id)
            target = target.to(self.gpu_id).float()
            loss = self._batch_run(source, target, training=training)
            self.lossCounter[word][epoch]['loss'] += loss*source.shape[0]
            self.lossCounter[word][epoch]['counter'] += source.shape[0]
            meanLoss = self.lossCounter[word][epoch]['loss'] / self.lossCounter[word][epoch]['counter']
            pbar.set_description(f"GPU#[{self.gpu_id}] {word} Epoch : {epoch} [batch {i+1}/{len(loader)}] - loss = {round(meanLoss,6)}")
        if training:
            self._save(epoch)
    
    def _save(self, epoch):
        state_dict = self.model.module.state_dict()
        torch.save(state_dict, f'{self.parent_dir}/outputs/{self.name}/weights/{self.name}_LATEST.pth')
        torch.save(state_dict, f'{self.parent_dir}/outputs/{self.name}/weights/{self.name}_ep{epoch}.pth')
        with open(f'{self.parent_dir}/outputs/{self.name}/logs/{self.name}_logs.json', 'w') as f:
            json.dump(self.lossCounter, f)
        
    def _plot_loss(self, epoch):
        epochs = list(self.lossCounter['Training'].keys())
        tr_loss = [(self.lossCounter['Training'][epoch]['loss']/self.lossCounter['Training'][epoch]['counter']) for epoch in epochs]
        te_loss = [(self.lossCounter['Testing'][epoch]['loss']/self.lossCounter['Testing'][epoch]['counter']) for epoch in epochs]
        plt.plot(epochs, tr_loss, label='train loss')
        plt.plot(epochs, te_loss, label='test loss')
        plt.legend()
        plt.savefig(f'{self.parent_dir}/outputs/{self.name}/lossPlot/{self.name}_LossPlot_GPU{self.gpu_id}.png')
        plt.close()
        
    def _plot_sample(self, epoch):
        fixedX = torch.unsqueeze(torch.tensor(self.fixedX),0)
        fixedY = torch.unsqueeze(torch.tensor(self.fixedY),0)
        self.model.eval()
        fixedX = fixedX.to(self.gpu_id)
        with torch.no_grad():
            pred = torch.abs(self.model(fixedX))
        fixedX = fixedX.cpu()
        pred = pred.cpu()
        plt.gray()
        fig, ax = plt.subplots(1, 3, figsize=(9,3))
        ax[0].imshow(torch.abs(fixedX[0,0]))
        ax[0].axis('off')
        ax[1].imshow(torch.abs(pred[0,0]))
        ax[1].axis('off')
        ax[2].imshow(torch.abs(fixedY[0,0]))
        ax[2].axis('off')
        plt.savefig(f'{self.parent_dir}/outputs/{self.name}/preds/{self.name}_pred_{epoch}.png')
        plt.close()
        
    def train(self, epochs=50, lr_patience=10):
        best_loss = 1e10
        for epoch in range(epochs):
            self._epoch_run(epoch, training=True)
            self._epoch_run(epoch, training=False)
            self._plot_loss(epoch)
            self._plot_sample(epoch)
            last_loss = self.lossCounter['Testing'][epoch]['loss']/self.lossCounter['Testing'][epoch]['counter']
            if best_loss > last_loss:
                best_loss = last_loss
                patienceCounter = 0
                torch.save(self.model.module.state_dict(), f'{self.parent_dir}/outputs/{self.name}/weights/{self.name}_BEST.pth')
            else:
                patienceCounter += 1
                if patienceCounter>=lr_patience and self.scheduler.get_last_lr()[-1]>=self.min_lr:
                    self.scheduler.step()
                    
def prepare_dataloader(dataset, batch_size):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )
                    
def run(rank, world_size, name, gt=False, batch_size=32, folds=5):
    
    ddp_setup(rank, world_size)
    traintestData  = []
    pbar = tqdm(range(len(T1path[0:40])), desc="loading datasets")
    for i in pbar:
        if gt:
            with open(f'/scratch/mrphys/T1pickled/GT_T1dataset_{i}.pickle', 'rb') as f:
                data = pickle.load(f)
                traintestData.append(data)
                del data  
        else:
            with open(f'/scratch/mrphys/T1pickled/Denoise_T1dataset_{i}.pickle', 'rb') as f:
                data = pickle.load(f)
                traintestData.append(data)
                del data  
                
    kfsplitter = kf(n_splits=folds, shuffle=True, random_state=69420)
    for i, (train_index, test_index) in enumerate(kfsplitter.split(traintestData)):
        fold = i+1
        model = unet.UNet(
            10,
            1,
            f_maps=16,
            layer_order=['separable convolution', 'relu', 'batch norm'],
            depth=4,
            layer_growth=2.0,
            residual=True,
            complex_input=True,
            complex_kernel=True,
            ndims=2,
            padding=1
        )
        trainData = [traintestData[i] for i in train_index]
        testData = [traintestData[i] for i in test_index]
        trainDataset = torch.utils.data.ConcatDataset(trainData)
        testDataset = torch.utils.data.ConcatDataset(testData)
        trainDataloader = prepare_dataloader(trainDataset, batch_size)
        testDataloader = prepare_dataloader(testDataset, batch_size)
        fixed_data = testDataset[150]
        trainer = Trainer(
            model,
            [trainDataloader, testDataloader],
            f'{name}_{fold}',
            fixed_data,
            gpu_id = rank,
        )
        trainer.train()
    destroy_process_group()
                    
if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Param Parser')
    parser.add_argument('--name', help='model name', default='T1_Denoised')
    parser.add_argument('--gt', help='from 9 minutes PCA or not', default=0, type=int)
    parser.add_argument('--batchsize', help='batch size of training and testing data', default=32, type=int)
    parser.add_argument('--num_gpu', help='number of gpus', default=torch.cuda.device_count()//2, type=int)
    args = parser.parse_args()
    name = args.name
    gt = bool(int(args.gt))
    batch_size = int(args.batchsize)
    world_size = int(args.num_gpu)
    print(f'Training model name = {name} with GT enabled = {gt} at batch size = {batch_size} using {world_size} gpus!')
    mp.spawn(run, args=(world_size, name, gt, batch_size), nprocs=world_size)