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
from mriDataset import mriSliceDataset
        
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
        #make directories for checkpoint
        os.makedirs(f'{self.parent_dir}/outputs/{self.name}', exist_ok=True)
        os.makedirs(f'{self.parent_dir}/outputs/{self.name}/logs', exist_ok=True)
        os.makedirs(f'{self.parent_dir}/outputs/{self.name}/weights', exist_ok=True)
        os.makedirs(f'{self.parent_dir}/outputs/{self.name}/lossPlot', exist_ok=True)
        os.makedirs(f'{self.parent_dir}/outputs/{self.name}/preds', exist_ok=True)
        
    def get_ssim(self, pred, true):
        loss = (1-ms_ssim(pred.real, true.real, data_range=self.norm_scale, size_average=False)).mean() + (1-ms_ssim(pred.imag, true.imag, data_range=self.norm_scale, size_average=False)).mean()
        return loss
        
    def _batch_run(self, source, target, training=True):
        if training:
            self.model.train()
            self.optimizer.zero_grad()
            true = torch.sigmoid(target)
            loss = self.get_ssim(torch.sigmoid(self.model(source)), true) + self.get_ssim(torch.sigmoid(self.model(target)), true)
            loss.backward() 
            self.optimizer.step()
        else:
            with torch.no_grad():
                self.model.eval()
                output = self.model(source)
                true = torch.sigmoid(target)
                loss = self.get_ssim(torch.sigmoid(self.model(source)), true) + self.get_ssim(torch.sigmoid(self.model(target)), true)
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
            target = target.to(self.gpu_id)
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
        torch.save(state_dict, f'{self.parent_dir}/outputs/{self.name}/weights/{self.name}_ep{epoch:03d}.pth')
        with open(f'{self.parent_dir}/outputs/{self.name}/logs/{self.name}_logs_GPU{self.gpu_id}.json', 'w') as f:
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
            pred = self.model(fixedX)
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
        plt.savefig(f'{self.parent_dir}/outputs/{self.name}/preds/{self.name}_pred_{epoch:03d}.png')
        plt.close()
        
    def train(self, epochs=100, lr_patience=10):
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
                    
def prepare_dataloader(dataset, batch_size, shuffle, world_size, rank):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset,num_replicas=world_size,rank=rank,shuffle=shuffle,seed=69420)
    )
                    
def run(rank, world_size, epochs, folds=5, batch_size=32): 
    ddp_setup(rank, world_size)
    try:
        allImages = sorted(glob("/study/mrphys/skunkworks/training_data//mover01/*/", recursive=True))[0:40]
        traintestData  = []
        pbar = tqdm(range(len(allImages)), desc="loading datasets")
        for i in pbar:
            with open(f'/scratch/mrphys/pickled/dataset_{i}.pickle', 'rb') as f:
                data = pickle.load(f)
                traintestData.append(data)
                del data  
        kfsplitter = kf(n_splits=folds, shuffle=True, random_state=69420)
        for i, (train_index, test_index) in enumerate(kfsplitter.split(traintestData)):
            fold = i+1
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
            trainData = [traintestData[i] for i in train_index]
            testData = [traintestData[i] for i in test_index]
            trainDataset = torch.utils.data.ConcatDataset(trainData)
            testDataset = torch.utils.data.ConcatDataset(testData)
            trainDataloader = prepare_dataloader(trainDataset, batch_size, True, world_size, rank)
            testDataloader = prepare_dataloader(testDataset, batch_size, False, world_size, rank)
            fixed_data = testDataset[150]
            trainer = Trainer(
                model,
                [trainDataloader, testDataloader],
                f'fullDenoiser_{fold}',
                fixed_data,
                gpu_id = rank,
            )
            trainer.train(epochs=epochs)
    except Exception as e:
        print('Error occured '+ str(e))
    finally:
        destroy_process_group()
                    
if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Param Parser')
    parser.add_argument('--num_gpu', help='number of gpus', default=torch.cuda.device_count()//2, type=int)
    parser.add_argument('--epochs', help='number of epochs', default=50, type=int)
    args = parser.parse_args()
    world_size = int(args.num_gpu)
    epochs = int(args.epochs)
    mp.spawn(run, args=(world_size, epochs), nprocs=world_size)