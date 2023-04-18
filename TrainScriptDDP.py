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
from fullmodel import fullModel
from torchvision import transforms
from torch.utils.data import Dataset
from sklearn.model_selection import KFold as kf
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from mriDataset import mriSliceDataset

# Dataset
T1path = sorted(glob('/study/mrphys/skunkworks/training_data/mover01/*/processed_data/T1_3_tv.nii'))
samples = [p.split('/')[6] for p in T1path]
class FullDataset(Dataset):   
    def __getitem__(self, index):
        if index<256:
            return self.xgt[:,index,:,:], self.ymask[:,index,:,:]
        elif index<512:
            index = index-256
            return self.xgt[:,:,index,:], self.ymask[:,:,index,:]
        else:
            index = index-512
            return self.xgt[:,:,:,index], self.ymask[:,:,:,index]
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
    os.environ["MASTER_PORT"] = "12369"
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
        bkg_lambda = 0.1,
        lr=1e-3,
    ):
        self.lossCounter = {
            'Training':{},
            'Testing':{},
        }
        self.trainLoader, self.testLoader = loaders
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.model = DDP(self.model, device_ids=[gpu_id])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.5)
        
        self.bkg_lambda = bkg_lambda
        self.min_lr = lr/1000
        self.training = True
        self.fixed_data = fixed_data
        self.norm_scale = norm_scale
        self.parent_dir = parent_dir
        self.name = name
        #make directories for checkpoint
        os.makedirs(f'{self.parent_dir}/outputs/{self.name}', exist_ok=True)
        os.makedirs(f'{self.parent_dir}/outputs/{self.name}/logs', exist_ok=True)
        os.makedirs(f'{self.parent_dir}/outputs/{self.name}/weights', exist_ok=True)
        os.makedirs(f'{self.parent_dir}/outputs/{self.name}/lossPlot', exist_ok=True)
        os.makedirs(f'{self.parent_dir}/outputs/{self.name}/preds', exist_ok=True)
        
    def _batch_run(self, data, training=True):
        xgt, ymask = data
        x = xgt[:,0:10].to(self.gpu_id)
        gt = xgt[:,10:20].to(self.gpu_id)
        y = ymask[:,0:1].to(self.gpu_id)
        mask = ymask[:,1:2].to(self.gpu_id)
        if training:
            self.model.train()
            self.optimizer.zero_grad()
            preds, losses, loss = self.model(x, gt, y, mask)
            loss.backward()
            self.optimizer.step()
        else:
            with torch.no_grad():
                self.model.eval()
                preds, losses, loss = self.model(x, gt, y, mask)
        del preds
        return losses, x.shape[0]
    
    def _epoch_run(self, epoch, training=True):
        word = 'Training' if training else 'Testing'
        loader = self.trainLoader if training else self.testLoader
        loader.sampler.set_epoch(epoch)
        if not epoch in self.lossCounter[word].keys():
            self.lossCounter[word][epoch] = {
                'counter':1e-7,
                'loss':{
                    "ssim":0.0,
                    "l2_denoiser":0.0,
                    "l2_groundtruth":0.0,
                    "total":0.0,
                },
            }   
        pbar = tqdm(enumerate(loader))
        for i, data in pbar:
            losses, count = self._batch_run(data, training=training)
            self.lossCounter[word][epoch]['counter'] += count
            lossmsg = "[s/d/gt/t] = "
            for lossname in losses.keys():
                self.lossCounter[word][epoch]['loss'][lossname] += losses[lossname].item()*count
                meanLoss = self.lossCounter[word][epoch]['loss'][lossname]/self.lossCounter[word][epoch]['counter']
                lossmsg += f"[{round(meanLoss,6)}]"
            pbar.set_description(f"[GPU #{self.gpu_id}] {word} Epoch : {epoch} [batch {i+1}/{len(loader)}] - {lossmsg}")
        if training:
            self._save(epoch)
    
    def _save(self, epoch):
        state_dict = self.model.module.state_dict()
        torch.save(state_dict, f'{self.parent_dir}/outputs/{self.name}/weights/{self.name}_LATEST.pth')
        torch.save(state_dict, f'{self.parent_dir}/outputs/{self.name}/weights/{self.name}_ep{epoch:03d}.pth')
        with open(f'{self.parent_dir}/outputs/{self.name}/logs/{self.name}_logs_GPU{self.gpu_id}.json', 'w') as f:
            json.dump(self.lossCounter, f)
        
    def _plot_loss(self, epoch):
        for lossname in self.lossCounter['Training'][0]['loss'].keys():
            epochs = list(self.lossCounter['Training'].keys())
            tr_loss = [(self.lossCounter['Training'][epoch]['loss'][lossname]/self.lossCounter['Training'][epoch]['counter']) for epoch in epochs]
            te_loss = [(self.lossCounter['Testing'][epoch]['loss'][lossname]/self.lossCounter['Testing'][epoch]['counter']) for epoch in epochs]
            plt.plot(epochs, tr_loss, label='train loss')
            plt.plot(epochs, te_loss, label='test loss')
            plt.legend()
            plt.savefig(f'{self.parent_dir}/outputs/{self.name}/lossPlot/{self.name}_{lossname}_LossPlot.png')
            plt.close()
        
    def _plot_sample(self, epoch):
        xgt, ymask = self.fixed_data
        x = xgt[0:10]
        gt = xgt[10:20]
        y = ymask[0:1]
        mask = ymask[1:2]
        x = torch.unsqueeze(torch.tensor(x),0).to(self.gpu_id)
        gt = torch.unsqueeze(torch.tensor(gt),0).to(self.gpu_id)
        y = torch.unsqueeze(torch.tensor(y),0).to(self.gpu_id)
        mask = torch.unsqueeze(torch.tensor(mask),0).to(self.gpu_id)
        with torch.no_grad():
            self.model.eval()
            (denoised, y_pred_denoised, y_pred_gt), _, _ = self.model(x,gt,y,mask)
        
        # plot denoiser's progress
        x = x.cpu()
        gt = gt.cpu()
        denoised = denoised.cpu()
        plt.gray()
        fig, ax = plt.subplots(3, 6, figsize=(20,6)) 
        for i in range(6):
            ax[0,i].imshow(torch.abs(x[0,i]))
            ax[0,i].axis('off')
        for i in range(6):
            ax[1,i].imshow(torch.abs(denoised[0,i]))
            ax[1,i].axis('off')
        for i in range(6):
            ax[2,i].imshow(torch.abs(gt[0,i]))
            ax[2,i].axis('off')
        plt.savefig(f'{self.parent_dir}/outputs/{self.name}/preds/{self.name}_denoiser_pred_{epoch:03d}.png')
        plt.close()
        
        # plot T1's progress
        plt.gray()
        y_pred_denoised = y_pred_denoised.cpu()
        y_pred_gt = y_pred_gt.cpu()
        y = y.cpu()
        fig, ax = plt.subplots(2, 3, figsize=(9,6))
        ax[0,0].imshow(torch.abs(x[0,0]))
        ax[0,0].axis('off')
        ax[0,1].imshow(torch.abs(y_pred_denoised[0,0]), vmin=np.min(y.numpy()), vmax=np.max(y.numpy()))
        ax[0,1].axis('off')
        ax[0,2].imshow(torch.abs(y[0,0]), vmin=np.min(y.numpy()), vmax=np.max(y.numpy()))
        ax[0,2].axis('off')
        ax[1,0].imshow(torch.abs(gt[0,0]))
        ax[1,0].axis('off')
        ax[1,1].imshow(torch.abs(y_pred_gt[0,0]), vmin=np.min(y.numpy()), vmax=np.max(y.numpy()))
        ax[1,1].axis('off')
        ax[1,2].imshow(torch.abs(y[0,0]), vmin=np.min(y.numpy()), vmax=np.max(y.numpy()))
        ax[1,2].axis('off')
        plt.savefig(f'{self.parent_dir}/outputs/{self.name}/preds/{self.name}_T1_pred_{epoch:03d}.png')
        plt.close()
        
    def train(self, epochs=200, lr_patience=10):
        best_loss = 1e10
        patienceCounter = 0
        for epoch in range(epochs):
            self._epoch_run(epoch, training=True)
            self._epoch_run(epoch, training=False)
            self._plot_loss(epoch)
            self._plot_sample(epoch)
            
            last_loss = self.lossCounter['Testing'][epoch]['loss']["total"]/self.lossCounter['Testing'][epoch]['counter']
            
            if best_loss > last_loss:
                torch.save(self.model.module.state_dict(), f'{self.parent_dir}/outputs/{self.name}/weights/{self.name}_BEST.pth')
                patienceCounter = 0
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
                    
def run(rank, world_size, name, epochs=100, batch_size=16, folds=5, nchans=10): 
    ddp_setup(rank, world_size)
    try:
        traintestData = []
        for i in tqdm(range(len(samples[0:35]))):
            with open(f'/scratch/mrphys/pickled/fullDataset_{i}.pickle', 'rb') as f:
                data = pickle.load(f)
                traintestData.append(data)  
        kfsplitter = kf(n_splits=folds, shuffle=True, random_state=69420)
        for i, (train_index, test_index) in enumerate(kfsplitter.split(traintestData)):
            print(f'Rank = {rank} -> Test Index = {test_index}')
            fold = i+1
            trainData = [traintestData[i] for i in train_index]
            testData = [traintestData[i] for i in test_index]
            trainDataset = torch.utils.data.ConcatDataset(trainData)
            testDataset = torch.utils.data.ConcatDataset(testData)
            trainDataloader = prepare_dataloader(trainDataset, batch_size, True, world_size, rank)
            testDataloader = prepare_dataloader(testDataset, batch_size, False, world_size, rank)
            fixed_data = testDataset[150]
            trainer = Trainer(
                fullModel(nchans=nchans),
                [trainDataloader, testDataloader],
                f'fullModel_{fold}',
                fixed_data,
                gpu_id = rank,
            )
            trainer.train()
    except Exception as e:
        print('Error occured : '+ str(e))
    finally:
        destroy_process_group()
                    
if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Param Parser')
    parser.add_argument('--name', help='model name', default='fullModel')
    parser.add_argument('--epochs', help='no. of epochs', default=100, type=int)
    parser.add_argument('--batchsize', help='batch size of training and testing data', default=32, type=int)
    parser.add_argument('--num_gpu', help='number of gpus', default=torch.cuda.device_count()//2, type=int)
    args = parser.parse_args()
    print(f'Training model "{args.name}" at batch = {int(args.batchsize)} using {int(args.num_gpu)} gpus for {int(args.epochs)} epochs!') 
    mp.spawn(run, args=(int(args.num_gpu), args.name, int(args.epochs), int(args.batchsize)), nprocs=int(args.num_gpu))