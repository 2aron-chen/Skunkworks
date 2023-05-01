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
from smoothing import GaussianSmoothing

import warnings
warnings.filterwarnings("ignore")
        
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
        self.smoother = GaussianSmoothing(1, 10, 0.2, 2).to(gpu_id)
        self.evaluations = {
            1:{"% diff mean":[],"% diff sd":[],"% abs diff mean":[],"% abs diff sd":[]},
            2:{"% diff mean":[],"% diff sd":[],"% abs diff mean":[],"% abs diff sd":[]},
            3:{"% diff mean":[],"% diff sd":[],"% abs diff mean":[],"% abs diff sd":[]},
        }
        self.mapnames = {
            1:"CSF",
            2:"Gray Matter",
            3:"White Matter",
        }
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
        
    def _evaluate(self, preds, trues, maps):
        if maps is None:
            return None
        with torch.no_grad():
            _, _, y_true, mask = trues
            _, y_pred_denoised, _, mask_d_logit, _ = preds
            y_pred_denoised = self.smoother((y_pred_denoised*torch.sigmoid(mask_d_logit)).float(), mask)
            y_true = self.smoother(y_true.float(), mask)
            for i in maps.keys():
                condition = mask==i
                y_pred_denoised_masked = y_pred_denoised[condition]
                y_true_masked = y_true[condition]
                maps[i]["denoised_t1"] += list(y_pred_denoised_masked.cpu().reshape(-1))
                maps[i]["t1"] += list(y_true_masked.cpu().reshape(-1))
        return maps
        
    def _batch_run(self, data, maps, training=True):
        xgt, ymask = data
        x = xgt[:,0:10].to(self.gpu_id)
        gt = xgt[:,10:20].to(self.gpu_id)
        y = ymask[:,0:1].to(self.gpu_id)
        mask = ymask[:,1:2].to(self.gpu_id)
        trues = (x, gt, y, mask)
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
        maps = self._evaluate(preds, trues, maps)
        return losses, x.shape[0], maps
    
    def _epoch_run(self, epoch, training=True):
        word = 'Training' if training else 'Testing'
        loader = self.trainLoader if training else self.testLoader
        loader.sampler.set_epoch(epoch)
        if not epoch in self.lossCounter[word].keys():
            self.lossCounter[word][epoch] = {
                'counter':1e-7,
                'loss':{
                    "ssim":0.0,
                    "error_denoiser":0.0,
                    "error_groundtruth":0.0,
                    "mask":0.0,
                    "total":0.0,
                },
            }   
        pbar = tqdm(enumerate(loader)) if self.gpu_id == 0 else enumerate(loader)
        maps = {
            1:{"denoised_t1":[],"t1":[]},
            2:{"denoised_t1":[],"t1":[]},
            3:{"denoised_t1":[],"t1":[]},
        } if not training else None
        for i, data in pbar:
            losses, count, maps = self._batch_run(data, maps, training=training)
            self.lossCounter[word][epoch]['counter'] += count
            lossmsg = "[s/d/gt/m/t] = "
            for lossname in losses.keys():
                self.lossCounter[word][epoch]['loss'][lossname] += losses[lossname].item()*count
                meanLoss = self.lossCounter[word][epoch]['loss'][lossname]/self.lossCounter[word][epoch]['counter']
                lossmsg += f"[{round(meanLoss,5)}]"
            if self.gpu_id == 0:
                pbar.set_description(f"[{word} [e={epoch}, b={i+1}/{len(loader)}] - {lossmsg}")
        if training:
            self._save(epoch)
            
        if not training:   
            for i in maps.keys():
                mapname = self.mapnames[i]
                pred = np.array(maps[i]["denoised_t1"])
                true = np.array(maps[i]["t1"])
                diff = (pred-true)/true
                self.evaluations[i]["% abs diff mean"] += [np.mean(np.abs(diff))*100]
                self.evaluations[i]["% abs diff sd"] += [np.std(np.abs(diff))*100]
                self.evaluations[i]["% diff mean"] += [np.mean(diff)*100]
                self.evaluations[i]["% diff sd"] += [np.std(diff)*100]
                
                # diff mean   
                mean = np.array(self.evaluations[i]["% diff mean"])
                sd = np.array(self.evaluations[i]["% diff sd"])
                plt.plot(mean, color='black', label="mean difference")
                plt.fill_between(np.arange(len(mean)), mean-sd, mean+sd, color='blue', alpha=0.2, label='std difference')
                plt.legend()
                plt.title(f"{mapname} error plot (% difference)")
                plt.savefig(f'{self.parent_dir}/outputs/{self.name}/logs/_PLOT_{self.name}_{mapname}_diff.png')
                plt.close()
                
                mean = np.array(self.evaluations[i]["% abs diff mean"])
                sd = np.array(self.evaluations[i]["% abs diff sd"])
                plt.plot(mean, color='black', label="mean difference")
                plt.fill_between(np.arange(len(mean)), mean-sd, mean+sd, color='blue', alpha=0.2, label='std difference')
                plt.legend()
                plt.title(f"{mapname} error plot (% difference)")
                plt.savefig(f'{self.parent_dir}/outputs/{self.name}/logs/_PLOT_{self.name}_{mapname}_abs_diff.png')
                plt.close()

        del maps
    
    def _save(self, epoch):
        state_dict = self.model.module.state_dict()
        torch.save(state_dict, f'{self.parent_dir}/outputs/{self.name}/weights/{self.name}_LATEST.pth')
        torch.save(state_dict, f'{self.parent_dir}/outputs/{self.name}/weights/{self.name}_ep{epoch:03d}.pth')
        with open(f'{self.parent_dir}/outputs/{self.name}/logs/{self.name}_logs_GPU{self.gpu_id}.json', 'w') as f:
            json.dump(self.lossCounter, f)
        with open(f'{self.parent_dir}/outputs/{self.name}/logs/{self.name}_evals_GPU{self.gpu_id}.json', 'w') as f:
            json.dump(self.evaluations, f)
        
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
            (denoised, y_pred_denoised, y_pred_gt, mask_d_logit, mask_gt_logit), _, _ = self.model(x,gt,y,mask)
        
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
        fig, ax = plt.subplots(2, 4, figsize=(12,6))
        ax[0,0].imshow(torch.abs(x[0,0]))
        ax[0,0].axis('off')
        ax[0,1].imshow(torch.abs(y_pred_denoised[0,0]), vmin=np.min(y.numpy()), vmax=np.max(y.numpy()))
        ax[0,1].axis('off')
        ax[0,2].imshow(torch.abs(y[0,0]), vmin=np.min(y.numpy()), vmax=np.max(y.numpy()))
        ax[0,2].axis('off')
        ax[0,3].imshow(torch.sigmoid(mask_d_logit.cpu()[0,0]), vmin=0, vmax=1)
        ax[0,3].axis('off')
        ax[1,0].imshow(torch.abs(gt[0,0]))
        ax[1,0].axis('off')
        ax[1,1].imshow(torch.abs(y_pred_gt[0,0]), vmin=np.min(y.numpy()), vmax=np.max(y.numpy()))
        ax[1,1].axis('off')
        ax[1,2].imshow(torch.abs(y[0,0]), vmin=np.min(y.numpy()), vmax=np.max(y.numpy()))
        ax[1,2].axis('off')
        ax[1,3].imshow(torch.sigmoid(mask_gt_logit.cpu()[0,0]), vmin=0, vmax=1)
        ax[1,3].axis('off')
        plt.savefig(f'{self.parent_dir}/outputs/{self.name}/preds/{self.name}_T1_pred_{epoch:03d}.png')
        plt.close()
        
    def train(self, epochs=100, lr_patience=10):
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
                    
def run(rank, world_size, name, epochs=100, batch_size=16, precomputed=False, num_samples=65, folds=5, nchans=10): 
    ddp_setup(rank, world_size)
    try:
        traintestData = []
        if not precomputed:
            T1path = sorted(glob('/study/mrphys/skunkworks/training_data/mover01/*/processed_data/T1_3_tv.nii'))
            samples = [p.split('/')[6] for p in T1path]
            for i in tqdm(range(len(samples[0:num_samples]))):
                with open(f'/scratch/mrphys/pickled/fullDataset_{i}.pickle', 'rb') as f:
                    data = pickle.load(f)
                    traintestData.append(data)  
        else:
            T1path = sorted(glob('/study/mrphys/skunkworks/training_data/mover01/*/processed_data/T1_3_tv.nii'))
            samples = [p.split('/')[6] for p in T1path]
            class FullDataset(Dataset):
                def __init__(self, index, norm_factor=1000):
                    self.name = samples[index]
                def __getitem__(self, index):
                    x,y,gt,mask = np.load(f'/scratch/mrphys/fullDataset/{self.name}/{index}.npy', allow_pickle=True)
                    return np.concatenate([x,gt],axis=0), np.concatenate([y, mask],axis=0)
                def __len__(self):
                    return 768
            for i in tqdm(range(len(samples[0:num_samples]))):
                data = FullDataset(i)
                traintestData.append(data)
                    
        kfsplitter = kf(n_splits=folds, shuffle=True, random_state=69420)
        for i, (train_index, test_index) in enumerate(kfsplitter.split(traintestData)):
            if rank==0:
                print(f'Train Index = {train_index}\nTest Index = {test_index}')
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
                f'{name}_{fold}',
                fixed_data,
                gpu_id = rank,
            )
            trainer.train(epochs = epochs)
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
    parser.add_argument('--precomputed', help='use precomputed numpy', default=1, type=int)
    parser.add_argument('--samples', help='number of samples', default=65, type=int)
    args = parser.parse_args()
    precomputed = bool(args.precomputed)
    samples = int(args.samples)
    print(f'Training model "{args.name}" at batch = {int(args.batchsize)} using {int(args.num_gpu)} gpus for {int(args.epochs)} epochs!') 
    print(f'Use precomputed values = {precomputed} on {samples} samples!')
    mp.spawn(run, args=(int(args.num_gpu), args.name, int(args.epochs), int(args.batchsize), precomputed, samples), nprocs=int(args.num_gpu))