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
import nibabel as nib
sys.path.insert(0,"/study/mrphys/skunkworks/kk/mriUnet")
from unet import UNet
from torchvision import transforms
from torch.utils.data import Dataset
from sklearn.model_selection import KFold as kf
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
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
    os.environ["MASTER_PORT"] = "42069"
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
        bkg_lambda = 1e-4,
        lr=1e-3,
        smoothen=False,
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
        if smoothen:
            self.smoother = GaussianSmoothing(1, 1, 0.2, 3).to(gpu_id)
        else:
            self.smoother = nn.Identity()
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
        self.parent_dir = parent_dir
        self.name = name
        #make directories for checkpoint
        os.makedirs(f'{self.parent_dir}/outputs/{self.name}', exist_ok=True)
        os.makedirs(f'{self.parent_dir}/outputs/{self.name}/logs', exist_ok=True)
        os.makedirs(f'{self.parent_dir}/outputs/{self.name}/weights', exist_ok=True)
        os.makedirs(f'{self.parent_dir}/outputs/{self.name}/lossPlot', exist_ok=True)
        os.makedirs(f'{self.parent_dir}/outputs/{self.name}/preds', exist_ok=True)
        
        self.mse = nn.MSELoss()
        
    def _evaluate(self, pred, true, mask, maps):
        if maps is None:
            return None
        with torch.no_grad():
            for i in maps.keys():
                condition = mask==i
                pred_masked = pred[condition]
                true_masked = true[condition]
                maps[i]["denoised_t1"] += list(pred_masked.cpu().reshape(-1))
                maps[i]["t1"] += list(true_masked.cpu().reshape(-1))
        return maps
        
    def _batch_run(self, data, maps, training=True):
        X, Y, mask = data
        M = torch.clone(mask)
        M[M!=0] = 1.0
        M[M==0] = self.bkg_lambda
        M = M.to(self.gpu_id).float()
        
        X = X.to(self.gpu_id).float()*M
        Y = Y.to(self.gpu_id).float()*M
        if training:
            self.model.train()
            self.optimizer.zero_grad()
            pred = self.model(X)
            loss = self.mse(self.smoother(pred), self.smoother(Y))
            loss.backward()
            self.optimizer.step()
        else:
            with torch.no_grad():
                self.model.eval()
                pred = self.model(X)
                loss = self.mse(self.smoother(pred), self.smoother(Y))

        maps = self._evaluate(self.smoother(pred), self.smoother(Y), mask, maps)
        losses = {
            "MSELoss":loss
        }
        return losses, X.shape[0], maps
    
    def _epoch_run(self, epoch, training=True):
        word = 'Training' if training else 'Testing'
        loader = self.trainLoader if training else self.testLoader
        loader.sampler.set_epoch(epoch)
        if not epoch in self.lossCounter[word].keys():
            self.lossCounter[word][epoch] = {
                'counter':1e-7,
                'loss':{
                    "MSELoss":0.0
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
            lossmsg = "Losses = "
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
        N = [40, 60, 80, 100, 120, 140, 160, 180, 200]
        X, Y, mask = self.fixed_data
        M = torch.unsqueeze(torch.tensor(mask),0)
        M[M!=0] = 1.0
        M[M==0] = self.bkg_lambda
        M = M.to(self.gpu_id).float()
        X = torch.unsqueeze(torch.tensor(X),0).to(self.gpu_id).float()*M
        Y = torch.unsqueeze(torch.tensor(Y),0).to(self.gpu_id).float()*M
        self.model.eval()
        with torch.no_grad():
            pred = self.model(X)
        
        # plot denoiser's progress
        X = X.cpu()
        Y = Y.cpu()
        pred = pred.cpu()
        plt.gray()
        fig, ax = plt.subplots(3, len(N), figsize=(int(10*len(N)/3),6)) 
        for i in range(len(N)):
            n = N[i]
            ax[0,i].imshow(X[0,0,n])
            ax[0,i].axis('off')
            ax[1,i].imshow(Y[0,0,n])
            ax[1,i].axis('off')
            ax[2,i].imshow(pred[0,0,n])
            ax[2,i].axis('off')
        ax[0,0].set_ylabel("Noisy")
        ax[1,0].set_ylabel("Ground Truth")
        ax[2,0].set_ylabel("Denoised")
        plt.savefig(f'{self.parent_dir}/outputs/{self.name}/preds/{self.name}_pred_{epoch:03d}.png')
        plt.close()
        
    def train(self, epochs=100, lr_patience=10):
        best_loss = 1e10
        patienceCounter = 0
        for epoch in range(epochs):
            self._epoch_run(epoch, training=True)
            self._epoch_run(epoch, training=False)
            self._plot_loss(epoch)
            self._plot_sample(epoch)
            
            last_loss = self.lossCounter['Testing'][epoch]['loss']["MSELoss"]/self.lossCounter['Testing'][epoch]['counter']
            
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
                    
def run(rank, world_size, name, epochs=100, batch_size=1, num_samples=65, folds=5, nchans=10): 
    ddp_setup(rank, world_size)
    try:
        class T1Dataset(Dataset):
    
            def __init__(self, indices):
                self.noisy_path = sorted(glob("/study/mrphys/skunkworks/training_data/mover01/*/processed_data/acc_2min/T1_3_tv.nii"))
                self.groundtruth_path = sorted(glob("/study/mrphys/skunkworks/training_data/mover01/*/processed_data/T1_3_tv.nii"))
                self.mask_path = sorted(glob('/scratch/mrphys/masks/*'))
                self.indices = indices

            def __getitem__(self, index):
                index = self.indices[index]
                X = (np.transpose(nib.load(self.noisy_path[index]).get_fdata())/1000).reshape(1,256,256,256)
                Y = (np.transpose(nib.load(self.groundtruth_path[index]).get_fdata())/1000).reshape(1,256,256,256)
                mask = np.load(self.mask_path[index]).reshape(1,256,256,256)
                return X, Y, mask

            def __len__(self):
                return len(self.indices)
                    
        kfsplitter = kf(n_splits=folds, shuffle=True, random_state=69420)
        for i, (train_index, test_index) in enumerate(kfsplitter.split(list(range(num_samples)))):
            if i in [3]:
                continue
            if rank==0:
                print(f'Train Index = {train_index}\nTest Index = {test_index}')
            fold = i+1
            trainDataset = T1Dataset(train_index)
            testDataset = T1Dataset(test_index)
            trainDataloader = prepare_dataloader(trainDataset, batch_size, True, world_size, rank)
            testDataloader = prepare_dataloader(testDataset, batch_size, False, world_size, rank)
            fixed_data = testDataset[0]
            
            model = UNet(
                1,
                1,
                f_maps=16,
                layer_order=['separable convolution', 'relu'],
                depth=4,
                layer_growth=2.0,
                residual=True,
                complex_input=False,
                complex_kernel=False,
                ndims=3,
                padding=1
            )
            
            trainer = Trainer(
                model,
                [trainDataloader, testDataloader],
                f'{name}_{fold}',
                fixed_data,
                gpu_id = rank,
                lr=1e-2,
            )
            trainer.train(epochs = epochs)
    except Exception as e:
        print('Error occured : '+ str(e))
    finally:
        destroy_process_group()
                    
if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Param Parser')
    parser.add_argument('--name', help='model name', default='T1denoiser_3d')
    parser.add_argument('--epochs', help='no. of epochs', default=200, type=int)
    parser.add_argument('--batchsize', help='batch size of training and testing data', default=1, type=int)
    parser.add_argument('--num_gpu', help='number of gpus', default=torch.cuda.device_count(), type=int)
    parser.add_argument('--samples', help='number of samples', default=65, type=int)
    args = parser.parse_args()
    samples = int(args.samples)
    print(f'Training model "{args.name}" at batch = {int(args.batchsize)} using {int(args.num_gpu)} gpus for {int(args.epochs)} epochs!') 
    print(f'Training on {samples} samples!')
    mp.spawn(run, args=(int(args.num_gpu), args.name, int(args.epochs), int(args.batchsize), samples), nprocs=int(args.num_gpu))