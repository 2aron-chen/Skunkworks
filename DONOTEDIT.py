import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from glob import glob
from tqdm import tqdm
import pickle

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='train script')
    parser.add_argument('fold', type=int, help='which fold of 1-5')
    args = parser.parse_args()

    allImages = sorted(glob("/study/mrphys/skunkworks/training_data//mover01/*/", recursive=True))
    traintestData = []

    pbar = tqdm(range(len(allImages)), desc="loading datasets")
    
    for i in pbar:
        with open(f'/scratch/mrphys/pickled/dataset_{i}.pickle', 'rb') as f:
            data = pickle.load(f)
            traintestData.append(data)
            del data

    kfsplitter = kf(n_splits=5, shuffle=True, random_state=69420)

    for i, (train_index, test_index) in enumerate(kfsplitter.split(traintestData)):
        
        fold = i+1
        if fold!=args.fold:
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