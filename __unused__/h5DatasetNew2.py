from torch.utils.data import Dataset
import h5py
import numpy as np
from glob import glob
from matplotlib import pyplot as plt

class h5DatasetNew(Dataset):
    def __init__(self, train):
        self.orginalPathList = []
        self.accelPathList = []
        self.orginalFileList = []
        self.accelFileList = []
        # self.mid = int(256/2) - 3  ## minus three because we are taking the middle 8 slices
        plt.rcParams["figure.figsize"] = [10, 10]
        plt.rcParams["figure.autolayout"] = True

        allImages = sorted(glob("/study/mrphys/skunkworks/training_data//mover01/*/", recursive=True))
        if train == True:
            # for folderName in allImages[:55]:
            for folderName in allImages[:5]: ## for testing
                self.orginalPathList.append(folderName + 'processed_data/C.h5')
                self.accelPathList.append(folderName +'processed_data/acc_2min/C.h5')
        else:
            # for folderName in allImages[55:]:
            for folderName in allImages[64:]: ## for testing
                self.orginalPathList.append(folderName + 'processed_data/C.h5')
                self.accelPathList.append(folderName +'processed_data/acc_2min/C.h5')
        
        for orginalPath, accelPath in zip(self.orginalPathList, self.accelPathList):
            prefix = 'C_000_0'
            orginalImageNumpy_Stack = None
            accelImageNumpy_Stack = None
            with h5py.File(orginalPath,'r') as hf:
                channel_one_max = abs(hf['Images']['C_000_000']['real']).max()
                for i in range(6):
                    n = prefix + str(i).zfill(2)
                    image = hf['Images'][n]
                    imageNumpy = image['real']
                    
                    imageNumpy = imageNumpy * (1/(channel_one_max))
                    orginalImageNumpy = np.array(imageNumpy + 0j*image['imag'])
                    if i == 0:
                        orginalImageNumpy_Stack = np.expand_dims(np.copy(orginalImageNumpy), axis=0)
                    else:
                        orginalImageNumpy_Stack = np.concatenate((orginalImageNumpy_Stack, np.expand_dims(orginalImageNumpy, axis=0)), axis=0)

            
            with h5py.File(accelPath,'r') as hf:
                channel_one_max = abs(hf['Images']['C_000_000']['real']).max()
                for i in range(6):
                    n = prefix + str(i).zfill(2)
                    image = hf['Images'][n]

                    imageNumpy = image['real']
                    imageNumpy = imageNumpy * (1/(channel_one_max))
                    accelImageNumpy = np.array(imageNumpy + 0j*image['imag'])
                    if i == 0:
                        accelImageNumpy_Stack = np.expand_dims(np.copy(accelImageNumpy), axis=0)
                    else:
                        accelImageNumpy_Stack = np.concatenate((accelImageNumpy_Stack, np.expand_dims(accelImageNumpy, axis=0)), axis=0)

            for i in range(256): ## train each slice for the first 6 channels for each subject
                for j in range(6):
                    if j == 0:
                        orginalStack =np.expand_dims(np.copy(orginalImageNumpy_Stack[j][i][32:224]), axis=0)
                        accelStack =np.expand_dims(np.copy(accelImageNumpy_Stack[j][i][32:224]), axis=0)
                    else:
                        orginalStack = np.concatenate((orginalStack, np.expand_dims(orginalImageNumpy_Stack[j][i][32:224], axis=0)), axis=0)
                        accelStack = np.concatenate((accelStack, np.expand_dims(accelImageNumpy_Stack[j][i][32:224], axis=0)), axis=0)
                self.orginalFileList.append(orginalStack)
                self.accelFileList.append(accelStack)
            
            # for i in range(256): ## train each slice for just the first channel for each subject
            #     if i == 0:
            #         orginalStack =np.expand_dims(np.copy(orginalImageNumpy_Stack[0][i][32:224]), axis=0)
            #         accelStack =np.expand_dims(np.copy(accelImageNumpy_Stack[0][i][32:224]), axis=0)
            #         # plt.imshow(abs(orginalImageNumpy_Stack[0][i][32:224].real), cmap='gray')
            #         # plt.show()
            #     else:
            #         orginalStack = np.concatenate((orginalStack, np.expand_dims(orginalImageNumpy_Stack[0][i][32:224], axis=0)), axis=0)
            #         accelStack = np.concatenate((accelStack, np.expand_dims(accelImageNumpy_Stack[0][i][32:224], axis=0)), axis=0)
            #         # plt.imshow(abs(orginalImageNumpy_Stack[0][i][32:224].real), cmap='gray')
            #         # plt.show()
            # self.orginalFileList.append(orginalStack)
            # self.accelFileList.append(accelStack)
            
            ## for 
            # self.orginalFileList.append(np.expand_dims(orginalImageNumpy[128][32:224], axis=0))
            # self.accelFileList.append(np.expand_dims(accelImageNumpy[128][32:224], axis=0))
            print('Image ' + orginalPath + ' loaded')

    def __getitem__(self, index):
        return self.accelFileList[index], self.orginalFileList[index]

    def __len__(self):
        return len(self.accelFileList)

# dataset = h5DatasetNew(train=True)