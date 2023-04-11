from torch.utils.data import Dataset
import h5py
import numpy as np
from glob import glob

class h5DatasetNew(Dataset):
    def __init__(self, train):
        self.orginalPathList = []
        self.accelPathList = []
        self.orginalFileList = []
        self.accelFileList = []
        # self.mid = int(256/2) - 3  ## minus three because we are taking the middle 8 slices

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
                for i in range(16):
                    n = prefix + str(i).zfill(2)
                    image = hf['Images'][n]
                
                    imageNumpy = image['real']
                    imageNumpy = imageNumpy * (255/(imageNumpy.max()))
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
                    imageNumpy = imageNumpy * (255/(imageNumpy.max()))
                    accelImageNumpy = np.array(imageNumpy + 0j*image['imag'])
                    if i == 0:
                        accelImageNumpy_Stack = np.expand_dims(np.copy(accelImageNumpy), axis=0)
                    else:
                        accelImageNumpy_Stack = np.concatenate((accelImageNumpy_Stack, np.expand_dims(accelImageNumpy, axis=0)), axis=0)

            for i in range(256): ## train each slice for each subject
                for j in range(16):
                    if j == 0:
                        orginalStack =np.expand_dims(np.copy(orginalImageNumpy_Stack[j][i][32:224]), axis=0)
                        accelStack =np.expand_dims(np.copy(accelImageNumpy_Stack[j][i][32:224]), axis=0)
                    else:
                        orginalStack = np.concatenate((orginalStack, np.expand_dims(orginalImageNumpy_Stack[j][i][32:224], axis=0)), axis=0)
                        accelStack = np.concatenate((accelStack, np.expand_dims(accelImageNumpy_Stack[j][i][32:224], axis=0)), axis=0)
                self.orginalFileList.append(orginalStack)
                self.accelFileList.append(accelStack)
            
            ## for 
            # self.orginalFileList.append(np.expand_dims(orginalImageNumpy[128][32:224], axis=0))
            # self.accelFileList.append(np.expand_dims(accelImageNumpy[128][32:224], axis=0))
            print('Image ' + orginalPath + ' loaded')

    def __getitem__(self, index):
        return self.accelFileList[index], self.orginalFileList[index]

    def __len__(self):
        return len(self.accelFileList)

#dataset = h5DatasetNew(train=True)