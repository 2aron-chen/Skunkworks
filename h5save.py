import h5py
import numpy as np
from glob import glob

allImages = sorted(glob("/study/mrphys/skunkworks/training_data//mover01/*/", recursive=True))

for imgIndex in range(len(allImages)):
    name = allImages[imgIndex].split('/')[-2]
    with h5py.File(f'/scratch/mrphys/denoised/comparison_{name}.h5','w') as f:
        grp = f.create_group('Original')
        with h5py.File(allImages[imgIndex]+'processed_data/C.h5','r') as hfOriginal:
            for n in range(6):
                n = 'C_000_0'+ str(n).zfill(2)
                grp.create_dataset(n, data=np.array(hfOriginal['Images'][n]))

        grp = f.create_group('Noisy')
        with h5py.File(allImages[imgIndex]+'processed_data/acc_2min/C.h5','r') as hfNoisy:
            for n in range(6):
                n = 'C_000_0'+ str(n).zfill(2)
                grp.create_dataset(n, data=np.array(hfNoisy['Images'][n]))

        pred = np.load(f'pred/denoised_{0}.npy')
        temp = pred.astype(np.dtype([('real','f'),('imag','f')]))
        temp['imag'] = pred.imag
        pred = temp
        grp = f.create_group('Denoised')
        for n in range(6):
            grp.create_dataset('C_000_0'+ str(n).zfill(2), data=pred[n])