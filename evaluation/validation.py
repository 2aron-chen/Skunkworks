from glob import glob
# import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from tqdm import tqdm

def cal_mean(array):
    # get rid of zeros in the array
    nonzero = np.nonzero(array)
    nonzero_row = nonzero[0]
    nonzero_col = nonzero[1]
    value_list = []
    for row, col in zip(nonzero_row, nonzero_col):
        value_list.append(array[row, col])
    
    if len(value_list) == 0:
        return 0
    else: 
        return np.mean(value_list)

wb_denoised = []
gm_denoised = []
wm_denoised = []

wb_gt = []
gm_gt = []
wm_gt = []

denoised_list_path = sorted(glob("/study/mrphys/skunkworks/kk/T1/*"))
gt_list_path = sorted(glob("/study/mrphys/skunkworks/training_data/mover01/*/processed_data/T1_3_tv.nii"))
mask_list_path = sorted(glob('/scratch/mrphys/fullDataset/*'))

gt_list = []
denoised_list = []
mask_list = []

for subject in range(65):
    print('subject: ', subject)
    gt = np.transpose(nib.load(gt_list_path[subject]).get_fdata())
    denoised = np.load(denoised_list_path[subject], allow_pickle=True)[0]
    gt_list.append(gt)
    denoised_list.append(denoised)
    temp = []

    for slice in tqdm(range(256)):
        _, _, _, mask = np.load(mask_list_path[subject] + f'/{slice}.npy', allow_pickle=True)
        temp.append(mask)
    
    mask_list.append(temp)
        

for subject in range(65):
    print('subject: ', subject)

    mean_whole_brain_mask_denoised = []
    mean_gray_matter_mask_denoised = []
    mean_white_matter_mask_denoised = []

    mean_whole_brain_mask_gt = []
    mean_gray_matter_mask_gt = []
    mean_white_matter_mask_gt = []

    for slice in tqdm(range(256)):
        whole_brain_mask = mask_list[subject][slice][0] != 0 * 1
        gray_matter_mask = mask_list[subject][slice][0] == 2 * 1
        white_matter_mask = mask_list[subject][slice][0] == 3 * 1
        whole_brain_mask_denoised = whole_brain_mask*denoised_list[subject][slice]
        gray_matter_mask_denoised = gray_matter_mask*denoised_list[subject][slice]
        white_matter_mask_denoised = white_matter_mask*denoised_list[subject][slice]

        whole_brain_mask_gt = whole_brain_mask*gt_list[subject][slice]
        gray_matter_mask_gt = gray_matter_mask*gt_list[subject][slice]
        white_matter_mask_gt = white_matter_mask*gt_list[subject][slice]

        mean_whole_brain_mask_denoised.append(cal_mean(whole_brain_mask_denoised))
        mean_gray_matter_mask_denoised.append(cal_mean(gray_matter_mask_denoised))
        mean_white_matter_mask_denoised.append(cal_mean(white_matter_mask_denoised))

        mean_whole_brain_mask_gt.append(cal_mean(whole_brain_mask_gt))
        mean_gray_matter_mask_gt.append(cal_mean(gray_matter_mask_gt))
        mean_white_matter_mask_gt.append(cal_mean(white_matter_mask_gt))

    # get rid of zeros in the list
    mean_whole_brain_mask_denoised = [i for i in mean_whole_brain_mask_denoised if i != 0]
    mean_gray_matter_mask_denoised = [i for i in mean_gray_matter_mask_denoised if i != 0]
    mean_white_matter_mask_denoised = [i for i in mean_white_matter_mask_denoised if i != 0]
    
    mean_whole_brain_mask_gt = [i for i in mean_whole_brain_mask_gt if i != 0]
    mean_gray_matter_mask_gt = [i for i in mean_gray_matter_mask_gt if i != 0]
    mean_white_matter_mask_gt = [i for i in mean_white_matter_mask_gt if i != 0]

    wb_denoised.append(np.mean(mean_whole_brain_mask_denoised))
    gm_denoised.append(np.mean(mean_gray_matter_mask_denoised))
    wm_denoised.append(np.mean(mean_white_matter_mask_denoised))

    wb_gt.append(np.mean(mean_whole_brain_mask_gt))
    gm_gt.append(np.mean(mean_gray_matter_mask_gt))
    wm_gt.append(np.mean(mean_white_matter_mask_gt))

print('Denoised whole brain: ', wb_denoised)
print('Denoised gray matter: ', gm_denoised)
print('Denoised white matter: ', wm_denoised)

print('GT whole brain: ', wb_gt)
print('GT gray matter: ', gm_gt)
print('GT white matter: ', wm_gt)