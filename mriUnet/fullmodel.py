import torch
from torch import nn
from unet import UNet
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

class fullModel(nn.Module):
    
    def __init__(self, nchans=10, norm_scale=1, bkg_lambda=1e-2, ssim_lambda=0.1):
        super(fullModel, self).__init__()
        self.denoiser = UNet(
            nchans,
            nchans,
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
        self.T1Predictor = UNet(
            nchans,
            2, # 1 - T1 mapping , 2 - mask
            f_maps=32,
            layer_order=['separable convolution', 'batch norm', 'relu'],
            depth=4,
            layer_growth=2.0,
            residual=True,
            complex_input=True,
            complex_kernel=True,
            ndims=2,
            padding=1
        )
        self.norm_scale = norm_scale
        self.bkg_lambda = bkg_lambda
        self.bce = nn.BCEWithLogitsLoss()
        self.ssim_lambda = ssim_lambda
        
    def get_ssim(self, pred, true):
        ssim_loss_real = (1-ms_ssim(pred.real, true.real, data_range=self.norm_scale, size_average=False)).mean()
        ssim_loss_imag = (1-ms_ssim(pred.imag, true.imag, data_range=self.norm_scale, size_average=False)).mean()
        return ssim_loss_real+ssim_loss_imag
    
    def get_error(self, pred, true):
        mseDiff = torch.mean((pred-true)**2)
        return mseDiff
        
    def forward(self, x, gt, y, mask):
        
        mask = mask.clone()
        
        mask[mask>0]=1.0
        mask[mask==0]=self.bkg_lambda
            
        # t1 predictor
        denoised_x = self.denoiser(x)
        
        denoised_pred = self.T1Predictor(denoised_x)
        denoised_t1 = torch.abs(denoised_pred[:,0:1])
        denoised_mask_logit = denoised_pred[:,1:2].real + denoised_pred[:,1:2].imag
        
        gt_pred = self.T1Predictor(gt)
        gt_t1 = torch.abs(gt_pred[:,0:1])
        gt_mask_logit = gt_pred[:,1:2].real + gt_pred[:,1:2].imag
        
        # mask BCE loss
        mask_loss = (self.bce(denoised_mask_logit, mask) + self.bce(gt_mask_logit, mask))/2
        
        # t1 l2 loss     
        l2_loss_gt = self.get_error(gt_t1*(torch.sigmoid(denoised_mask_logit)).to(mask.dtype), y*mask)
        l2_loss_d = self.get_error(denoised_t1*(torch.sigmoid(gt_mask_logit)).to(mask.dtype), y*mask)
        
        # denoiser
        denoised_gt = self.denoiser(gt)
        gt = torch.sigmoid(gt)
        ssim_loss_x = self.get_ssim(torch.sigmoid(denoised_x),gt)
        ssim_loss_gt = self.get_ssim(torch.sigmoid(denoised_gt),gt)
        ssim_loss = ssim_loss_x+ssim_loss_gt
        
        # final loss
        loss = ssim_loss*self.ssim_lambda + l2_loss_d + l2_loss_gt + mask_loss
        
        losses = {
            "ssim":ssim_loss,
            "error_denoiser":l2_loss_d,
            "error_groundtruth":l2_loss_gt,
            "mask":mask_loss,
            "total":loss,
        }
        preds = [denoised_x, denoised_t1, gt_t1, denoised_mask_logit, gt_mask_logit]
        
        return preds, losses, loss