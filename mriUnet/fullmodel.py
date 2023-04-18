import torch
from torch import nn
from unet import UNet
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

class fullModel(nn.Module):
    
    def __init__(self, nchans=10, norm_scale=1, bkg_lambda=0.1, ssim_lambda=1.0):
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
            1,
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
        self.l2 = nn.MSELoss()
        self.ssim_lambda = ssim_lambda
        
    def get_ssim(self, pred, true):
        ssim_loss_real = (1-ms_ssim(pred.real, true.real, data_range=self.norm_scale, size_average=False)).mean()
        ssim_loss_imag = (1-ms_ssim(pred.imag, true.imag, data_range=self.norm_scale, size_average=False)).mean()
        return ssim_loss_real+ssim_loss_imag
        
    def forward(self, x, gt, y, mask):
        
        mask[mask>0]=1.0
        mask[mask==0]=self.bkg_lambda
            
        # t1 predictor
        denoised_x = self.denoiser(x)
        y_pred_denoised = torch.abs(self.T1Predictor(denoised_x))
        y_pred_gt = torch.abs(self.T1Predictor(gt))
        l2_loss_gt = self.l2(y_pred_gt*mask, y*mask)
        l2_loss_d = self.l2(y_pred_denoised*mask, y*mask)
        
        # denoiser
        denoised_gt = self.denoiser(gt)
        gt = torch.sigmoid(gt)
        ssim_loss_x = self.get_ssim(torch.sigmoid(denoised_x),gt)
        ssim_loss_gt = self.get_ssim(torch.sigmoid(denoised_gt),gt)
        ssim_loss = ssim_loss_x+ssim_loss_gt
        
        # final loss
        loss = ssim_loss*self.ssim_lambda + l2_loss_d + l2_loss_gt
        
        losses = {
            "ssim":ssim_loss,
            "l2_denoiser":l2_loss_d,
            "l2_groundtruth":l2_loss_gt,
            "total":loss,
        }
        preds = [denoised_x, y_pred_denoised, y_pred_gt]
        
        return preds, losses, loss