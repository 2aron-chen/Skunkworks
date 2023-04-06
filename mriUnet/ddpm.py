import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import Adam

class DDPM:
    
    def __init__(diffuser, T=1000, L1loss=True, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        # Define beta schedule
        self.T = T
        self.betas = torch.linspace(0.0001, 0.02, T)
        # Pre-calculate different terms for closed form
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.device = device
        
        self.diffuser = diffuser #model
        
        if L1loss:
            self.loss_fn = nn.L1Loss()
        else:
            self.loss_fn = nn.MSELoss()

    def get_index_from_list(self, vals, t, x_shape):
        """ 
        Returns a specific index t of a passed list of values vals
        while considering the batch dimension.
        """
        batch_size = t.shape[0]
        out = vals.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def forward_diffusion_sample(self, x_0, t, device=self.device):
        """ 
        Takes an image and a timestep as input and 
        returns the noisy version of it
        """
        noise = torch.randn_like(x_0) + torch.randn_like(x_0)*1J
        self.sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        self.sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )
        # mean + variance
        return self.sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
        + self.sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)
    
    def get_loss(self, x_0, t, device=self.device):
        x_noisy, noise = self.forward_diffusion_sample(x_0, t, device)
        noise_pred = self.diffuser(x_noisy, t)
        return self.loss_fn(noise.real, noise_pred.real) + self.loss_fn(noise.imag, noise_pred.imag)
    
    def train(self, trainloader, num_epochs=100):
        optimizer = Adam(self.diffuser.parameters(), lr=1e-3)
        self.diffuser.train()
        for epoch in range(num_epochs):
            pbar = tqdm(enumerate(trainloader))
            avg_loss = 0
            count = 0
            for step, batch in pbar:
                optimizer.zero_grad()
                t = torch.randint(0, self.T, (batch[0].shape[0],), device=self.device).long()
                loss = get_loss(model, batch[0], t)
                loss.backward()
                optimizer.step()
                avg_loss += loss.item()
                count += batch[0].shape[0]
                pbar.set_description(f"Epoch {epoch} | step {step:04d}/{len(data_loader)} Loss: {round(avg_loss/count,5)} ")
    
    @torch.no_grad()
    def sample_timestep(self, x, t):
        """
        Calls the model to predict the noise in the image and returns 
        the denoised image. 
        Applies noise to this image, if we are not in the last step yet.
        """
        betas_t = self.get_index_from_list(betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = self.get_index_from_list(self.sqrt_recip_alphas, t, x.shape)

        # Call model (current image - noise prediction)
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self.diffuser(x, t) / sqrt_one_minus_alphas_cumprod_t
        )
        posterior_variance_t = self.get_index_from_list(self.posterior_variance, t, x.shape)
        if t == 0:
            return model_mean
        else:
            noise = torch.randn_like(x) + torch.randn_like(x)*1J
            return model_mean + torch.sqrt(posterior_variance_t) * noise 