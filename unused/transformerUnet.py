import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from torch import einsum


class ConvBlock(nn.Module):

  def __init__(self,in_ch,out_ch,down):
    super(ConvBlock,self).__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False, padding_mode="reflect") if down #downsample
        else nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False), #upsample
        nn.InstanceNorm2d(out_ch),
        nn.LeakyReLU(0.2) if down #downsample
        else nn.ReLU(), #upsample
    )

  def forward(self,x):
    return self.conv(x)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            nn.GroupNorm(1, dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class TransformerUNet(nn.Module):

  def __init__(self,in_ch=1,out_ch=1,features_dim=16,depth=5,use_sigmoid=True):
    super(TransformerUNet,self).__init__()

    self.depth = depth
    self.factors = [1,2,4] + [8]*depth
    self.UpConvLayers = nn.ModuleList()
    self.DownConvLayers = nn.ModuleList()
    self.use_sigmoid = use_sigmoid

    self.init_down = nn.Sequential(
            nn.Conv2d(in_ch, features_dim, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
    self.last_up = nn.Sequential(
            nn.ConvTranspose2d(features_dim, out_ch, kernel_size=4, stride=2, padding=1),
        )

    for i in range(self.depth):
      self.DownConvLayers.append(
          nn.ModuleList([
            ConvBlock(features_dim*self.factors[i],features_dim*self.factors[i+1],True),
            Residual(PreNorm(features_dim*self.factors[i+1], LinearAttention(features_dim*self.factors[i+1]))),
            ])
          )
      
      self.UpConvLayers.append(
          nn.ModuleList([
            ConvBlock(features_dim*self.factors[self.depth-i+1]*2,features_dim*self.factors[self.depth-i],False),
            Residual(PreNorm(features_dim*self.factors[self.depth-i], LinearAttention(features_dim*self.factors[self.depth-i]))),
          ])
          )
      
    self.UpConvLayers.append(
        nn.ModuleList([
            ConvBlock(features_dim*self.factors[1]*2,features_dim*self.factors[0],False),
            Residual(PreNorm(features_dim*self.factors[0], LinearAttention(features_dim*self.factors[0]))),
        ])
    )
    
    self.bottleneck = nn.Sequential(
            nn.Conv2d(features_dim*self.factors[self.depth], 
                      features_dim*self.factors[self.depth]*2, 
                      4, 2, 1), 
            nn.ReLU(),
        )
      
  def forward(self,x):
    downsampled = []
    downsampled.append(self.init_down(x))

    for i in range(self.depth): 
      x = self.DownConvLayers[i][0](downsampled[-1]) #conv downsampling
      x = self.DownConvLayers[i][1](x) #attention
      downsampled.append(x)

    downsampled.append(self.bottleneck(downsampled[-1]))
    up = self.UpConvLayers[0][0](downsampled[-1])
    up = self.UpConvLayers[0][1](up)

    for i in range(1,self.depth+1):
      up = torch.cat([up,downsampled[self.depth-i+1]],dim=1)
      up = self.UpConvLayers[i][0](up) #conv downsampling
      up = self.UpConvLayers[i][1](up) #attention
    
    if self.use_sigmoid:
        return torch.sigmoid(self.last_up(up))
    else:
        return self.last_up(up)