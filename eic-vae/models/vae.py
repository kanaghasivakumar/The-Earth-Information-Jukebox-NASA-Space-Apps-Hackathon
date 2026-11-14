"""Convolutional VAE for 64x64 RGB plus optical flow."""
import torch
from torch import nn
import torch.nn.functional as F
class ConvVAE(nn.Module):
 def __init__(self,latent_dim=2,image_size=64):
  super().__init__()
  if latent_dim not in (2,3,8,16): raise ValueError('latent_dim must be one of 2, 3, 8, or 16')
  if image_size < 8 or image_size % 8: raise ValueError('image_size must be at least 8 and divisible by 8')
  self.latent_dim=latent_dim; self.image_size=image_size; self.base_size=image_size//8
  self.encoder=nn.Sequential(nn.Conv2d(5,32,4,2,1),nn.ReLU(True),nn.Conv2d(32,64,4,2,1),nn.ReLU(True),nn.Conv2d(64,128,4,2,1),nn.ReLU(True),nn.Flatten())
  features=128*self.base_size*self.base_size
  self.fc_mu=nn.Linear(features,latent_dim); self.fc_logvar=nn.Linear(features,latent_dim)
  self.decoder_fc=nn.Linear(latent_dim,features)
  self.decoder=nn.Sequential(nn.Unflatten(1,(128,self.base_size,self.base_size)),nn.ConvTranspose2d(128,64,4,2,1),nn.ReLU(True),nn.ConvTranspose2d(64,32,4,2,1),nn.ReLU(True),nn.ConvTranspose2d(32,16,4,2,1),nn.ReLU(True))
  self.rgb_head=nn.Sequential(nn.Conv2d(16,3,3,padding=1),nn.Sigmoid()); self.flow_head=nn.Sequential(nn.Conv2d(16,2,3,padding=1),nn.Tanh())
 def encode(self,x):
  h=self.encoder(x); return self.fc_mu(h),self.fc_logvar(h).clamp(-20,10)
 @staticmethod
 def reparameterize(mu,logvar,deterministic=False): return mu if deterministic else mu+torch.randn_like(mu)*torch.exp(.5*logvar)
 def decode(self,z):
  h=self.decoder(self.decoder_fc(z)); return self.rgb_head(h),self.flow_head(h)
 def forward(self,x,deterministic=False):
  if x.shape[-2:] != (self.image_size,self.image_size): raise ValueError(f'Expected {self.image_size}x{self.image_size} input')
  mu,lv=self.encode(x); z=self.reparameterize(mu,lv,deterministic); rgb,flow=self.decode(z); return {'rgb':rgb,'flow':flow,'mu':mu,'logvar':lv,'z':z}
def vae_loss(output,image,flow,beta=.01,flow_weight=1.):
 rgb_mse=F.mse_loss(output['rgb'],image); flow_mse=F.mse_loss(output['flow'],flow)
 kl=-.5*(1+output['logvar']-output['mu'].pow(2)-output['logvar'].exp()).sum(dim=1).mean()
 total=rgb_mse+flow_weight*flow_mse+beta*kl
 return total,{'loss':total.detach(),'rgb_mse':rgb_mse.detach(),'flow_mse':flow_mse.detach(),'kl':kl.detach()}
