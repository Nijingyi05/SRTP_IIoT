'''
Author: Ni Jingyi 3230102776@zju.edu.cn
Date: 2025-12-04 16:38:56
LastEditors: Ni Jingyi 3230102776@zju.edu.cn
LastEditTime: 2025-12-04 16:39:00
FilePath: \SRTP_IIoT\try.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(64, latent_dim)    # 均值μ
        self.fc_logvar = nn.Linear(64, latent_dim) # 对数方差log(σ²)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()  # 假设输入在[0,1]范围内
        )
    
    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)  # σ = exp(0.5 * log(σ²))
        epsilon = torch.randn_like(std)  # ε ~ N(0,1)
        return mu + epsilon * std  # z = μ + ε·σ
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        # 编码
        mu, logvar = self.encode(x)
        
        # 重参数化采样
        z = self.reparameterize(mu, logvar)
        
        # 解码
        x_recon = self.decode(z)
        
        return x_recon, mu, logvar

# 损失函数
def vae_loss(x_recon, x, mu, logvar):
    # 重构损失（二值交叉熵）
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
    
    # KL散度（正则项）
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + kl_loss