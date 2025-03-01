import torch
from torch import nn
from .blocks import Residual, Non_Local, Up, GroupNorm, Swish
import os

class Decoder(nn.Module):
    def __init__(self, m=3, embedding_dim=512, device=None):
        super(Decoder, self).__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.m = m
        self.embedding_dim = embedding_dim
        channel_sizes = [embedding_dim, 256, 128, 3]  # Reverse of encoder
        layers = []  # Temporary list to build the Sequential
        
        layers.append(nn.Conv2d(channel_sizes[0], channel_sizes[1], kernel_size=3, padding=1))
        
        layers.append(Residual(in_channels=channel_sizes[1], out_channels=channel_sizes[1]))
        layers.append(Non_Local(in_channels=channel_sizes[1]))
        layers.append(Residual(in_channels=channel_sizes[1], out_channels=channel_sizes[1]))
        
        for i in range(m):
            layers.append(Residual(in_channels=channel_sizes[1], out_channels=channel_sizes[1]))
            if i == m-1:
                layers.append(Up(in_channels=channel_sizes[1], out_channels=channel_sizes[2]))
            else:
                layers.append(Up(in_channels=channel_sizes[1], out_channels=channel_sizes[1]))
        
        layers.append(Residual(in_channels=channel_sizes[2], out_channels=channel_sizes[2]))
        layers.append(GroupNorm(num_groups=32, num_channels=channel_sizes[2]))
        layers.append(Swish())
        layers.append(nn.Conv2d(channel_sizes[2], channel_sizes[3], kernel_size=3, padding=1))
        
        self.layers = nn.Sequential(*layers)
        self.to(self.device)

    def forward(self, z_q):
        # print(f"Decoder input shape: {z_q.shape}")
        output = self.layers(z_q)
        # print(f"Decoder output shape: {output.shape}")
        return output 

    def save_checkpoint(self, save_path):
        """Save decoder checkpoint"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.state_dict(), save_path)
    
    def load_checkpoint(self, checkpoint_path):
        """Load decoder checkpoint"""
        self.load_state_dict(torch.load(checkpoint_path, map_location=self.device)) 