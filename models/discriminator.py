import torch
from torch import nn
import os

class Discriminator(nn.Module):
    def __init__(self, input_channels=3, feature_maps=64, device=None):
        super(Discriminator, self).__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # First layer: No BatchNorm, LeakyReLU with slope 0.2
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, feature_maps, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Second layer: BatchNorm, LeakyReLU
        self.layer2 = nn.Sequential(
            nn.Conv2d(feature_maps, feature_maps * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Third layer: BatchNorm, LeakyReLU
        self.layer3 = nn.Sequential(
            nn.Conv2d(feature_maps * 2, feature_maps * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Fourth layer: BatchNorm, LeakyReLU
        self.layer4 = nn.Sequential(
            nn.Conv2d(feature_maps * 4, feature_maps * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Final layer: Convolution to 1D output, followed by Sigmoid
        self.final_layer = nn.Sequential(
            nn.Conv2d(feature_maps * 8, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.final_layer(x)
        return x

    def save_checkpoint(self, save_path):
        """Save discriminator checkpoint"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.state_dict(), save_path)
    
    def load_checkpoint(self, checkpoint_path):
        """Load discriminator checkpoint"""
        self.load_state_dict(torch.load(checkpoint_path, map_location=self.device)) 