import torch
from torch import nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

# Single image first!
class VQGAN(nn.Module):
    def __init__(self, m=3, embedding_dim=512):
        # Basically uses H x W x 3 image as input
        super(VQGAN, self).__init__()
        self.encoder = Encoder(m, embedding_dim=embedding_dim) # Output is h x w x nz
        self.quantizer = Quantizer(num_embeddings=512, embedding_dim=embedding_dim) #Output is h x w x nz
        self.decoder = Decoder(m, embedding_dim=embedding_dim) # Output is H X W X 3

    def forward(self, x):
        # CNN to section
        z_hat = self.encoder(x)
        # Assign latent vector
        z_q = self.quantizer(z_hat)
        # Decoder
        x_rep = self.decoder(z_q)
        return x_rep

class Discriminator(nn.Module):
    def __init__(self, input_channels=3, feature_maps=64):
        super(Discriminator, self).__init__()
        
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

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.final_layer(x)
        return x

class Encoder(nn.Module):
    def __init__(self, m=3, embedding_dim=512):
        super(Encoder, self).__init__()
        channel_sizes = [3, 128, 256, embedding_dim] # It is static for the input channel to be 3
        layers = []  # Temporary list to build the Sequential
        
        layers.append(nn.Conv2d(channel_sizes[0], channel_sizes[1], kernel_size=3, padding=1))

        for i in range(m):
            if i == 0:
                layers.append(Residual(in_channels=channel_sizes[1], out_channels=channel_sizes[1]))
                layers.append(Down(in_channels=channel_sizes[1], out_channels=channel_sizes[2]))
            else:
                layers.append(Residual(in_channels=channel_sizes[2], out_channels=channel_sizes[2]))
                layers.append(Down(in_channels=channel_sizes[2], out_channels=channel_sizes[2]))

        layers.append(Residual(in_channels=channel_sizes[2], out_channels=channel_sizes[2]))
        layers.append(Non_Local(in_channels=channel_sizes[2]))
        layers.append(Residual(in_channels=channel_sizes[2], out_channels=channel_sizes[2]))
        layers.append(GroupNorm(num_groups=32, num_channels=channel_sizes[2]))
        layers.append(Swish())
        layers.append(nn.Conv2d(channel_sizes[2], channel_sizes[3], kernel_size=3, padding=1))
        self.layers = nn.Sequential(*layers)  # Convert list to Sequential

    def forward(self, x):
        print(f"Encoder input shape: {x.shape}")
        output = self.layers(x)
        print(f"Encoder output shape: {output.shape}")
        return output

class Decoder(nn.Module):
    def __init__(self, m=3, embedding_dim=512):
        super(Decoder, self).__init__()
        channel_sizes = [embedding_dim, 256, 128, 3]  # Reverse of encoder
        layers = []  # Temporary list to build the Sequential
        
        # Initial convolution from embedding dimension
        layers.append(nn.Conv2d(channel_sizes[0], channel_sizes[1], kernel_size=3, padding=1))
        
        # First residual blocks and non-local attention
        layers.append(Residual(in_channels=channel_sizes[1], out_channels=channel_sizes[1]))
        layers.append(Non_Local(in_channels=channel_sizes[1]))
        layers.append(Residual(in_channels=channel_sizes[1], out_channels=channel_sizes[1]))
        
        # Middle section with up-sampling
        for i in range(m):
            layers.append(Residual(in_channels=channel_sizes[1], out_channels=channel_sizes[1]))
            if i == m-1:
                layers.append(Up(in_channels=channel_sizes[1], out_channels=channel_sizes[2]))
            else:
                layers.append(Up(in_channels=channel_sizes[1], out_channels=channel_sizes[1]))
        
        # Final layers
        layers.append(Residual(in_channels=channel_sizes[2], out_channels=channel_sizes[2]))
        layers.append(GroupNorm(num_groups=32, num_channels=channel_sizes[2]))
        layers.append(Swish())
        layers.append(nn.Conv2d(channel_sizes[2], channel_sizes[3], kernel_size=3, padding=1))
        
        self.layers = nn.Sequential(*layers)

    def forward(self, z_q):
        print(f"Decoder input shape: {z_q.shape}")
        output = self.layers(z_q)
        print(f"Decoder output shape: {output.shape}")
        return output

class Quantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(Quantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.codebook = nn.Embedding(num_embeddings=self.num_embeddings, embedding_dim=self.embedding_dim)
    
    def forward(self, z):
        z_permuted = z.permute(0, 2, 3, 1)
        z_flattened = z_permuted.view(-1, self.embedding_dim)
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.codebook.weight**2, dim=1) - \
            2 * (torch.matmul(z_flattened, self.codebook.weight.t()))
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.codebook(min_encoding_indices).view(z.shape)
        return z_q


## Helper functions
class Down(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super(Down, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1):
        super(Up, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, 
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Residual(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class Non_Local(nn.Module):
    def __init__(self, in_channels):
        super(Non_Local, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = in_channels // 2

        self.g = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.theta = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.phi = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.W = nn.Conv2d(self.inter_channels, in_channels, kernel_size=1)

    def forward(self, x):
        batch_size, C, H, W = x.size()
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        theta_x = phi_x.permute(0, 2, 1)

        f = torch.matmul(theta_x, phi_x)
        f_div_C = f / f.size(-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, H, W)
        W_y = self.W(y)
        z = W_y + x

        return z

class GroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels):
        super(GroupNorm, self).__init__()
        self.group_norm = nn.GroupNorm(num_groups, num_channels)

    def forward(self, x):
        return self.group_norm(x)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


if __name__ == "__main__":
    # Generate a random image tensor
    random_image = torch.randn(1, 3, 64, 64)  # Using randn instead of rand for normalized values

    # Function to convert tensor to a displayable image
    def tensor_to_image(tensor):
        img = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0,1] for display
        return img

    # Create VQGAN model
    vqgan = VQGAN()
    
    # Generate reconstruction
    with torch.no_grad():
        reconstruction = vqgan(random_image)
    
    # Plot original and reconstruction
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(tensor_to_image(random_image))
    plt.title('Random Input Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(tensor_to_image(reconstruction))
    plt.title('Reconstructed Image')
    plt.axis('off')
    
    plt.show()
    
    # Print shapes for verification
    print(f"Input shape: {random_image.shape}")
    print(f"Output shape: {reconstruction.shape}")   