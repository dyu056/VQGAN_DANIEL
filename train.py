import torch
print(torch.cuda.is_available())
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.vqgan import VQGAN
from models.discriminator import Discriminator
from utils.config import Config
import os
from tqdm import tqdm
import wandb  # for logging
import argparse

def calculate_lambda(recon_loss, gan_loss, vqgan, delta=1e-6):
    recon_grad = torch.autograd.grad(recon_loss, vqgan.parameters(), retain_graph=True, create_graph=True, allow_unused=True)
    gan_grad = torch.autograd.grad(gan_loss, vqgan.parameters(), retain_graph=True, create_graph=True, allow_unused=True)
    
    # Handle None gradients
    recon_grad_norm = sum(g.norm() for g in recon_grad if g is not None)
    gan_grad_norm = sum(g.norm() for g in gan_grad if g is not None)
    
    lambda_weight = recon_grad_norm / (gan_grad_norm + delta)
    return lambda_weight

def train(config):
    # Set up device and print info
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # Create directories
    os.makedirs(config.training['checkpoint_dir'], exist_ok=True)
    os.makedirs(config.training['output_dir'], exist_ok=True)

    # Set up data transforms
    transform = transforms.Compose([
        transforms.Resize((config.training['image_size'], config.training['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load dataset
    dataset = datasets.ImageFolder(config.training['data_dir'], transform=transform)
    dataloader = DataLoader(dataset, batch_size=config.training['batch_size'], shuffle=True, num_workers=4, pin_memory=True)

    # Initialize models
    vqgan = VQGAN(m=config.model['m'], embedding_dim=config.model['embedding_dim'], device=device).to(device)
    discriminator = Discriminator(device=device).to(device)

    # Initialize optimizers
    vqgan_optimizer = optim.Adam(vqgan.parameters(), lr=config.training['learning_rate'])
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=config.training['learning_rate'])

    # Loss functions
    mse_loss = nn.MSELoss().to(device)
    bce_loss = nn.BCELoss().to(device)

    # Initialize wandb
    if config.training['use_wandb']:
        wandb.init(project="vqgan-training", config=config.config)

    # Clear GPU cache if using CUDA
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    accumulation_steps = 4  # Accumulate gradients over 4 batches
    
    try:
        # Training loop
        for epoch in range(config.training['epochs']):
            vqgan.train()
            discriminator.train()
            
            progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{config.training["epochs"]}')
            for batch_idx, (images, _) in enumerate(progress_bar):
                images = images.to(device, non_blocking=True)
                batch_size = images.size(0)

                # Forward pass through VQGAN
                vqgan_optimizer.zero_grad()
                
                # Get encoder output and quantized vectors
                z_e = vqgan.encoder(images)  # E(x)
                z_q = vqgan.quantizer(z_e)   # quantized vectors
                x_recon = vqgan.decoder(z_q)  # reconstructed image
                
                # Reconstruction loss: ||x - x̂||²
                recon_loss = mse_loss(x_recon, images)
                
                # Commitment loss: ||sg[E(x)] - zq||² + ||sg[zq] - E(x)||²
                commitment_loss = mse_loss(z_e.detach(), z_q)
                codebook_loss = mse_loss(z_e, z_q.detach())
                
                # Total VQ loss
                vq_loss = recon_loss + commitment_loss + codebook_loss
                
                # GAN loss
                real_labels = torch.ones(batch_size, 1, 1, 1).to(device)
                fake_labels = torch.zeros(batch_size, 1, 1, 1).to(device)   

                # Discriminator forward pass
                disc_optimizer.zero_grad()
                real_output = discriminator(images)
                fake_output = discriminator(x_recon.detach())
                d_loss_real = bce_loss(real_output, real_labels)
                d_loss_fake = bce_loss(fake_output, fake_labels)
                GAN_loss = d_loss_real + d_loss_fake

                # Calculate lambda
                lambda_weight = calculate_lambda(recon_loss, GAN_loss, vqgan, delta=1e-6)
                
                # Total loss
                total_loss = recon_loss + lambda_weight * GAN_loss
                
                # Backward pass and optimization
                vqgan_optimizer.zero_grad()
                total_loss.backward(retain_graph=True)  # Retain graph for GAN_loss backward
                vqgan_optimizer.step()
                
                # Update discriminator
                disc_optimizer.zero_grad()
                GAN_loss.backward()
                disc_optimizer.step()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'VQ_loss': f'{vq_loss.item():.4f}',
                    'Recon_loss': f'{recon_loss.item():.4f}',
                    'Commit_loss': f'{commitment_loss.item():.4f}'
                })

                # Log metrics
                if config.training['use_wandb']:
                    wandb.log({
                        'VQ_loss': vq_loss.item(),
                        'Recon_loss': recon_loss.item(),
                        'Commit_loss': commitment_loss.item()
                    })

                # Only update weights after accumulating enough gradients
                if (batch_idx + 1) % accumulation_steps == 0:
                    vqgan_optimizer.step()
                    vqgan_optimizer.zero_grad()
                    disc_optimizer.step()
                    disc_optimizer.zero_grad()

            # Save checkpoints
            if (epoch + 1) % config.training['save_freq'] == 0:
                checkpoint_path = os.path.join(
                    config.training['checkpoint_dir'], 
                    f'vqgan_epoch_{epoch+1}.pt'
                )
                vqgan.save_checkpoint(checkpoint_path, epoch=epoch, optimizer=vqgan_optimizer)
                discriminator.save_checkpoint(
                    os.path.join(config.training['checkpoint_dir'], f'discriminator_epoch_{epoch+1}.pt')
                )

            # Optional: clear cache periodically
            if torch.cuda.is_available() and (epoch + 1) % 10 == 0:
                torch.cuda.empty_cache()
                
    except RuntimeError as e:
        if "out of memory" in str(e):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("WARNING: GPU out of memory, try reducing batch size")
        raise e
    
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train VQGAN')
    parser.add_argument('--config', type=str, help='Path to config file')
    args = parser.parse_args()
    
    # Load config
    config = Config(args.config)
    
    # Save the actual config used for this run
    os.makedirs(config.training['output_dir'], exist_ok=True)
    config.save(os.path.join(config.training['output_dir'], 'used_config.json'))
    
    # Set CUDA options
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  # May improve performance
        torch.backends.cudnn.deterministic = False  # May improve performance but reduces reproducibility
    
    train(config) 