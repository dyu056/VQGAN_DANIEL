import torch
import matplotlib.pyplot as plt
from models.vqgan import VQGAN
from utils.image_utils import tensor_to_image
import os

if __name__ == "__main__":
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create checkpoints directory if it doesn't exist
    os.makedirs('checkpoints', exist_ok=True)

    # Generate a random image tensor
    random_image = torch.randn(1, 3, 64, 64, device=device)

    # Create and train VQGAN model
    vqgan = VQGAN(device=device)
    
    # Save the full model
    vqgan.save_checkpoint('checkpoints/vqgan.pt')
    
    # Extract encoder and decoder
    encoder = vqgan.extract_encoder()
    decoder = vqgan.extract_decoder()
    
    # Save individual components
    encoder.save_checkpoint('checkpoints/encoder.pt')
    decoder.save_checkpoint('checkpoints/decoder.pt')
    
    # Use components separately
    with torch.no_grad():
        # Full pipeline
        reconstruction = vqgan(random_image)
        
        # Separate pipeline
        latent = encoder(random_image)
        reconstruction2 = decoder(latent)
    
    # Load from checkpoint
    new_vqgan = VQGAN(device=device)
    epoch, loss = new_vqgan.load_checkpoint('checkpoints/vqgan.pt')
    
    # Move tensors to CPU for plotting
    random_image_cpu = random_image.cpu()
    reconstruction_cpu = reconstruction.cpu()
    reconstruction2_cpu = reconstruction2.cpu()
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(tensor_to_image(random_image_cpu))
    plt.title('Input Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(tensor_to_image(reconstruction_cpu))
    plt.title('Full VQGAN Reconstruction')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(tensor_to_image(reconstruction2_cpu))
    plt.title('Separate Encoder-Decoder')
    plt.axis('off')
    
    plt.show()
    
    # Print shapes for verification
    print(f"Input shape: {random_image.shape}")
    print(f"Output shape: {reconstruction.shape}") 