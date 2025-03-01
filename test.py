import torch
from models.vqgan import VQGAN
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import os
from tqdm import tqdm
from utils.image_utils import tensor_to_image
from utils.config import Config

def test_single_image(model, image_path, device, output_path=None, image_size=256):
    """Test VQGAN on a single image"""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Generate reconstruction
    with torch.no_grad():
        reconstruction = model(image_tensor)
    
    # Visualize results
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(tensor_to_image(reconstruction.cpu()))
    plt.title('Reconstruction')
    plt.axis('off')
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def test_directory(model, input_dir, output_dir, device, image_size=256):
    """Test VQGAN on a directory of images"""
    os.makedirs(output_dir, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    for image_file in tqdm(image_files, desc="Processing images"):
        input_path = os.path.join(input_dir, image_file)
        output_path = os.path.join(output_dir, f'reconstructed_{image_file}')
        
        test_single_image(model, input_path, device, output_path, image_size)

def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = VQGAN(
        m=config.model['m'],
        embedding_dim=config.model['embedding_dim'],
        device=device
    )
    model.load_checkpoint(config.testing['checkpoint_path'])
    model.eval()
    
    if os.path.isfile(args.input_path):
        test_single_image(
            model, 
            args.input_path, 
            device, 
            args.output_path,
            config.testing['image_size']
        )
    else:
        test_directory(
            model, 
            args.input_path, 
            args.output_path, 
            device,
            config.testing['image_size']
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test VQGAN')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--input_path', type=str, required=True, help='Path to input image or directory')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save output')
    
    args = parser.parse_args()
    
    # Load config
    config = Config(args.config)
    
    main(config) 