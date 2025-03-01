import torch
from torch import nn
from .encoder import Encoder
from .decoder import Decoder
from .quantizer import Quantizer
import os

class VQGAN(nn.Module):
    def __init__(self, m=3, embedding_dim=512, device=None):
        # Basically uses H x W x 3 image as input
        super(VQGAN, self).__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.encoder = Encoder(m, embedding_dim=embedding_dim).to(self.device)
        self.quantizer = Quantizer(num_embeddings=512, embedding_dim=embedding_dim, device=self.device).to(self.device)
        self.decoder = Decoder(m, embedding_dim=embedding_dim).to(self.device)
        
        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        # CNN to section
        z_hat = self.encoder(x)
        # Assign latent vector
        z_q = self.quantizer(z_hat)
        # Decoder
        x_rep = self.decoder(z_q)
        return x_rep

    def encode(self, x):
        """Encode input to latent representation"""
        z_hat = self.encoder(x)
        z_q = self.quantizer(z_hat)
        return z_q
    
    def decode(self, z_q):
        """Decode latent representation to image"""
        return self.decoder(z_q)

    def save_checkpoint(self, save_path, epoch=0, optimizer=None, loss=None):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'quantizer_state_dict': self.quantizer.state_dict()
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if loss is not None:
            checkpoint['loss'] = loss
            
        torch.save(checkpoint, save_path)
        
    def load_checkpoint(self, checkpoint_path, optimizer=None):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        return checkpoint.get('epoch', 0), checkpoint.get('loss', None)

    def extract_encoder(self):
        """Extract encoder for separate use"""
        encoder = Encoder(self.encoder.m, self.encoder.embedding_dim).to(self.device)
        encoder.load_state_dict(self.encoder.state_dict())
        return encoder

    def extract_decoder(self):
        """Extract decoder for separate use"""
        decoder = Decoder(self.decoder.m, self.decoder.embedding_dim).to(self.device)
        decoder.load_state_dict(self.decoder.state_dict())
        return decoder 