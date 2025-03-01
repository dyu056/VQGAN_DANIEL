import torch
from torch import nn
import os

class Quantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None):
        super(Quantizer, self).__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.codebook = nn.Embedding(num_embeddings=self.num_embeddings, embedding_dim=self.embedding_dim)
        self.to(self.device)
    
    def forward(self, z):
        z = z.to(self.device)
        z_permuted = z.permute(0, 2, 3, 1)
        z_flattened = z_permuted.view(-1, self.embedding_dim)
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.codebook.weight**2, dim=1) - \
            2 * (torch.matmul(z_flattened, self.codebook.weight.t()))
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.codebook(min_encoding_indices).view(z.shape)
        return z_q

    def save_checkpoint(self, save_path):
        """Save quantizer checkpoint"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.state_dict(), save_path)
    
    def load_checkpoint(self, checkpoint_path):
        """Load quantizer checkpoint"""
        self.load_state_dict(torch.load(checkpoint_path, map_location=self.device)) 