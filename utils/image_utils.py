import torch
import numpy as np

def tensor_to_image(tensor):
    """Convert a tensor to a displayable image.
    
    Args:
        tensor (torch.Tensor): Input tensor of shape [C, H, W] or [1, C, H, W]
    
    Returns:
        numpy.ndarray: Normalized image array
    """
    img = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0,1] for display
    return img 