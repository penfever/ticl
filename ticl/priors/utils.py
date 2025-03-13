import random

import torch
from torch import nn

from ticl.distributions import zipf_sampler_f

def order_by_y(x, y):
    """
    Sort input features (x) according to target values (y).
    Compatible with all backends (CUDA, MPS, CPU, ROCm).
    
    Args:
        x: Input features tensor
        y: Target values tensor
        
    Returns:
        Tuple of (sorted_x, sorted_y)
    """
    # Randomly decide whether to sort in ascending or descending order
    # Using Python's random is safe as it only affects the sort direction
    sort_ascending = random.randint(0, 1) == 1
    sort_values = y if sort_ascending else -y
    
    # Sort indices - argsort should work on all backends
    order = torch.argsort(sort_values, dim=0)[:, 0, 0]
    
    # Reshape the order tensor - these operations are backend-agnostic
    order = order.reshape(2, -1).transpose(0, 1).reshape(-1)
    
    # Apply the sorting order to both tensors
    sorted_x = x[order]
    sorted_y = y[order]

    return sorted_x, sorted_y


def randomize_classes(x, num_classes):
    """
    Randomize class assignments in a tensor, supporting all backends (CUDA, MPS, CPU, ROCm).
    
    Args:
        x: Tensor with class indices
        num_classes: Number of classes to consider
        
    Returns:
        x: Tensor with randomized class indices
    """
    device = x.device
    dtype = x.dtype
    
    # Create the class indices on the appropriate device
    classes = torch.arange(0, num_classes, device=device)
    
    # Generate random permutation in a backend-compatible way
    # torch.randperm may not be implemented on all backends (e.g., older MPS versions)
    if device.type == 'mps' and not hasattr(torch, '_C') or not hasattr(torch._C, '_mps_randperm'):
        # Fallback for MPS devices where randperm might not be supported directly
        random_classes = torch.randperm(num_classes, device='cpu').to(device=device, dtype=dtype)
    else:
        try:
            # Try to use native randperm on the device
            random_classes = torch.randperm(num_classes, device=device).to(dtype=dtype)
        except RuntimeError:
            # Fallback if randperm on the specific device fails
            random_classes = torch.randperm(num_classes, device='cpu').to(device=device, dtype=dtype)
    
    # Perform the remapping operation - this should work on all backends 
    # Create a one-hot representation, multiply by the new class assignments, and sum
    x_remapped = ((x.unsqueeze(-1) == classes) * random_classes).sum(-1)
    
    return x_remapped


class CategoricalActivation(nn.Module):
    """
    Neural network module that transforms continuous features into categorical ones.
    Compatible with all backends (CUDA, MPS, CPU, ROCm).
    """
    def __init__(self, categorical_p=0.1, ordered_p=0.7, num_classes_sampler=None):
        """
        Initialize the CategoricalActivation module.
        
        Args:
            categorical_p: Probability of converting a feature to categorical
            ordered_p: Probability of randomizing the order of categorical features
            num_classes_sampler: Function that samples the number of classes
        """
        super().__init__()
        
        #TODO: refactor this, num classes should not be hardcoded
        if num_classes_sampler is None:
            num_classes_sampler = zipf_sampler_f(0.8, 1, 10)
        self.categorical_p = categorical_p
        self.ordered_p = ordered_p
        self.num_classes_sampler = num_classes_sampler

    def forward(self, x):
        """
        Forward pass to transform continuous features into categorical ones.
        
        Args:
            x: Input tensor (shape: T, B, H)
            
        Returns:
            x: Transformed tensor with categorical features
        """
        # Get device and dtype from input tensor
        device = x.device
        dtype = x.dtype
        
        # Apply softsign activation
        x = nn.Softsign()(x)
        
        # Sample number of classes
        num_classes = self.num_classes_sampler()

        # Generate mask for categorical features (ensure it's on the same device)
        categorical_classes = (torch.rand((x.shape[1], x.shape[2]), device=device) < self.categorical_p)
        
        # Create tensor for class boundaries on the right device
        class_boundaries = torch.zeros((num_classes - 1, x.shape[1], x.shape[2]), 
                                      device=device, dtype=dtype)
        
        # Sample a different index for each hidden dimension, but shared for all batches
        for b in range(x.shape[1]):
            for h in range(x.shape[2]):
                # Generate random indices on the appropriate device
                ind = torch.randint(0, x.shape[0], (num_classes - 1,), device=device)
                class_boundaries[:, b, h] = x[ind, b, h]

        # Transform continuous features to categorical based on boundaries
        for b in range(x.shape[1]):
            if not categorical_classes[b].any():
                continue  # Skip if no categorical features for this batch
                
            x_rel = x[:, b, categorical_classes[b]]
            boundaries_rel = class_boundaries[:, b, categorical_classes[b]].unsqueeze(1)
            
            # Convert to categorical by counting how many boundaries each value crosses
            new_values = (x_rel > boundaries_rel).sum(dim=0).float() - num_classes / 2
            x[:, b, categorical_classes[b]] = new_values

        # Determine which categorical features should have randomized orderings
        ordered_classes = (torch.rand((x.shape[1], x.shape[2]), device=device) < self.ordered_p)
        ordered_classes = torch.logical_and(ordered_classes, categorical_classes)
        
        # Only proceed if there are ordered classes to randomize
        if ordered_classes.any():
            # Randomize class assignments for ordered categorical features
            x[:, ordered_classes] = randomize_classes(x[:, ordered_classes], num_classes)

        return x
