import torch
import numpy as np

class PolynomialBoundaries:
    """
    Creates class boundaries using polynomial functions.
    
    Parameters
    ----------
    num_classes : int
        Number of classes to create
    degree : int, default=2
        Degree of the polynomial (2=quadratic, 3=cubic, etc.)
    """
    def __init__(self, num_classes, degree=2):
        self.num_classes = num_classes
        self.degree = degree
        
    def __call__(self, x):
        # x has shape (T,B,H)
        T, B = x.shape[0], x.shape[1]
        
        # Normalize x to [0, 1] range for better polynomial behavior
        x_min, x_max = x.min(), x.max()
        x_norm = (x - x_min) / (x_max - x_min) if x_max > x_min else x
        
        # Create polynomial coefficients
        # For each batch, we'll have a different polynomial
        coeffs = torch.randn((B, self.degree + 1))
        
        # Create normalized positions along sequence length
        positions = torch.linspace(0, 1, T).view(T, 1, 1).expand(T, B, 1)
        
        # Evaluate polynomials for each batch element
        # Start with the constant term
        result = torch.ones((T, B)) * coeffs[:, 0].view(1, B)
        
        # Add higher-order terms
        for i in range(1, self.degree + 1):
            result += coeffs[:, i].view(1, B) * positions[:, :, 0].pow(i)
        
        # Convert to class indices by binning
        min_val, max_val = result.min(), result.max()
        bins = torch.linspace(min_val, max_val, self.num_classes + 1)
        
        # Assign classes based on which bin each value falls into
        d = torch.zeros_like(result, dtype=torch.long)
        for i in range(self.num_classes):
            d += (result > bins[i]).long()
            
        return d


class PeriodicBoundaries:
    """
    Creates class boundaries using sinusoidal periodic functions.
    
    Parameters
    ----------
    num_classes : int
        Number of classes to create
    frequencies : int, default=3
        Number of different frequency components to combine
    """
    def __init__(self, num_classes, frequencies=3):
        self.num_classes = num_classes
        self.frequencies = frequencies
        
    def __call__(self, x):
        # x has shape (T,B,H)
        T, B = x.shape[0], x.shape[1]
        
        # Create position indices along sequence
        positions = torch.linspace(0, 2 * np.pi, T).view(T, 1)
        
        # Create different frequencies and phases for each batch
        freqs = torch.randint(1, self.frequencies + 1, (B, self.frequencies))
        phases = torch.rand((B, self.frequencies)) * 2 * np.pi
        amplitudes = torch.rand((B, self.frequencies))
        
        # Sum of sinusoids for each batch
        result = torch.zeros((T, B))
        for i in range(self.frequencies):
            sin_waves = torch.sin(freqs[:, i].view(1, B) * positions + phases[:, i].view(1, B))
            result += amplitudes[:, i].view(1, B) * sin_waves
        
        # Discretize into classes
        # Scale to [0, num_classes-1] range
        result = (result - result.min()) / (result.max() - result.min()) * (self.num_classes - 1)
        d = result.round().long()
        
        return d


class ClusteredBoundaries:
    """
    Creates class boundaries with multiple isolated regions of the same class.
    
    Parameters
    ----------
    num_classes : int
        Number of classes to create
    num_centers : int, default=5
        Number of cluster centers to use for generating boundaries
    """
    def __init__(self, num_classes, num_centers=5):
        self.num_classes = num_classes
        self.num_centers = num_centers
        
    def __call__(self, x):
        # x has shape (T,B,H)
        T, B = x.shape[0], x.shape[1]
        
        # Create random cluster centers for each batch
        # Centers are positioned along the sequence
        centers = torch.rand((B, self.num_centers)) * T
        centers = centers.sort(dim=1)[0]  # Sort centers for better distribution
        
        # Assign each class to a set of centers
        # We'll create a mapping from centers to classes
        center_classes = torch.randint(0, self.num_classes, (B, self.num_centers))
        
        # Calculate distance to each center for all positions
        positions = torch.arange(T).view(T, 1, 1).expand(T, B, self.num_centers)
        centers_expanded = centers.view(1, B, self.num_centers).expand(T, B, self.num_centers)
        distances = torch.abs(positions - centers_expanded)
        
        # Find closest center for each position
        closest_center = torch.argmin(distances, dim=2)
        
        # Map to corresponding class
        d = torch.zeros((T, B), dtype=torch.long)
        for b in range(B):
            d[:, b] = center_classes[b, closest_center[:, b]]
            
        return d


class ThresholdWithExceptions:
    """
    Creates generally monotonic boundaries but with specific exception regions.
    
    Parameters
    ----------
    num_classes : int
        Number of classes to create
    exception_p : float, default=0.2
        Probability of a value being an exception to the monotonic rule
    """
    def __init__(self, num_classes, exception_p=0.2):
        self.num_classes = num_classes
        self.exception_p = exception_p
        
    def __call__(self, x):
        # x has shape (T,B,H)
        T, B = x.shape[0], x.shape[1]
        
        # First create monotonic class assignments (like the original)
        # Sample class boundaries
        class_boundaries = torch.quantile(
            x.reshape(T, -1), 
            torch.linspace(0.0, 0.9, self.num_classes).to(x.device),
        ).unsqueeze(1)
        
        # Count how many boundaries each value exceeds
        d = (x > class_boundaries).sum(axis=0)
        
        # Now add exceptions - randomly reassign classes for some values
        exception_mask = torch.rand(T, B) < self.exception_p
        random_classes = torch.randint(0, self.num_classes, (T, B))
        d[exception_mask] = random_classes[exception_mask]
        
        return d


class InformationTheoreticBoundaries:
    """
    Creates boundaries based on information content of the input.
    Uses entropy and mutual information principles.
    
    Parameters
    ----------
    num_classes : int
        Number of classes to create
    window_size : int, default=5
        Size of rolling window to compute local statistics
    """
    def __init__(self, num_classes, window_size=5):
        self.num_classes = num_classes
        self.window_size = window_size
        
    def __call__(self, x):
        # x has shape (T,B,H)
        T, B = x.shape[0], x.shape[1]
        d = torch.zeros((T, B), dtype=torch.long)
        
        # Calculate local entropy using rolling windows
        for t in range(self.window_size, T):
            for b in range(B):
                # Get window of values
                window = x[t-self.window_size:t, b].flatten()
                
                # Convert to numpy for entropy calculation
                window_np = window.detach().cpu().numpy()
                
                # Estimate entropy (using histogram approximation)
                hist, _ = np.histogram(window_np, bins=min(10, len(window_np)))
                p = hist / hist.sum() if hist.sum() > 0 else hist
                p = p[p > 0]  # Remove zeros
                
                # Calculate entropy
                entropy = -np.sum(p * np.log2(p)) if len(p) > 0 else 0
                
                # Scale entropy to class index
                # Higher entropy -> higher class (more unpredictable regions)
                normalized_entropy = min(entropy / np.log2(min(10, len(window_np))), 1)
                class_idx = min(int(normalized_entropy * self.num_classes), self.num_classes - 1)
                d[t, b] = class_idx
        
        # For the initial window where we can't calculate entropy,
        # just use a simple threshold based on the values
        for t in range(min(self.window_size, T)):
            d[t, :] = (x[t, :] > x[t, :].mean()).long() * (self.num_classes - 1)
        
        return d