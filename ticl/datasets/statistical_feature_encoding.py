"""
Statistical feature encoding for semantic models.
This module provides utilities for encoding numerical features as statistical tokens
that can be used in both training and inference.
"""

import torch
import numpy as np
from typing import Dict, Optional, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)


def calculate_feature_statistics(
    X: torch.Tensor, 
    numeric_indices: Optional[List[int]] = None,
    return_quantiles: bool = True
) -> Dict[int, Dict[str, Any]]:
    """
    Calculate statistics for numerical features in a tensor.
    
    Parameters:
    -----------
    X : torch.Tensor
        Input tensor with shape [samples, batch, features] or [batch, features]
    numeric_indices : list of int, optional
        Indices of numeric features. If None, all features are considered numeric.
    return_quantiles : bool
        Whether to calculate and return quantile information
        
    Returns:
    --------
    Dict[int, Dict[str, any]]
        Dictionary mapping feature index to statistics dictionary containing:
        - min: minimum value
        - max: maximum value
        - mean: mean value
        - std: standard deviation
        - quantiles: dictionary with 25%, 50%, 75% quantiles (if return_quantiles=True)
    """
    # Handle different input shapes
    if X.dim() == 3:
        # [samples, batch, features] -> [samples*batch, features]
        X_flat = X.reshape(-1, X.shape[-1])
    elif X.dim() == 2:
        # Already in [batch, features] format
        X_flat = X
    else:
        raise ValueError(f"Unsupported input tensor shape: {X.shape}")
    
    # Move to CPU if needed
    if X_flat.device.type != 'cpu':
        X_flat = X_flat.cpu()
    
    # Convert to numpy for faster statistical operations
    X_np = X_flat.numpy()
    
    # Determine which features to process
    if numeric_indices is None:
        numeric_indices = list(range(X_np.shape[1]))
    
    # Calculate statistics for each numeric feature
    feature_stats = {}
    
    for idx in numeric_indices:
        if idx >= X_np.shape[1]:
            continue
        
        # Get values for this feature
        values = X_np[:, idx]
        
        # Skip if all zeros or all the same value (uninformative)
        if np.all(values == 0) or np.all(values == values[0]):
            continue
            
        # Calculate basic statistics
        stats = {
            'type': 'numerical',
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'mean': float(np.mean(values)),
            'std': float(np.std(values))
        }
        
        # Add quantiles if requested
        if return_quantiles:
            stats['quantiles'] = {
                '25%': float(np.percentile(values, 25)),
                '50%': float(np.percentile(values, 50)),
                '75%': float(np.percentile(values, 75))
            }
        
        feature_stats[idx] = stats
    
    return feature_stats


def encode_numerical_features_as_tokens(
    X: torch.Tensor,
    feature_stats: Dict[int, Dict[str, Any]],
    reserved_indices: List[int],
    column_mapping: Optional[Dict[int, str]] = None
) -> Tuple[torch.Tensor, Dict]:
    """
    Encode numerical features as discrete tokens for statistical representation.
    
    Parameters:
    -----------
    X : torch.Tensor
        Input tensor with shape [samples, batch, features] or [batch, features]
    feature_stats : Dict[int, Dict[str, any]]
        Statistics for each feature as returned by calculate_feature_statistics
    reserved_indices : List[int]
        Indices of reserved features to fill with statistical tokens
    column_mapping : Dict[int, str], optional
        Mapping from feature indices to column names
        
    Returns:
    --------
    Tuple[torch.Tensor, Dict]
        Modified tensor with statistical tokens and token mapping information
    """
    if not reserved_indices or not feature_stats:
        return X, {}
    
    # Create a copy of X to avoid modifying the original
    X = X.clone()
    
    # Create token mapping to return
    token_mapping = {
        'low_tokens': [],
        'med_tokens': [],
        'high_tokens': [],
        'feature_indices': [],
        'column_names': []
    }
    
    # Get list of numeric features from stats
    numeric_features = sorted([idx for idx in feature_stats.keys()])
    
    # Skip if no numeric features found
    if not numeric_features:
        return X, token_mapping
    
    # Process up to two numeric features
    for i, reserved_idx in enumerate(reserved_indices[:min(2, len(reserved_indices))]):
        if i >= len(numeric_features):
            break
            
        # Select the numeric feature to encode
        numeric_idx = numeric_features[i]
        stats = feature_stats[numeric_idx]
        
        # Set token values based on feature index (different ranges for different features)
        token_offset = i * 30  # 0 for first feature, 30 for second
        low_token = 10 + token_offset
        med_token = 20 + token_offset
        high_token = 30 + token_offset
        
        # Add to token mapping
        token_mapping['low_tokens'].append(low_token)
        token_mapping['med_tokens'].append(med_token)
        token_mapping['high_tokens'].append(high_token)
        token_mapping['feature_indices'].append(numeric_idx)
        
        # Add column name if available
        if column_mapping and numeric_idx in column_mapping:
            column_name = column_mapping[numeric_idx]
            token_mapping['column_names'].append(column_name)
        else:
            token_mapping['column_names'].append(f"feature_{numeric_idx}")
        
        # Get quantile thresholds
        if 'quantiles' in stats:
            quant_25 = stats['quantiles'].get('25%', stats['min'] + (stats['max'] - stats['min']) * 0.25)
            quant_75 = stats['quantiles'].get('75%', stats['min'] + (stats['max'] - stats['min']) * 0.75)
        else:
            # Define even thresholds if no quantiles available
            range_val = stats['max'] - stats['min']
            quant_25 = stats['min'] + range_val * 0.25
            quant_75 = stats['min'] + range_val * 0.75
        
        # Apply encoding based on tensor shape
        if X.dim() == 3:
            # [samples, batch, features]
            for j in range(X.shape[0]):
                for k in range(X.shape[1]):
                    if numeric_idx < X.shape[2]:
                        value = X[j, k, numeric_idx].item()
                        
                        # Set token based on value
                        if value <= quant_25:
                            X[j, k, reserved_idx] = low_token
                        elif value <= quant_75:
                            X[j, k, reserved_idx] = med_token
                        else:
                            X[j, k, reserved_idx] = high_token
        elif X.dim() == 2:
            # [batch, features]
            for j in range(X.shape[0]):
                if numeric_idx < X.shape[1]:
                    value = X[j, numeric_idx].item()
                    
                    # Set token based on value
                    if value <= quant_25:
                        X[j, reserved_idx] = low_token
                    elif value <= quant_75:
                        X[j, reserved_idx] = med_token
                    else:
                        X[j, reserved_idx] = high_token
    
    # Add stats to the token mapping
    token_mapping['feature_stats'] = feature_stats
    
    return X, token_mapping


def get_numeric_feature_indices(X: torch.Tensor, threshold: float = 0.7) -> List[int]:
    """
    Identify numeric feature indices by analyzing the data.
    A feature is considered numeric if it has a high proportion of unique values.
    
    Parameters:
    -----------
    X : torch.Tensor
        Input tensor of shape [samples, batch, features] or [batch, features]
    threshold : float
        Threshold of uniqueness ratio to consider a feature numeric
        
    Returns:
    --------
    List[int]
        Indices of features that appear to be numeric
    """
    # Flatten the first two dimensions if 3D tensor
    if X.dim() == 3:
        X_flat = X.reshape(-1, X.shape[-1])
    else:
        X_flat = X
    
    # Move to CPU if needed
    if X_flat.device.type != 'cpu':
        X_flat = X_flat.cpu()
    
    numeric_indices = []
    
    # For each feature, calculate uniqueness ratio
    for i in range(X_flat.shape[1]):
        values = X_flat[:, i].numpy()
        unique_values = np.unique(values)
        
        # Skip features with very few unique values (likely categorical)
        if len(unique_values) <= 5:
            continue
            
        # Calculate ratio of unique values to total values
        uniqueness_ratio = len(unique_values) / len(values)
        
        # If high uniqueness, likely numeric
        if uniqueness_ratio > threshold:
            numeric_indices.append(i)
    
    return numeric_indices