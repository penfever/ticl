import random
import torch
import logging
import numpy as np
import time
from typing import List, Dict, Tuple, Any, Optional, Union
from functools import lru_cache

# Setup logging
logger = logging.getLogger(__name__)

# Global variables for semantic data (used as in the original implementation)
semantic_data = torch.zeros(1, 1)  # Placeholder, will be replaced in runtime
semantic_data_column_names = []  # Placeholder

class ClassificationAdapterOptimized:
    """
    Optimized version of the ClassificationAdapter class.
    This implementation focuses only on the _apply_semantic_prior method.
    """
    
    def __init__(self, base_prior, h):
        """Initialize with the same parameters as the original"""
        self.base_prior = base_prior
        self.h = h
        
        # Class-level cache for semantic data to avoid redundant reloading
        self._semantic_data_cache = {}
        
        # Class-level cache for token patterns to avoid recomputation
        self._class_token_patterns_cache = {}
        
        # Track performance statistics for monitoring
        self._performance_stats = {
            'semantic_prior_calls': 0,
            'cache_hits': 0,
            'total_time': 0,
            'data_loading_time': 0,
            'token_processing_time': 0,
            'feature_assignment_time': 0
        }
    
    def create_semantic_class_mapping(self, num_classes, semantic_data, device):
        """
        Create a mapping between class indices and semantic token patterns.
        Each class is associated with a specific pattern of semantic tokens.
        
        Parameters:
        -----------
        num_classes : int
            Number of classes to create mappings for
        semantic_data : torch.Tensor
            Tensor of semantic tokens with shape [num_semantic_classes, num_tokens]
        device : torch.device
            Device to use
            
        Returns:
        --------
        dict
            Mapping from class indices to token patterns
        """
        # Check cache first
        cache_key = f"{num_classes}_{semantic_data.shape[0]}_{id(semantic_data)}"
        if cache_key in self._class_token_patterns_cache:
            return self._class_token_patterns_cache[cache_key]
            
        # For each class, assign a characteristic pattern of semantic tokens
        class_token_patterns = {}
        
        num_semantic_classes = semantic_data.shape[0]
        
        # Get a seed for reproducibility if configured
        seed = self.h.get('random_seed', None)
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
        
        # OPTIMIZATION: Vectorized selection of semantic classes for each class
        if num_classes <= num_semantic_classes:
            # Sample without replacement when we have enough classes
            selected_indices = torch.randperm(num_semantic_classes)[:num_classes].tolist()
        else:
            # If more classes than semantic classes, we'll have some duplicates
            selected_indices = torch.randint(0, num_semantic_classes, (num_classes,)).tolist()
                
        # Get the actual column names if available
        global semantic_data_column_names
        
        # OPTIMIZATION: Pre-compute token counts and indices in advance for all classes
        # Use more tokens per class to maximize information utilization
        # We'll use between 25% and 50% of all available tokens per class
        min_ratio, max_ratio = 0.25, 0.5
        min_tokens = max(5, int(semantic_data.shape[1] * min_ratio))  # At least 5 tokens
        max_tokens = min(int(semantic_data.shape[1] * max_ratio), semantic_data.shape[1])
        
        # OPTIMIZATION: Generate all token counts at once
        num_signature_tokens = torch.randint(min_tokens, max_tokens + 1, (num_classes,)).tolist()
        
        # OPTIMIZATION: Generate all indices at once for all classes
        all_indices = [
            random.sample(range(semantic_data.shape[1]), num_signature_tokens[class_idx])
            for class_idx in range(num_classes)
        ]
        
        # Create all token patterns in one batch
        for class_idx in range(num_classes):
            # Get the semantic class for this class
            semantic_class = selected_indices[class_idx]
            
            # Extract pre-generated indices for this class
            token_indices = all_indices[class_idx]
            
            # Extract the tokens for this class - keep on device
            signature_tokens = semantic_data[semantic_class, token_indices].to(device)
            
            # Create descriptive name for class
            class_name = f"Class_{class_idx}_Type_{semantic_class}"
            
            # Get column name if available
            column_name = None
            if semantic_data_column_names and len(semantic_data_column_names) > semantic_class:
                column_name = semantic_data_column_names[semantic_class]
            
            # Store the class token pattern
            class_token_patterns[class_idx] = {
                'tokens': signature_tokens,
                'semantic_class': semantic_class,
                'class_name': class_name,
                'column_name': column_name
            }
        
        # Cache the result
        self._class_token_patterns_cache[cache_key] = class_token_patterns
        
        return class_token_patterns
    
    def create_semantic_targets(self, y, class_token_patterns):
        """
        Create semantic targets tensor based on class assignments in y.
        
        Parameters:
        -----------
        y : torch.Tensor
            The class targets tensor with shape (samples, batch_size)
        class_token_patterns : dict
            Mapping from class indices to token patterns
            
        Returns:
        --------
        torch.Tensor
            Semantic targets tensor
        """
        # OPTIMIZATION: Avoid repeatedly checking dict keys
        valid_classes = set(class_token_patterns.keys())
        
        # Get shape info
        if len(y.shape) == 3:
            # Handle case where y has shape (samples, batch_size, 1)
            y = y.squeeze(-1)
        
        sample_size, batch_size = y.shape
        
        # Create targets tensor - initialized with ignore index (-100)
        semantic_targets = torch.full((sample_size, batch_size), -100, 
                                    device=y.device, dtype=torch.long)
        
        # Convert y to long for indexing
        y_long = y.to(torch.long)
        
        # OPTIMIZATION: Vectorized assignment for valid classes
        # Create a mask of valid class indices in y
        valid_mask = torch.zeros_like(y_long, dtype=torch.bool)
        
        for class_idx in valid_classes:
            valid_mask = valid_mask | (y_long == class_idx)
            
            # Get semantic class for this class
            semantic_class = class_token_patterns[class_idx]['semantic_class']
            
            # Assign semantic targets where y matches this class
            semantic_targets = torch.where(
                y_long == class_idx,
                torch.tensor(semantic_class, device=y.device, dtype=torch.long),
                semantic_targets
            )
        
        return semantic_targets
    
    def _apply_semantic_prior(self, x, semantic_features, device, y=None):
        """
        Optimized implementation of the semantic prior application.
        Apply semantic prior to the features, ensuring consistent 
        relationships between semantic features and class labels.
        
        Parameters:
        -----------
        x : torch.Tensor
            The feature tensor with shape (samples, batch_size, num_features)
        semantic_features : list
            List of indices of semantic features
        device : torch.device
            The device to use
        y : torch.Tensor, optional
            The target tensor with shape (samples, batch_size)
            
        Returns:
        --------
        tuple
            (Updated feature tensor, Semantic info dictionary)
        """
        # Track performance
        start_time = time.time()
        self._performance_stats['semantic_prior_calls'] += 1
        
        # Make a mutable copy of the feature tensor
        x_new = x.clone()
        
        global semantic_data, semantic_data_column_names
        
        # Get the number of classes from config
        num_classes = max(2, self.h['num_classes'])
        
        # If y is not provided (initial call), generate proxy targets
        if y is None:
            # Create a temporary random target for initial feature generation
            y = torch.randint(0, num_classes, (x.shape[0], x.shape[1]), device=device).float()
        
        # OPTIMIZATION: Check class-level cache for semantic data
        data_start_time = time.time()
        
        # Cache key based on number of classes
        semantic_cache_key = f"semantic_data_{num_classes}"
        
        # Check if we need to reload semantic data
        if semantic_cache_key not in self._semantic_data_cache or semantic_data.shape[0] != num_classes:
            # Load new semantic data if needed with proper number of classes
            seed = self.h.get('random_seed', None)
            
            # Get random semantic data with proper randomization and caching
            from ticl.datasets.semantic_prior_data_loader import get_random_semantic_data
            semantic_data, semantic_data_column_names = get_random_semantic_data(
                num_classes=num_classes,
                seed=seed,
                use_cache=True
            )
            
            # Cache the loaded data
            self._semantic_data_cache[semantic_cache_key] = (semantic_data, semantic_data_column_names)
        else:
            # Use cached data
            self._performance_stats['cache_hits'] += 1
            semantic_data, semantic_data_column_names = self._semantic_data_cache[semantic_cache_key]
        
        # Make sure semantic_data is on the correct device
        semantic_data = semantic_data.to(device)
        num_semantic_classes = semantic_data.shape[0]
        
        data_loading_time = time.time() - data_start_time
        self._performance_stats['data_loading_time'] += data_loading_time
        
        # OPTIMIZATION: Cache token patterns at the class level
        token_start_time = time.time()
        
        # Create class-token mapping if not already created
        token_patterns_key = f"token_patterns_{num_classes}"
        if token_patterns_key not in self._class_token_patterns_cache:
            self._class_token_patterns_cache[token_patterns_key] = self.create_semantic_class_mapping(
                num_classes, semantic_data, device
            )
        
        class_token_patterns = self._class_token_patterns_cache[token_patterns_key]
        
        # Create semantic targets for training
        semantic_targets = self.create_semantic_targets(y, class_token_patterns)
        
        token_processing_time = time.time() - token_start_time
        self._performance_stats['token_processing_time'] += token_processing_time
        
        # Basic dimensions
        batch_size = x.shape[1]
        sample_size = x.shape[0]
        n_semantic_features = len(semantic_features)
        
        # Control randomness for reproducibility
        seed = self.h.get('random_seed', None)
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed) 
            np.random.seed(seed)
        
        # Convert y to integer class indices for easier processing
        y_int = y.to(torch.int64)
        
        # Dictionary to track causal features for each batch
        causal_features_map = {}
        
        # OPTIMIZATION: Feature assignment using vectorized operations
        feature_start_time = time.time()
        
        # OPTIMIZATION: Pre-compute token pools for all semantic classes
        # Create a direct tensor lookup table instead of dictionaries
        token_values_table = semantic_data.to(dtype=torch.float32, device=device)
        
        # OPTIMIZATION: Pre-extract tokens for class patterns
        class_tokens_table = torch.zeros(
            (num_classes, semantic_data.shape[1]), 
            dtype=torch.float32, 
            device=device
        )
        
        class_semantic_map = torch.zeros(num_classes, dtype=torch.long, device=device)
        
        for class_idx in range(num_classes):
            if class_idx in class_token_patterns:
                pattern = class_token_patterns[class_idx]
                tokens = pattern['tokens']
                
                # Get semantic class for this class
                semantic_class = pattern['semantic_class']
                class_semantic_map[class_idx] = semantic_class
        
        # OPTIMIZATION: Process all batch items at once using vectorized operations
        
        # 1. Create a mask of size [samples, batch_size] identifying where each class appears
        class_masks = [(y_int == class_idx) for class_idx in range(num_classes)]
        
        # 2. Create semantic feature values 
        for feat_idx, feature_pos in enumerate(semantic_features):
            # Determine which pattern to use for this feature (cycle if needed)
            pattern_offset = feat_idx % n_semantic_features
            
            # Create a tensor to hold values for this feature
            feature_values = torch.zeros((sample_size, batch_size), device=device)
            
            # Assign token values for each class
            for class_idx in range(num_classes):
                # Only process if this class exists in the patterns
                if class_idx in class_token_patterns:
                    # Get class mask
                    mask = class_masks[class_idx]
                    
                    # Get semantic class for this class
                    semantic_class = class_token_patterns[class_idx]['semantic_class']
                    
                    # Choose token indices based on feature position
                    token_idx = (feat_idx + class_idx) % semantic_data.shape[1]
                    
                    # Get token value for this class and feature
                    token_value = token_values_table[semantic_class, token_idx]
                    
                    # Assign token value to feature where this class appears
                    feature_values = torch.where(mask, token_value, feature_values)
            
            # OPTIMIZATION: Single assignment to the output tensor 
            x_new[:, :, feature_pos] = feature_values
        
        feature_assignment_time = time.time() - feature_start_time
        self._performance_stats['feature_assignment_time'] += feature_assignment_time
        
        # Prepare return info
        semantic_info = {
            'semantic_targets': semantic_targets,
            'class_token_patterns': class_token_patterns
        }
        
        # Update total time
        total_time = time.time() - start_time
        self._performance_stats['total_time'] += total_time
        
        # Log performance details occasionally
        if self._performance_stats['semantic_prior_calls'] % 10 == 0:
            calls = self._performance_stats['semantic_prior_calls']
            avg_time = self._performance_stats['total_time'] / calls
            cache_ratio = self._performance_stats['cache_hits'] / calls if calls > 0 else 0
            
            logger.debug(f"_apply_semantic_prior: avg_time={avg_time:.4f}s, "
                        f"cache_hit_ratio={cache_ratio:.2f}, calls={calls}")
        
        return x_new, semantic_info
    
    def get_performance_stats(self):
        """
        Get performance statistics for optimization tracking.
        
        Returns:
        --------
        dict
            Dictionary of performance statistics
        """
        stats = dict(self._performance_stats)
        
        # Calculate averages
        calls = stats['semantic_prior_calls']
        if calls > 0:
            stats['avg_total_time'] = stats['total_time'] / calls
            stats['avg_data_loading_time'] = stats['data_loading_time'] / calls
            stats['avg_token_processing_time'] = stats['token_processing_time'] / calls
            stats['avg_feature_assignment_time'] = stats['feature_assignment_time'] / calls
            stats['cache_hit_ratio'] = stats['cache_hits'] / calls
        
        return stats