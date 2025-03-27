import random
import torch
import logging
import numpy as np
import time
from typing import List, Dict, Tuple, Any, Optional, Union
from functools import lru_cache

from ticl.utils import (get_nan_value, normalize_by_used_features_f, normalize_data,
                            remove_outliers)

from ticl.distributions import sample_distributions, uniform_int_sampler_f, parse_distributions, safe_randint
from ticl.datasets.semantic_prior_data_loader import get_random_semantic_data
from ticl.datasets.labeled_numeric_prior_data_loader import labeled_numeric_data, get_random_value, column_metadata
from .utils import CategoricalActivation, randomize_classes

# Setup logging
logger = logging.getLogger(__name__)
memory_logger = logging.getLogger("memory_profiling")

# For backward compatibility, using the same variable name
from ticl.datasets.semantic_prior_data_loader import random_tensor, semantic_data_column_names
semantic_data = random_tensor

class BalancedBinarize:
    def __call__(self, x):
        return (x > torch.median(x)).float()


def class_sampler_f(min_, max_):
    def s():
        if random.random() > 0.5:
            return uniform_int_sampler_f(min_, max_)()
        return 2
    return s


class RegressionNormalized:
    def __call__(self, x):
        # x has shape (T,B)

        # already gets normalized later
        # TODO: Normalize to -1, 1 or gaussian normal
        # maxima = torch.max(x, 0)[0]
        # minima = torch.min(x, 0)[0]
        # norm = (x - minima) / (maxima-minima)

        return x


class MulticlassSteps:
    """"Sample piecewise constant functions 
    with random number of steps 
    and random class boundaries"""

    def __init__(self, num_classes, max_steps=10):
        self.num_classes = class_sampler_f(2, num_classes)()
        self.num_steps = np.random.randint(1, max_steps) if max_steps > 1 else 1

    def __call__(self, x):
        # x has shape (T,B,H) ?!
        # x has shape (samples, batch)
        # CAUTION: This samples the same idx in sequence for each class boundary in a batch
        class_boundary_indices = torch.randint(0, x.shape[0], ((self.num_classes - 1) * (self.num_steps - 1) + 1,), device=x.device)
        class_boundaries_sorted, _ = x[class_boundary_indices].sort(axis=0)
        step_assignments = torch.searchsorted(class_boundaries_sorted.T.contiguous(), x.T.contiguous()).T
        class_assignments = torch.randint(0, self.num_classes, (step_assignments.max() + 1, x.shape[1]), device=x.device)
        classes = torch.gather(class_assignments, 0, step_assignments)
        return classes


class MulticlassRank:
    def __init__(self, num_classes, ordered_p=0.5):
        self.num_classes = class_sampler_f(2, num_classes)()
        self.ordered_p = ordered_p

    def __call__(self, x):
        # x has shape (T,B,H)

        # CAUTION: This samples the same idx in sequence for each class boundary in a batch
        class_boundaries = torch.randint(0, x.shape[0], (self.num_classes - 1,))
        class_boundaries = x[class_boundaries].unsqueeze(1)

        d = (x > class_boundaries).sum(axis=0)

        randomized_classes = torch.rand((d.shape[1], )) > self.ordered_p
        d[:, randomized_classes] = randomize_classes(d[:, randomized_classes], self.num_classes)
        reverse_classes = torch.rand((d.shape[1],)) > 0.5
        d[:, reverse_classes] = self.num_classes - 1 - d[:, reverse_classes]
        return d


class ClassificationAdapter:
    # This class samples the number of features actually use (num_features_used), the number of samples
    # adds NaN and potentially categorical features
    # and discretizes the classification output variable
    # It's instantiated anew for each batch that's created
    def __init__(self, base_prior, config):
        self.h = sample_distributions(parse_distributions(config))
        self.base_prior = base_prior
        
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
        
        if self.h['max_num_classes'] == 0:
            self.class_assigner = RegressionNormalized()
        else:
            if self.h['num_classes'] > 1 and not self.h['balanced']:
                if self.h['multiclass_type'] == 'rank':
                    self.class_assigner = MulticlassRank(
                        self.h['num_classes'], 
                        ordered_p=self.h['output_multiclass_ordered_p']
                    )
                elif self.h['multiclass_type'] == 'steps':
                    self.class_assigner = MulticlassSteps(self.h['num_classes'], self.h['multiclass_max_steps'])
                else:
                    raise ValueError("Unknown Multiclass type")
            elif self.h['num_classes'] == 2 and self.h['balanced']:
                self.class_assigner = BalancedBinarize()
            elif self.h['num_classes'] > 2 and self.h['balanced']:
                raise NotImplementedError("Balanced multiclass training is not possible")

    def drop_for_reason(self, x, v):
        # Categorical activation only called here
        nan_prob_sampler = CategoricalActivation(ordered_p=0.0, categorical_p=1.0, num_classes_sampler=lambda: 20)
        d = nan_prob_sampler(x)
        # TODO: Make a different ordering for each activation
        # actually only half that probability but that's fine
        x[d < torch.rand((1, d.shape[1], d.shape[2]), device=x.device) * 20 * self.h['nan_prob_a_reason'] - 10] = v
        return x

    def drop_for_no_reason(self, x, v):
        x[torch.rand(x.shape, device=x.device) < random.random() * self.h['nan_prob_no_reason']] = v
        return x
        
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

    def __call__(self, batch_size, n_samples, num_features, device, epoch=None, single_eval_pos=None):
        info = {}
        # num_features is constant for all batches, num_features_used is passed down to wrapped priors to change number of features
        if self.h['feature_curriculum']:
            num_features = min(num_features, epoch + 1)
        if self.h['num_features_sampler'] == 'uniform':
            num_features_used = safe_randint(1, num_features)
        elif self.h['num_features_sampler'] == 'double_sample':
            num_features_used = safe_randint(1, safe_randint(1, num_features))
        else:
            raise ValueError(f"Unknown num_features_sampler: {self.h['num_features_sampler']}")
        args = {'device': device, 'n_samples': n_samples, 'num_features': num_features_used,
                'batch_size': batch_size, 'epoch': epoch, 'single_eval_pos': single_eval_pos}
        
        # Get batch from base prior - handles both old and new interfaces
        try:
            # Try the new interface with causality info
            result = self.base_prior.get_batch(**args)
            if len(result) == 4:  # New interface with info dictionary
                x, y, y_, prior_info = result
                if 'causality_info' in prior_info:
                    info['causality_info'] = prior_info['causality_info']
            else:  # Old interface
                x, y, y_ = result
        except Exception:
            # Fall back to old interface
            x, y, y_ = self.base_prior.get_batch(**args)
        # x is of shape (n_samples, batch_size, num_features_used)
        assert x.shape[2] == num_features_used

        if self.h['nan_prob_no_reason']+self.h['nan_prob_a_reason'] > 0 and random.random() > 0.5:  # Only one out of two datasets should have nans
            if random.random() < self.h['nan_prob_no_reason']:  # Missing for no reason
                x = self.drop_for_no_reason(x, get_nan_value(self.h['set_value_to_nan']))

            if random.random() < self.h['nan_prob_a_reason']:  # Missing for a reason
                x = self.drop_for_reason(x, get_nan_value(self.h['set_value_to_nan']))

        # Categorical features and Semantic features
        categorical_features = []
        semantic_features = []
        
        # Check for semantic features parameter
        semantic_feature_p = self.h.get('semantic_feature_p', 0.0)
        info['semantic_feature_p'] = semantic_feature_p
        
        # If semantic feature probability is > 0, prepare to add 50 extra features
        use_semantic_features = semantic_feature_p > 0.0
        
        if random.random() < self.h['categorical_feature_p']:
            p = random.random()
            for col in range(x.shape[2]):
                num_unique_features = max(round(random.gammavariate(1, 10)), 2)
                m = MulticlassRank(num_unique_features, ordered_p=0.3)
                if random.random() < p:
                    categorical_features.append(col)
                    x[:, :, col] = m(x[:, :, col])
        
        # Add the 50 extra semantic features if needed
        if use_semantic_features:
            memory_logger.debug(f"=== Adding semantic features with probability {semantic_feature_p} ===")
            memory_logger.debug(f"Tensor x before adding semantic features: shape={x.shape}, device={x.device}, dtype={x.dtype}")
            
            num_semantic_features = 50
            info['num_semantic_features'] = num_semantic_features
            memory_logger.debug(f"Adding {num_semantic_features} semantic features")
            
            # Add the semantic features (initialized to 0.0)
            semantic_padding = torch.zeros((x.shape[0], x.shape[1], num_semantic_features), device=device)
            
            # Concatenate to original tensor
            original_shape = x.shape
            x = torch.cat([x, semantic_padding], dim=2)
            memory_logger.debug(f"After concatenation: shape changed from {original_shape} to {x.shape}")
            
            # Record indices of semantic features
            semantic_features = list(range(x.shape[2] - num_semantic_features, x.shape[2]))
            memory_logger.debug(f"Semantic feature indices: {semantic_features[:5]}... to {semantic_features[-1]}")
            
            # Apply semantic prior to the features with class-based consistency
            memory_logger.debug(f"Applying semantic prior to features")
            x, semantic_info = self._apply_semantic_prior(x, semantic_features, device, y)
            
            # Verify semantic features were applied
            semantic_region = x[:, :, semantic_features]
            non_zeros = (semantic_region != 0).sum().item()
            total = semantic_region.numel()
            fill_percentage = (non_zeros / total) * 100
            memory_logger.debug(f"Semantic features filled: {non_zeros}/{total} elements ({fill_percentage:.1f}% non-zero)")
            
            # Generate synthetic column names for tracking during training
            # Sample from labeled_numeric_prior_data_loader
            synthetic_column_names = []
            available_columns = list(labeled_numeric_data.keys())
            
            # First, use the reserved feature names (always consistent)
            if len(semantic_features) >= 3:
                synthetic_column_names.extend([
                    "feature_statistics",
                    "class_statistics", 
                    "table_metadata"
                ])
                # Start with indices after the reserved features
                start_idx = 3
            else:
                start_idx = 0
            
            # Then add random numeric column names for the remaining features
            for i in range(start_idx, len(semantic_features)):
                # Sample a random column name
                col_name = random.choice(available_columns)
                synthetic_column_names.append(col_name)
            
            # Add the synthetic column names to the info dictionary
            info['semantic_column_names'] = synthetic_column_names
            
            # Store semantic information in the info dictionary
            info.update(semantic_info)
            
            # Verify semantic targets
            if 'semantic_targets' in semantic_info:
                semantic_targets = semantic_info['semantic_targets']
                valid_targets = (semantic_targets != -100).sum().item()
                total_targets = semantic_targets.numel()
                valid_percentage = (valid_targets / total_targets) * 100
                
                # Log a sample of the semantic targets
                if semantic_targets.numel() > 0:
                    memory_logger.debug(f"Semantic targets sample (first 5x5 values):")
                    for i in range(min(5, semantic_targets.shape[0])):
                        row_values = semantic_targets[i, :min(5, semantic_targets.shape[1])].tolist()
                        memory_logger.debug(f"  Row {i}: {row_values}")
                    
                    # Add batch-level semantic tokens for the semantic model
                    # This allows the semantic model to directly access the tokens
                    # Select the first class's tokens for simplicity
                    if 'class_token_patterns' in semantic_info:
                        first_class = min(semantic_info['class_token_patterns'].keys())
                        first_tokens = semantic_info['class_token_patterns'][first_class]['tokens']
                        
                        # Add the semantic tokens to the info dictionary
                        info['semantic_tokens'] = first_tokens
                        memory_logger.debug(f"Added semantic_tokens to info with shape: {first_tokens.shape}")
            
            memory_logger.debug(f"=== Semantic features added successfully ===")
        else:
            info['semantic_targets'] = None
        
        info['categorical_features'] = categorical_features
        info['semantic_features'] = semantic_features
        x = remove_outliers(x, categorical_features=categorical_features)
        x, y = normalize_data(x), normalize_data(y)

        # Cast to classification if enabled
        # In case of regression two normalizations.
        y = self.class_assigner(y).float()
        if self.h['max_num_classes'] == 0:
            # Inpute potential nan values after normalization
            y[y.isnan()] = 0
            
        # Perform statistical analysis on features if needed
        if 'causality_info' in info and use_semantic_features:
            try:
                # Import statistical analysis tools
                from ticl.priors.feature_statistical_analyzer import (
                    analyze_numerical_feature, 
                    analyze_categorical_feature,
                    get_class_description_from_stats
                )
                
                # Only perform detailed analysis if needed
                feature_stats = {}
                statistical_class_terms = {}
                
                if len(semantic_features) >= 3:  # Only if we have reserved features
                    # Get the reserved feature indices
                    reserved_indices = semantic_features[-3:]
                    info['reserved_feature_indices'] = reserved_indices
                    
                    # Lightweight analysis of features for statistical terms
                    for b in range(x.shape[1]):  # For each batch
                        batch_stats = {}
                        if b in info['causality_info']:
                            batch_causality = info['causality_info'][b]
                            causal_features = batch_causality.get('causal_features', [])
                            
                            # Analyze just a sample of features
                            for feature_idx in causal_features[:5]:  # Just analyze top causal features
                                feature_values = x[:, b, feature_idx]
                                batch_labels = y[:, b]
                                
                                # Skip if all values are the same
                                if torch.all(feature_values == feature_values[0]):
                                    continue
                                
                                is_categorical = feature_idx in categorical_features
                                if is_categorical:
                                    stats = analyze_categorical_feature(
                                        feature_values, 
                                        batch_labels,
                                        feature_name=f"feature_{feature_idx}",
                                        is_causal=True
                                    )
                                    stats['type'] = 'categorical'
                                else:
                                    stats = analyze_numerical_feature(
                                        feature_values, 
                                        batch_labels,
                                        feature_name=f"feature_{feature_idx}",
                                        is_causal=True
                                    )
                                    stats['type'] = 'numerical'
                                
                                batch_stats[feature_idx] = stats
                            
                            # Get class descriptions based on statistical properties
                            batch_unique_classes = torch.unique(batch_labels[batch_labels != -100]).tolist()
                            for class_id in batch_unique_classes:
                                terms = get_class_description_from_stats(batch_stats, int(class_id))
                                if b not in statistical_class_terms:
                                    statistical_class_terms[b] = {}
                                statistical_class_terms[b][int(class_id)] = terms
                        
                        # Store feature statistics for this batch
                        feature_stats[b] = batch_stats
                    
                    # Add statistical information to info dictionary
                    info['feature_stats'] = feature_stats
                    info['statistical_class_terms'] = statistical_class_terms
                    
                    # Add information about reserved features
                    info['reserved_features_content'] = {
                        'feature_1': 'feature_statistics' if len(reserved_indices) >= 1 else None,
                        'feature_2': 'class_statistical_terms' if len(reserved_indices) >= 2 else None,
                        'feature_3': 'table_metadata' if len(reserved_indices) >= 3 else None
                    }
            except Exception as e:
                logger.warning(f"Error during statistical analysis: {e}")
                pass
        
        # Append empty features if enabled
        if self.h['pad_zeros']:
            x = normalize_by_used_features_f(
                x, num_features_used, num_features)
            x = torch.cat(
                [x, torch.zeros((x.shape[0], x.shape[1], num_features - num_features_used), device=device)], -1)

        if torch.isnan(y).any():
            print('Nans in target!')
            if self.h['max_num_classes'] == 0:
                y[torch.isnan(y)] = 0
            else:
                y[torch.isnan(y)] = -100

        if self.h['max_num_classes'] != 0:
            for b in range(y.shape[1]):
                is_compatible, N = False, 0
                while not is_compatible and N < 10:
                    targets_in_train = torch.unique(y[:single_eval_pos, b], sorted=True)
                    targets_in_eval = torch.unique(y[single_eval_pos:, b], sorted=True)

                    is_compatible = len(targets_in_train) == len(targets_in_eval) and (
                        targets_in_train == targets_in_eval).all() and len(targets_in_train) > 1

                    if not is_compatible:
                        randperm = torch.randperm(x.shape[0])
                        x[:, b], y[:, b] = x[randperm, b], y[randperm, b]
                    N = N + 1
                if not is_compatible:
                    if self.h['max_num_classes'] != 0:
                        # todo check that it really does this and how many together
                        y[:, b] = -100  # Relies on CE having `ignore_index` set to -100 (default)
                    # todo check that it really does this and how many together
                    y[:, b] = -100  # Relies on CE having `ignore_index` set to -100 (default)
            for b in range(y.shape[1]):
                valid_labels = y[:, b] != -100
                y[valid_labels, b] = (y[valid_labels, b] > y[valid_labels, b].unique().unsqueeze(1)).sum(axis=0).unsqueeze(0).float()

                if y[valid_labels, b].numel() != 0:
                    num_classes_float = (y[valid_labels, b].max() + 1).cpu()
                    num_classes = num_classes_float.int().item()
                    assert num_classes == num_classes_float.item()
                    random_shift = torch.randint(0, num_classes, (1,), device=device)
                    y[valid_labels, b] = (y[valid_labels, b] + random_shift) % num_classes

        return x, y, y, info  # x.shape = (T,B,H)


class ClassificationAdapterPrior:
    def __init__(self, base_prior, **config):
        self.base_prior = base_prior
        self.config = config

    def get_batch(self, batch_size, n_samples, num_features, device, epoch=None, single_eval_pos=None):
        with torch.no_grad():
            args = {'device': device, 'n_samples': n_samples, 'num_features': num_features, 'epoch': epoch, 'single_eval_pos': single_eval_pos}
            x, y, y_, info = ClassificationAdapter(self.base_prior, self.config)(batch_size=batch_size, **args)
            x, y, y_ = x.detach(), y.detach(), y_.detach()

        return x, y, y_, info