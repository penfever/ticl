import random
import numpy as np
import torch
import logging

from ticl.utils import (get_nan_value, normalize_by_used_features_f, normalize_data,
                             remove_outliers)

from ticl.distributions import sample_distributions, uniform_int_sampler_f, parse_distributions, safe_randint
from ticl.priors.boundaries import *
from ticl.datasets.semantic_prior_data_loader import get_random_semantic_data
from .utils import CategoricalActivation, randomize_classes

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
    """
    Creates multiclass classification targets with various non-monotonic boundary types.
    
    Parameters
    ----------
    num_classes : int
        Number of classes to create
    boundary_type : str
        Type of boundaries to use: 'monotonic', 'polynomial', 'periodic', 
        'clustered', 'threshold_exceptions', or 'information_theoretic'
    """
    def __init__(self, num_classes, boundary_type='monotonic', **kwargs):
        self.num_classes = num_classes
        self.boundary_type = boundary_type
        
        # Initialize the appropriate boundary generator
        if boundary_type == 'monotonic':
            self.ordered_p = kwargs.get('ordered_p', 0.5)
            self.boundary_generator = None  # Will use original logic
        elif boundary_type == 'polynomial':
            self.boundary_generator = PolynomialBoundaries(
                num_classes,
                degree=kwargs.get('degree', 2)
            )
        elif boundary_type == 'periodic':
            self.boundary_generator = PeriodicBoundaries(
                num_classes,
                frequencies=kwargs.get('frequencies', 3)
            )
        elif boundary_type == 'clustered':
            self.boundary_generator = ClusteredBoundaries(
                num_classes,
                num_centers=kwargs.get('num_centers', 5)
            )
        elif boundary_type == 'threshold_exceptions':
            self.boundary_generator = ThresholdWithExceptions(
                num_classes,
                exception_p=kwargs.get('exception_p', 0.2)
            )
        elif boundary_type == 'information_theoretic':
            self.boundary_generator = InformationTheoreticBoundaries(
                num_classes,
                window_size=kwargs.get('window_size', 5)
            )
        else:
            raise ValueError(f"Unknown boundary type: {boundary_type}")

    def __call__(self, x):
        # If using original monotonic logic
        if self.boundary_type == 'monotonic':
            # Use the original MulticlassRank logic
            class_boundaries = torch.randint(0, x.shape[0], (self.num_classes - 1,))
            class_boundaries = x[class_boundaries].unsqueeze(1)

            # Count how many boundaries each value exceeds
            d = (x > class_boundaries).sum(axis=0)

            # Randomly shuffle class assignments based on ordered_p
            randomized_classes = torch.rand((d.shape[1], )) > self.ordered_p
            d[:, randomized_classes] = self._randomize_classes(d[:, randomized_classes])
            
            # Randomly reverse class direction for ~half the samples
            reverse_classes = torch.rand((d.shape[1],)) > 0.5
            d[:, reverse_classes] = self.num_classes - 1 - d[:, reverse_classes]
            
            return d
        else:
            # Use the appropriate boundary generator
            return self.boundary_generator(x)
    
    def _randomize_classes(self, d):
        """Helper method to randomize class assignments"""
        if d.numel() == 0:
            return d
            
        randomized = torch.zeros_like(d)
        for b in range(d.shape[1]):
            # Create random mapping for each batch
            perm = torch.randperm(self.num_classes)
            for c in range(self.num_classes):
                randomized[:, b][d[:, b] == c] = perm[c]
                
        return randomized


class ClassificationAdapter:
    # This class samples the number of features actually use (num_features_used), the number of samples
    # adds NaN and potentially categorical features
    # and discretizes the classification output variable
    # It's instantiated anew for each batch that's created
    def __init__(self, base_prior, config):
        self.h = sample_distributions(parse_distributions(config))
        self.base_prior = base_prior
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
        # For each class, assign a characteristic pattern of semantic tokens
        class_token_patterns = {}
        
        num_semantic_classes = semantic_data.shape[0]
        
        # Get a seed for reproducibility if configured
        seed = self.h.get('random_seed', None)
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
        
        # Vectorized selection of semantic classes for each class
        if num_classes <= num_semantic_classes:
            # Sample without replacement when we have enough classes
            selected_indices = torch.randperm(num_semantic_classes)[:num_classes].tolist()
        else:
            # If more classes than semantic classes, we'll have some duplicates
            selected_indices = torch.randint(0, num_semantic_classes, (num_classes,)).tolist()
                
        # Get the actual column names if available
        global semantic_data_column_names
        
        # Prepare token counts and indices in advance
        # Use more tokens per class to maximize information utilization
        # We'll use between 25% and 50% of all available tokens per class
        min_ratio, max_ratio = 0.25, 0.5
        min_tokens = max(5, int(semantic_data.shape[1] * min_ratio))  # At least 5 tokens
        max_tokens = min(int(semantic_data.shape[1] * max_ratio), semantic_data.shape[1])
        num_signature_tokens = torch.randint(min_tokens, max_tokens + 1, (num_classes,)).tolist()
        
        memory_logger.debug(f"Using {min_tokens}-{max_tokens} tokens per class out of {semantic_data.shape[1]} total tokens")
        
        # Create all token patterns in one batch
        for class_idx in range(num_classes):
            # Get the semantic class for this class
            semantic_class = selected_indices[class_idx]
            
            # Generate token indices
            token_indices = random.sample(range(semantic_data.shape[1]), num_signature_tokens[class_idx])
            
            # Extract the tokens for this class
            signature_tokens = semantic_data[semantic_class, token_indices].to(device)
            
            # Create descriptive name for class
            class_name = f"Class_{class_idx}_Type_{semantic_class}"
            
            # Get column name if available
            column_name = None
            if 'semantic_data_column_names' in globals() and semantic_data_column_names is not None:
                if len(semantic_data_column_names) > semantic_class:
                    column_name = semantic_data_column_names[semantic_class]
            
            # Store the class token pattern
            class_token_patterns[class_idx] = {
                'tokens': signature_tokens,
                'semantic_class': semantic_class,
                'token_indices': token_indices,
                'class_name': class_name,
                'column_name': column_name
            }
        
        # Only log details if debug is enabled
        if memory_logger.isEnabledFor(logging.DEBUG):
            memory_logger.debug(f"Created patterns for {num_classes} classes using {num_semantic_classes} semantic classes")
            # Log a few samples for debugging
            for class_idx in range(min(3, num_classes)):
                pattern = class_token_patterns[class_idx]
                memory_logger.debug(f"Class {class_idx} mapped to semantic class {pattern['semantic_class']}")
                
        return class_token_patterns
    
    def create_semantic_targets(self, y, class_token_patterns):
        """
        Create semantic target tensor based on class labels.
        This will be used for the self-supervised learning objective.
        
        Parameters:
        -----------
        y : torch.Tensor
            Class labels tensor of shape [samples, batch_size] or [samples, batch_size, 1]
        class_token_patterns : dict
            Mapping from class indices to token patterns
            
        Returns:
        --------
        torch.Tensor
            Semantic target tensor of shape [samples, batch_size] containing integer target indices
        """
        # First, log the input shape to understand exactly what's coming in
        if memory_logger.isEnabledFor(logging.DEBUG):
            memory_logger.debug(f"y tensor shape in create_semantic_targets: {y.shape}, dtype: {y.dtype}")
        
        # Handle 3D input by squeezing the last dimension if needed
        if y.dim() == 3 and y.shape[2] == 1:
            y = y.squeeze(-1)
            if memory_logger.isEnabledFor(logging.DEBUG):
                memory_logger.debug(f"Squeezed y tensor to shape: {y.shape}")
            
        # Initialize targets with ignore index (-100) with the same shape as y
        semantic_targets = torch.full_like(y, -100, dtype=torch.long)  # Explicitly use long type
            
        # Vectorized operations for processing
        # Get all the class indices once by converting to integer tensor
        class_indices = y.to(torch.int64)
        
        # Create a validity mask for indices within valid range
        valid_indices_mask = (class_indices >= 0) & (class_indices < len(class_token_patterns))
        
        # Track class-to-semantic mapping for logging
        class_to_semantic_map = {}
        
        # Get the flattened valid indices for faster processing
        valid_y_indices = torch.nonzero(valid_indices_mask)
        valid_class_indices = class_indices[valid_indices_mask]
        
        # Process all valid indices in a vectorized way
        # Log shapes in debug mode for understanding the structure
        if memory_logger.isEnabledFor(logging.DEBUG):
            memory_logger.debug(f"Shape of valid_y_indices: {valid_y_indices.shape}")
            memory_logger.debug(f"Shape of valid_class_indices: {valid_class_indices.shape}")
        
        # Create a lookup tensor mapping class_idx to semantic_class 
        lookup_size = max(valid_class_indices.max().item() + 1, len(class_token_patterns))
        class_to_semantic_lookup = torch.full((lookup_size,), -1, device=y.device, dtype=torch.long)
        
        # Fill the lookup table with semantic classes from class_token_patterns
        for class_idx in range(len(class_token_patterns)):
            semantic_class = class_token_patterns[class_idx]['semantic_class']
            class_to_semantic_lookup[class_idx] = semantic_class
            # Track in mapping for logging
            class_to_semantic_map[class_idx] = semantic_class
            
        # Get semantic classes for all valid class indices at once
        semantic_classes = class_to_semantic_lookup[valid_class_indices]
        
        # Use direct advanced indexing to set all values at once
        # valid_y_indices has shape [N, 3] where the first two dimensions are what we need
        # (Each entry has 3 coords because y is a 3D tensor, but we only need i,b coordinates)
        row_indices = valid_y_indices[:, 0]  
        col_indices = valid_y_indices[:, 1]
        
        # Apply all updates at once using advanced indexing
        semantic_targets[row_indices, col_indices] = semantic_classes
            
        # For debug/logging purposes only if memory_logger.isEnabledFor(logging.DEBUG)
        if memory_logger.isEnabledFor(logging.DEBUG):
            memory_logger.debug(f"=== create_semantic_targets ===")
            memory_logger.debug(f"Input y shape: {y.shape}, dtype: {y.dtype}, device: {y.device}")
            memory_logger.debug(f"Class token patterns: {len(class_token_patterns)} classes")
            
            # Log class distribution
            try:
                # Get valid counts more efficiently
                valid_count = valid_indices_mask.sum().item()
                invalid_count = y.numel() - valid_count
                memory_logger.debug(f"Class to semantic class mapping: {class_to_semantic_map}")
                memory_logger.debug(f"Set {valid_count} valid semantic targets, skipped {invalid_count} invalid positions")
                
                # Check for unassigned positions
                ignored_count = (semantic_targets == -100).sum().item()
                ignored_percentage = (ignored_count / semantic_targets.numel()) * 100
                memory_logger.debug(f"Ignored positions (-100): {ignored_count}/{semantic_targets.numel()} ({ignored_percentage:.1f}%)")
            except Exception as e:
                memory_logger.debug(f"Error in semantic targets logging: {e}")
        
        return semantic_targets
    
    def _apply_semantic_prior(self, x, semantic_features, device, y=None):
        """
        Apply semantic prior to the features, ensuring consistent 
        relationships between semantic features and class labels.
        Uses a batched approach to efficiently apply tokens based on class groups.
        
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
        global semantic_data, semantic_data_column_names
        
        # Get the number of classes from config
        num_classes = max(2, self.h['num_classes'])
        
        # If y is not provided (initial call), generate proxy targets
        if y is None:
            # Create a temporary random target for initial feature generation
            y = torch.randint(0, num_classes, (x.shape[0], x.shape[1]), device=device).float()
        
        # Load new semantic data if needed with proper number of classes
        seed = self.h.get('random_seed', None)
        if semantic_data.shape[0] != num_classes or not hasattr(globals(), 'semantic_data_column_names'):
            # Get random semantic data with proper randomization and caching
            from ticl.datasets.semantic_prior_data_loader import get_random_semantic_data
            semantic_data, semantic_data_column_names = get_random_semantic_data(
                num_classes=num_classes,
                seed=seed,
                use_cache=True
            )
            
        # Make sure semantic_data is on the correct device
        semantic_data = semantic_data.to(device)
        num_semantic_classes = semantic_data.shape[0]
        
        # Create class-token mapping if not already created
        if not hasattr(self, 'class_token_patterns') or len(self.class_token_patterns) != num_classes:
            self.class_token_patterns = self.create_semantic_class_mapping(num_classes, semantic_data, device)
        
        # Create semantic targets for training
        semantic_targets = self.create_semantic_targets(y, self.class_token_patterns)
        
        # Basic dimensions
        batch_size = x.shape[1]
        sample_size = x.shape[0]
        n_semantic_features = len(semantic_features)
        
        # Control randomness for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed) 
            np.random.seed(seed)
        
        # Convert y to integer class indices for easier processing
        y_int = y.to(torch.int64)
        
        # Dictionary to track causal features for each batch
        causal_features_map = {}
        
        # Initialize tensor to hold semantic features (zeros by default)
        semantic_feature_tensor = torch.zeros(
            sample_size, batch_size, n_semantic_features, device=device)
        
        # Pre-compute token pools for all semantic classes - fully vectorized approach
        token_pools = {}
        for class_idx in range(num_semantic_classes):
            tokens = semantic_data[class_idx]
            # Direct tensor conversion is much faster than list comprehension with .item() calls
            token_values = tokens.to(dtype=semantic_feature_tensor.dtype, device=device)
            token_pools[class_idx] = token_values
            
        # Pre-extract tokens for class patterns - fully vectorized approach
        class_token_pools = {}
        for class_idx in range(num_classes):
            if class_idx in self.class_token_patterns:
                pattern = self.class_token_patterns[class_idx]
                tokens = pattern['tokens']
                # Direct tensor conversion without list comprehension
                token_values = tokens.to(dtype=semantic_feature_tensor.dtype, device=device)
                class_token_pools[class_idx] = {
                    'tokens': token_values,
                    'semantic_class': pattern['semantic_class']
                }
        
        # Process each batch separately
        for b in range(batch_size):
            # 1. Determine which features will be causal for this batch
            # Increase causal feature ratios to use more semantic features effectively
            min_causal_ratio, max_causal_ratio = 0.4, 0.7  # Increased from 0.3-0.6
            num_causal = max(
                int(n_semantic_features * min_causal_ratio),
                min(int(n_semantic_features * max_causal_ratio), 1)
            )
            
            # Select random features to be causal (using tensor operations)
            causal_indices = torch.randperm(n_semantic_features)[:num_causal].tolist()
            causal_features = [semantic_features[i] for i in causal_indices]
            causal_features_map[b] = causal_features
            
            # 2. Create boolean mask for causal features
            is_causal = torch.zeros(n_semantic_features, dtype=torch.bool, device=device)
            is_causal[causal_indices] = True
            
            # 3. Get class assignments for this batch
            batch_classes = y_int[:, b].flatten()
            
            # 4. Create fill probability constants
            # Different probabilities for causal vs non-causal features
            # Higher fill rates to maximize information usage
            causal_fill_prob = 0.95  # 95% fill rate for causal features (increased from 80%)
            noncausal_fill_prob = 0.7  # 70% fill rate for non-causal features (increased from 50%)
            
            # 5. Process each class group together in a batched manner
            for class_idx in range(num_classes):
                # Get mask for all samples of this class in the batch
                class_mask = (batch_classes == class_idx)
                
                # Skip if no samples of this class
                if not class_mask.any():
                    continue
                
                # Get indices of samples belonging to this class
                class_positions = torch.nonzero(class_mask).flatten()
                
                # Get token patterns for this class using our precomputed pools
                if class_idx in class_token_pools:
                    class_pattern = class_token_pools[class_idx]
                    class_tokens = class_pattern['tokens']
                    semantic_class = class_pattern['semantic_class']
                    token_count = len(class_tokens)
                else:
                    # If missing class pattern, use random tokens
                    class_tokens = token_pools[0]
                    semantic_class = 0
                    token_count = len(class_tokens)
                
                # 6. Process features for this class (both causal and non-causal)
                for feat_idx in range(n_semantic_features):
                    # Determine if this feature is causal
                    feature_is_causal = is_causal[feat_idx]
                    
                    # Different fill probabilities based on causality
                    fill_prob = causal_fill_prob if feature_is_causal else noncausal_fill_prob
                    
                    # Generate random mask determining which positions to fill
                    should_fill = torch.rand(len(class_positions), device=device) < fill_prob
                    
                    # Skip if no positions to fill for this feature
                    if not should_fill.any():
                        continue
                    
                    # Get positions to fill
                    fill_positions = class_positions[should_fill]
                    
                    # For causal features, use tokens from this class
                    if feature_is_causal:
                        # Generate random token indices for each position
                        # (One random index per position to fill)
                        token_indices = torch.randint(0, token_count, (len(fill_positions),), device=device)
                        
                        # Extract the tokens directly from our precomputed pool
                        tokens = class_tokens[token_indices]
                        
                        # Fill the selected positions with tokens
                        semantic_feature_tensor[fill_positions, b, feat_idx] = tokens
                    else:
                        # For non-causal features, use fully vectorized batched token assignment
                        
                        # Generate random semantic classes for all positions at once
                        random_classes = torch.randint(0, num_semantic_classes, (len(fill_positions),), device=device)
                        
                        # Create a tensor of all possible token pools concatenated
                        # First, stack all token pools into one tensor
                        token_pool_sizes = [len(token_pools[c]) for c in range(num_semantic_classes)]
                        max_pool_size = max(token_pool_sizes)
                        
                        # Pre-select random indices for all positions at once
                        # For each position, generate a random index in the range of its class's token pool
                        random_indices = torch.zeros(len(fill_positions), dtype=torch.long, device=device)
                        
                        # Generate random token indices and gather tokens in a vectorized way
                        token_values = torch.zeros(len(fill_positions), device=device, dtype=semantic_feature_tensor.dtype)
                        
                        # Create one-hot encoding of random classes for vectorized indexing
                        # Each row is a one-hot vector for a class, size [num_positions, num_classes]
                        one_hot = torch.zeros(len(random_classes), num_semantic_classes, device=device)
                        one_hot.scatter_(1, random_classes.unsqueeze(1), 1)
                        
                        # Create a tensor with token pool sizes for each class
                        pool_sizes = torch.tensor([len(token_pools[c]) for c in range(num_semantic_classes)], 
                                                device=device)
                        
                        # Generate a tensor of random indices for all positions at once
                        # For each position, we need a random index within its class's pool size
                        # First, create a mask capping the maximum random value by class
                        max_indices = torch.matmul(one_hot, pool_sizes.float()).to(torch.long)
                        
                        # Generate random numbers for all positions at once, between 0 and the corresponding pool size
                        rand_values = torch.rand(len(random_classes), device=device)
                        # Scale the random values to the appropriate range for each class
                        rand_indices = (rand_values * max_indices).to(torch.long)
                        
                        # Now use these indices to efficiently select tokens from each pool
                        # We'll create a tensor that holds all class tokens in order
                        # and use advanced indexing to select the right tokens for each position
                        
                        # Create offsets tensor to adjust indices for each class's starting position in the flattened array
                        offsets = torch.zeros(num_semantic_classes, device=device, dtype=torch.long)
                        cumulative_size = 0
                        
                        # Vector to hold all tokens from all pools
                        all_tokens = []
                        
                        # Fill the all_tokens list and calculate offsets
                        for class_idx in range(num_semantic_classes):
                            # Track offset for this class
                            offsets[class_idx] = cumulative_size
                            
                            # Add all tokens from this pool to the flattened vector
                            class_tokens = token_pools[class_idx]
                            all_tokens.append(class_tokens)
                            
                            # Update cumulative size
                            cumulative_size += len(class_tokens)
                            
                        # Concatenate all tokens into one tensor
                        all_tokens_tensor = torch.cat(all_tokens)
                        
                        # Calculate the actual indices into the flattened token array
                        # For each position, add its class offset to its random index
                        class_offsets = torch.matmul(one_hot, offsets.float()).to(torch.long)
                        final_indices = rand_indices + class_offsets
                        
                        # Select all tokens at once from the flattened array
                        token_values = all_tokens_tensor[final_indices]
                        
                        # Assign all tokens at once
                        semantic_feature_tensor[fill_positions, b, feat_idx] = token_values
        
        # Transfer the semantic features to the original tensor
        for i, feat in enumerate(semantic_features):
            x[:, :, feat] = semantic_feature_tensor[:, :, i]
        
        # Create info dictionary with semantic information
        semantic_info = {
            'class_token_patterns': self.class_token_patterns,
            'semantic_targets': semantic_targets,
            'causal_features_map': causal_features_map
        }
        
        # Only run detailed diagnostic checks if debug logging is enabled
        if memory_logger.isEnabledFor(logging.DEBUG):
            memory_logger.debug(f"=== _apply_semantic_prior ===")
            # Check for zeros and non-zeros in semantic features
            semantic_feature_view = x[:, :, semantic_features]
            non_zeros = (semantic_feature_view != 0).sum().item()
            total_elements = semantic_feature_view.numel()
            non_zero_percentage = (non_zeros / total_elements) * 100
            memory_logger.debug(f"Semantic features filled: {non_zeros}/{total_elements} ({non_zero_percentage:.1f}% non-zero)")
            
            # Get information about valid targets
            valid_targets = (semantic_targets != -100)
            valid_count = valid_targets.sum().item()
            valid_ratio = valid_count / semantic_targets.numel() * 100
            memory_logger.debug(f"Valid semantic targets: {valid_count}/{semantic_targets.numel()} ({valid_ratio:.1f}%)")
        
        return x, semantic_info

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
                #never less than two, high prob. of 10, low prob. of 100+
                num_unique_features = max(round(random.gammavariate(1, 10)), 2)
                #Randomly select feature relationship to targets
                q = random.random()
                if q < 0.5:
                    boundary_type = 'monotonic'
                else:
                    options = [
                        'polynomial',
                        'periodic', 
                        'clustered', 
                        'information_theoretic',
                    ]
                    boundary_type = options[random.randint(0, len(options)-1)]  
                m = MulticlassRank(
                    num_unique_features,
                    boundary_type,
                    ordered_p=0.3
                )
                #around 50% of the features become categorical in categorical datasets
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
            memory_logger.debug(f"Created semantic padding: shape={semantic_padding.shape}, device={semantic_padding.device}")
            
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
            
            # Store semantic information in the info dictionary
            memory_logger.debug(f"Updating info dictionary with semantic info: {list(semantic_info.keys())}")
            info.update(semantic_info)
            
            # Verify semantic targets
            if 'semantic_targets' in semantic_info:
                semantic_targets = semantic_info['semantic_targets']
                valid_targets = (semantic_targets != -100).sum().item()
                total_targets = semantic_targets.numel()
                valid_percentage = (valid_targets / total_targets) * 100
                memory_logger.debug(f"Valid semantic targets: {valid_targets}/{total_targets} ({valid_percentage:.1f}%)")
                
                # Log a sample of the semantic targets
                if semantic_targets.numel() > 0:
                    memory_logger.debug(f"Semantic targets sample (first 5x5 values):")
                    for i in range(min(5, semantic_targets.shape[0])):
                        row_values = semantic_targets[i, :min(5, semantic_targets.shape[1])].tolist()
                        memory_logger.debug(f"  Row {i}: {row_values}")
            
            memory_logger.debug(f"=== Semantic features added successfully ===")
        else:
            memory_logger.debug(f"Skipping semantic features (random probability {semantic_feature_p} not triggered)")
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
