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
semantic_data = get_random_semantic_data()


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
        memory_logger.debug(f"create_semantic_class_mapping - Creating patterns for {num_classes} classes using {num_semantic_classes} semantic classes")
        
        # Ensure each class gets a different semantic class if possible
        available_semantic_classes = list(range(num_semantic_classes))
        if num_classes <= num_semantic_classes:
            selected_semantic_classes = random.sample(available_semantic_classes, num_classes)
        else:
            # If more classes than semantic classes, we'll have some duplicates
            selected_semantic_classes = [random.choice(available_semantic_classes) for _ in range(num_classes)]
        
        memory_logger.debug(f"Selected semantic classes: {selected_semantic_classes[:5]}...")
        
        for class_idx in range(num_classes):
            # Select a semantic class (row) from our semantic data
            semantic_class = selected_semantic_classes[class_idx]
            
            # Select a subset of tokens to represent this class (5-10 tokens)
            num_signature_tokens = random.randint(5, min(10, semantic_data.shape[1]))
            token_indices = random.sample(range(semantic_data.shape[1]), num_signature_tokens)
            signature_tokens = semantic_data[semantic_class, token_indices].to(device)
            
            # Create descriptive name for class
            class_name = f"Class_{class_idx}_Type_{semantic_class}"
            
            class_token_patterns[class_idx] = {
                'tokens': signature_tokens,
                'semantic_class': semantic_class,
                'token_indices': token_indices,
                'class_name': class_name
            }
            
            if class_idx < 3:  # Log a few for debugging
                memory_logger.debug(f"Class {class_idx} mapped to semantic class {semantic_class} with {num_signature_tokens} tokens")
                memory_logger.debug(f"  Token indices: {token_indices[:5]}...")
                memory_logger.debug(f"  Token values: {signature_tokens[:5].cpu().tolist()}...")
        
        memory_logger.debug(f"Created class token patterns for {len(class_token_patterns)} classes")
        return class_token_patterns
    
    def create_semantic_targets(self, y, class_token_patterns):
        """
        Create semantic target tensor based on class labels.
        This will be used for the self-supervised learning objective.
        
        Parameters:
        -----------
        y : torch.Tensor
            Class labels tensor of shape [samples, batch_size]
        class_token_patterns : dict
            Mapping from class indices to token patterns
            
        Returns:
        --------
        torch.Tensor
            Semantic target tensor of shape [samples, batch_size] containing integer target indices
        """
        memory_logger.debug(f"=== create_semantic_targets START ===")
        memory_logger.debug(f"Input y shape: {y.shape}, dtype: {y.dtype}, device: {y.device}")
        memory_logger.debug(f"Class token patterns: {len(class_token_patterns)} classes")
        
        # Sample values from y for debugging
        memory_logger.debug(f"Y sample values (first few positions):")
        for i in range(min(3, y.shape[0])):
            for b in range(min(3, y.shape[1])):
                memory_logger.debug(f"  y[{i},{b}] = {y[i, b].item()}")
        
        # Log class distribution
        try:
            unique_classes, counts = torch.unique(y, return_counts=True)
            class_dist = {int(cls.item()): int(count.item()) for cls, count in zip(unique_classes, counts)}
            memory_logger.debug(f"Class distribution in y: {class_dist}")
        except Exception as e:
            memory_logger.debug(f"Could not compute class distribution: {e}")
        
        # Initialize targets with ignore index (-100)
        semantic_targets = torch.full_like(y, -100, dtype=torch.long)  # Explicitly use long type
        memory_logger.debug(f"Created semantic_targets tensor with shape {semantic_targets.shape}, dtype {semantic_targets.dtype}")
        
        # Track class-to-semantic mapping
        class_to_semantic_map = {}
        invalid_count = 0
        valid_count = 0
        
        # For each sample, set the target based on the class
        for i in range(y.shape[0]):
            for b in range(y.shape[1]):
                class_idx = int(y[i, b].item())
                
                # Skip if invalid class
                if class_idx < 0 or class_idx >= len(class_token_patterns):
                    invalid_count += 1
                    # Only log some instances to avoid spam
                    if invalid_count < 10:
                        memory_logger.debug(f"  Skipping invalid class_idx {class_idx} at position [{i},{b}]")
                    continue
                
                # Get the semantic class for this class
                semantic_class = class_token_patterns[class_idx]['semantic_class']
                
                # Track in mapping
                if class_idx not in class_to_semantic_map:
                    class_to_semantic_map[class_idx] = semantic_class
                
                # Set the semantic target to the semantic class (ensure integer)
                semantic_targets[i, b] = int(semantic_class)
                valid_count += 1
                
                # Log samples for debugging
                if i < 3 and b < 3:
                    memory_logger.debug(f"  Mapping class {class_idx} → semantic class {semantic_class} at position [{i},{b}]")
        
        memory_logger.debug(f"Class to semantic class mapping: {class_to_semantic_map}")
        memory_logger.debug(f"Set {valid_count} valid semantic targets, skipped {invalid_count} invalid positions")
        
        # Double-check that we're returning integer tensor
        if not semantic_targets.dtype == torch.long:
            memory_logger.warning(f"Semantic targets have incorrect dtype: {semantic_targets.dtype}. Converting to long.")
            semantic_targets = semantic_targets.long()
        
        # Log distribution of semantic targets
        try:
            valid_mask = semantic_targets != -100
            semantic_classes, counts = torch.unique(semantic_targets[valid_mask], return_counts=True)
            sem_dist = {int(cls.item()): int(count.item()) for cls, count in zip(semantic_classes, counts)}
            memory_logger.debug(f"Semantic target class distribution: {sem_dist}")
        except Exception as e:
            memory_logger.debug(f"Could not compute semantic target distribution: {e}")
        
        # Check for unassigned positions
        ignored_count = (semantic_targets == -100).sum().item()
        ignored_percentage = (ignored_count / semantic_targets.numel()) * 100
        memory_logger.debug(f"Ignored positions (-100): {ignored_count}/{semantic_targets.numel()} ({ignored_percentage:.1f}%)")
        
        memory_logger.debug(f"=== create_semantic_targets END ===")
        return semantic_targets
    
    def _apply_semantic_prior(self, x, semantic_features, device, y=None):
        """
        Apply semantic prior to the features, ensuring consistent 
        relationships between semantic features and class labels.
        With a guaranteed minimum of causal features per class.
        
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
        # Use the semantic prior data from the top of the file
        # which is loaded from semantic_prior_data_loader
        global semantic_data
        
        memory_logger.debug(f"=== _apply_semantic_prior START ===")
        memory_logger.debug(f"Input tensor x: shape={x.shape}, device={x.device}, dtype={x.dtype}")
        memory_logger.debug(f"Semantic features: {len(semantic_features)} features, first few: {semantic_features[:5]}")
        memory_logger.debug(f"Device: {device}")
        
        # Get the number of classes from the configuration or y tensor
        if y is not None:
            memory_logger.debug(f"Target tensor y: shape={y.shape}, device={y.device}, dtype={y.dtype}")
            # Log class distribution
            try:
                classes, counts = torch.unique(y, return_counts=True)
                class_dist = {int(cls.item()): int(count.item()) for cls, count in zip(classes, counts)}
                memory_logger.debug(f"Class distribution: {class_dist}")
                
                # Get number of unique classes from y
                num_classes = len(classes)
                memory_logger.debug(f"Detected {num_classes} classes from target tensor")
            except Exception as e:
                memory_logger.debug(f"Could not compute class distribution: {e}")
                num_classes = max(2, self.h['num_classes'])
        else:
            memory_logger.debug(f"No target tensor provided, will generate proxy targets")
            num_classes = max(2, self.h['num_classes'])
        
        # Ensure we have at least 2 classes (binary classification minimum)
        num_classes = max(2, num_classes)
        memory_logger.debug(f"Using {num_classes} classes for semantic features")
        
        # Load new semantic data if needed (with the correct number of classes)
        from ticl.datasets.semantic_prior_data_loader import get_random_semantic_data
        if semantic_data.shape[0] != num_classes:
            memory_logger.debug(f"Existing semantic data has {semantic_data.shape[0]} classes, but we need {num_classes} classes")
            memory_logger.debug(f"Loading new semantic data with {num_classes} classes")
            semantic_data = get_random_semantic_data(num_classes=num_classes)
        
        # Get semantic data info
        memory_logger.debug(f"Semantic data tensor: shape={semantic_data.shape}, device={semantic_data.device}, dtype={semantic_data.dtype}")
        
        # Make sure semantic_data is on the correct device
        semantic_data = semantic_data.to(device)
        memory_logger.debug(f"Moved semantic data to {device}")
        
        # Number of semantic classes (first dimension of semantic_data)
        num_semantic_classes = semantic_data.shape[0]
        memory_logger.debug(f"Number of semantic classes: {num_semantic_classes}")
        
        # Sample and log a few semantic data points
        memory_logger.debug(f"Semantic data samples:")
        for i in range(min(3, num_semantic_classes)):
            memory_logger.debug(f"  Class {i} sample: {semantic_data[i, :10]}")
        
        # If y is not provided (initial call), generate proxy targets
        if y is None:
            # Create a temporary random target for initial feature generation
            memory_logger.debug(f"Generating proxy targets with {num_classes} classes")
            y = torch.randint(0, num_classes, 
                             (x.shape[0], x.shape[1]), device=device).float()
            memory_logger.debug(f"Generated proxy targets: shape={y.shape}, min={y.min().item()}, max={y.max().item()}")
        
        # Create class-token mapping if not already created
        if not hasattr(self, 'class_token_patterns') or len(self.class_token_patterns) != num_classes:
            memory_logger.debug(f"Creating class-token mapping for {num_classes} classes")
            self.class_token_patterns = self.create_semantic_class_mapping(num_classes, semantic_data, device)
            memory_logger.debug(f"Class token patterns created for {len(self.class_token_patterns)} classes")
            
            # Log a sample of the class-token patterns
            sample_class = 0
            if sample_class in self.class_token_patterns:
                pattern = self.class_token_patterns[sample_class]
                memory_logger.debug(f"Sample class {sample_class} token pattern:")
                memory_logger.debug(f"  Semantic class: {pattern['semantic_class']}")
                memory_logger.debug(f"  Token indices: {pattern['token_indices'][:5]}...")
                memory_logger.debug(f"  Tokens sample: {pattern['tokens'][:5]}...")
        else:
            memory_logger.debug(f"Using existing class-token patterns for {len(self.class_token_patterns)} classes")
        
        # Create semantic targets for training
        memory_logger.debug(f"Creating semantic targets from class labels")
        semantic_targets = self.create_semantic_targets(y, self.class_token_patterns)
        memory_logger.debug(f"Semantic targets: shape={semantic_targets.shape}, dtype={semantic_targets.dtype}")
        
        # Log distribution of semantic targets
        valid_targets = (semantic_targets != -100)
        valid_count = valid_targets.sum().item()
        valid_ratio = valid_count / semantic_targets.numel() * 100
        memory_logger.debug(f"Valid semantic targets: {valid_count}/{semantic_targets.numel()} ({valid_ratio:.1f}%)")
        
        # Check distribution of semantic target values
        if valid_count > 0:
            valid_values = semantic_targets[valid_targets]
            try:
                classes, counts = torch.unique(valid_values, return_counts=True)
                semantic_dist = {int(cls.item()): int(count.item()) for cls, count in zip(classes, counts)}
                memory_logger.debug(f"Semantic target distribution: {semantic_dist}")
            except Exception as e:
                memory_logger.debug(f"Could not compute semantic target distribution: {e}")
        
        # Track which features are causal for each batch item
        causal_features_map = {b: [] for b in range(x.shape[1])}
        
        # Minimum proportion of semantic features that should be causal
        min_causal_ratio = 0.3
        max_causal_ratio = 0.6
        memory_logger.debug(f"Causal feature ratio range: {min_causal_ratio} to {max_causal_ratio}")
        
        # For each batch item
        for b in range(x.shape[1]):
            # Determine how many features should be causal (at least 30%, at most 60%)
            num_causal = max(
                int(len(semantic_features) * min_causal_ratio),
                min(int(len(semantic_features) * max_causal_ratio), 1)
            )
            
            # Randomly select causal features
            causal_indices = random.sample(range(len(semantic_features)), num_causal)
            causal_features = [semantic_features[i] for i in causal_indices]
            causal_features_map[b] = causal_features
            
            if b < 2:  # Only log for first few batch items to avoid too much output
                memory_logger.debug(f"Batch item {b}: {num_causal}/{len(semantic_features)} causal features")
                memory_logger.debug(f"  Causal feature indices: {causal_features[:5]}...")
            
            # For all semantic features
            for feat_idx, feat in enumerate(semantic_features):
                is_causal = feat in causal_features
                
                if is_causal:
                    # This feature will be causally related to the class
                    
                    # Set zero probability (between 0.1 and 0.3 for causal features)
                    # Less zeros for causal features to ensure stronger signal
                    zero_prob = random.randrange(1, 3) / 10
                    
                    if b < 2 and feat_idx < 5:  # Log only for first few features of first few batches
                        memory_logger.debug(f"  Causal feature {feat}: zero_prob={zero_prob}")
                    
                    # For each sample position
                    for pos in range(x.shape[0]):
                        # Get the class for this sample
                        class_idx = int(y[pos, b].item())
                        
                        # Skip if invalid class
                        if class_idx < 0 or class_idx >= len(self.class_token_patterns):
                            continue
                        
                        # Get tokens specific to this class
                        class_tokens = self.class_token_patterns[class_idx]['tokens']
                        
                        # Randomly decide whether to use a class-specific token or zero
                        if random.random() > zero_prob:
                            # Use a token from the class pattern
                            token_idx = random.randint(0, len(class_tokens) - 1)
                            token_value = class_tokens[token_idx].item()
                            x[pos, b, feat] = token_value
                            
                            # Log sample token assignments for debugging
                            if b < 2 and feat_idx < 5 and pos < 3:  # Only log few samples
                                memory_logger.debug(f"    Sample {pos}, class {class_idx}, assigned token: {token_value}")
                else:
                    # Non-causal feature - just random tokens with zeros
                    # Set zero probability (between 0.2 and 0.7)
                    zero_prob = random.randrange(2, 7) / 10
                    
                    # Get random semantic class
                    semantic_class = random.randint(0, num_semantic_classes - 1)
                    tokens = semantic_data[semantic_class]
                    
                    if b < 2 and feat_idx < 5:  # Log only for first few features of first few batches
                        memory_logger.debug(f"  Non-causal feature {feat}: zero_prob={zero_prob}, random semantic class={semantic_class}")
                    
                    # For each sample position
                    for pos in range(x.shape[0]):
                        # Randomly decide whether to use a token or zero
                        if random.random() < zero_prob:
                            # Keep as zero (already initialized to zero)
                            pass
                        else:
                            # Randomly select a token from the available tokens
                            token_pos = random.randint(0, len(tokens) - 1)
                            token_value = tokens[token_pos].item()
                            x[pos, b, feat] = token_value
                            
                            # Log sample token assignments for debugging
                            if b < 2 and feat_idx < 5 and pos < 3:  # Only log few samples
                                memory_logger.debug(f"    Sample {pos}, random token: {token_value}")
        
        # Create info dictionary with semantic information
        semantic_info = {
            'class_token_patterns': self.class_token_patterns,
            'semantic_targets': semantic_targets,
            'causal_features_map': causal_features_map
        }
        
        # Log final tensor stats
        memory_logger.debug(f"Final tensor x after semantic feature injection:")
        memory_logger.debug(f"  Shape: {x.shape}, device: {x.device}, dtype: {x.dtype}")
        
        # Check for zeros and non-zeros in semantic features
        semantic_feature_tensor = x[:, :, semantic_features]
        non_zeros = (semantic_feature_tensor != 0).sum().item()
        total_elements = semantic_feature_tensor.numel()
        non_zero_percentage = (non_zeros / total_elements) * 100
        memory_logger.debug(f"  Semantic features filled: {non_zeros}/{total_elements} ({non_zero_percentage:.1f}% non-zero)")
        
        # Check how many tokens were actually used
        try:
            unique_tokens = torch.unique(semantic_feature_tensor)
            memory_logger.debug(f"  Unique tokens used: {len(unique_tokens)}")
            if len(unique_tokens) < 20:  # If few enough to list
                memory_logger.debug(f"  Token values: {unique_tokens.tolist()}")
        except Exception as e:
            memory_logger.debug(f"Error analyzing unique tokens: {e}")
        
        memory_logger.debug(f"=== _apply_semantic_prior END ===")
        
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
