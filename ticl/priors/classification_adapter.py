import random
import numpy as np
import torch

from ticl.utils import (get_nan_value, normalize_by_used_features_f, normalize_data,
                             remove_outliers)

from ticl.distributions import sample_distributions, uniform_int_sampler_f, parse_distributions, safe_randint
from ticl.priors.boundaries import *
from ticl.datasets.semantic_prior_data_loader import get_random_semantic_data
from .utils import CategoricalActivation, randomize_classes

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
        
        # Ensure each class gets a different semantic class if possible
        available_semantic_classes = list(range(num_semantic_classes))
        if num_classes <= num_semantic_classes:
            selected_semantic_classes = random.sample(available_semantic_classes, num_classes)
        else:
            # If more classes than semantic classes, we'll have some duplicates
            selected_semantic_classes = [random.choice(available_semantic_classes) for _ in range(num_classes)]
        
        for class_idx in range(num_classes):
            # Select a semantic class (row) from our semantic data
            semantic_class = selected_semantic_classes[class_idx]
            
            # Select a subset of tokens to represent this class (5-10 tokens)
            num_signature_tokens = random.randint(5, min(10, semantic_data.shape[1]))
            token_indices = random.sample(range(semantic_data.shape[1]), num_signature_tokens)
            signature_tokens = semantic_data[semantic_class, token_indices].to(device)
            
            class_token_patterns[class_idx] = {
                'tokens': signature_tokens,
                'semantic_class': semantic_class,
                'token_indices': token_indices
            }
        
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
            Semantic target tensor of shape [samples, batch_size]
        """
        # Initialize targets with ignore index (-100)
        semantic_targets = torch.full_like(y, -100)
        
        # For each sample, set the target based on the class
        for i in range(y.shape[0]):
            for b in range(y.shape[1]):
                class_idx = int(y[i, b].item())
                
                # Skip if invalid class
                if class_idx < 0 or class_idx >= len(class_token_patterns):
                    continue
                
                # Get the semantic class for this class
                semantic_class = class_token_patterns[class_idx]['semantic_class']
                
                # Set the semantic target to the semantic class
                semantic_targets[i, b] = semantic_class
        
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
        # Import the semantic prior data
        from ticl.datasets.semantic_prior_data_sample import random_tensor as semantic_data
        
        # Make sure semantic_data is on the correct device
        semantic_data = semantic_data.to(device)
        
        # Number of semantic classes (first dimension of semantic_data)
        num_semantic_classes = semantic_data.shape[0]
        
        # If y is not provided (initial call), generate proxy targets
        if y is None:
            # Create a temporary random target for initial feature generation
            y = torch.randint(0, max(2, self.h['num_classes']), 
                             (x.shape[0], x.shape[1]), device=device).float()
        
        # Create class-token mapping if not already created
        if not hasattr(self, 'class_token_patterns'):
            num_classes = max(2, self.h['num_classes'])
            self.class_token_patterns = self.create_semantic_class_mapping(num_classes, semantic_data, device)
        
        # Create semantic targets for training
        semantic_targets = self.create_semantic_targets(y, self.class_token_patterns)
        
        # Track which features are causal for each batch item
        causal_features_map = {b: [] for b in range(x.shape[1])}
        
        # Minimum proportion of semantic features that should be causal
        min_causal_ratio = 0.3
        max_causal_ratio = 0.6
        
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
            
            # For all semantic features
            for feat_idx, feat in enumerate(semantic_features):
                is_causal = feat in causal_features
                
                if is_causal:
                    # This feature will be causally related to the class
                    
                    # Set zero probability (between 0.1 and 0.3 for causal features)
                    # Less zeros for causal features to ensure stronger signal
                    zero_prob = random.randrange(1, 3) / 10
                    
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
                            x[pos, b, feat] = class_tokens[token_idx]
                else:
                    # Non-causal feature - just random tokens with zeros
                    # Set zero probability (between 0.2 and 0.7)
                    zero_prob = random.randrange(2, 7) / 10
                    
                    # Get random semantic class
                    semantic_class = random.randint(0, num_semantic_classes - 1)
                    tokens = semantic_data[semantic_class]
                    
                    # For each sample position
                    for pos in range(x.shape[0]):
                        # Randomly decide whether to use a token or zero
                        if random.random() < zero_prob:
                            # Keep as zero (already initialized to zero)
                            pass
                        else:
                            # Randomly select a token from the available tokens
                            token_pos = random.randint(0, len(tokens) - 1)
                            x[pos, b, feat] = tokens[token_pos]
        
        # Create info dictionary with semantic information
        semantic_info = {
            'class_token_patterns': self.class_token_patterns,
            'semantic_targets': semantic_targets,
            'causal_features_map': causal_features_map
        }
        
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
            num_semantic_features = 50
            info['num_semantic_features'] = num_semantic_features
            
            # Add the semantic features (initialized to 0.0)
            semantic_padding = torch.zeros((x.shape[0], x.shape[1], num_semantic_features), device=device)
            x = torch.cat([x, semantic_padding], dim=2)
            
            # Record indices of semantic features
            semantic_features = list(range(x.shape[2] - num_semantic_features, x.shape[2]))
            
            # Apply semantic prior to the features with class-based consistency
            x, semantic_info = self._apply_semantic_prior(x, semantic_features, device, y)
            
            # Store semantic information in the info dictionary
            info.update(semantic_info)
        
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
