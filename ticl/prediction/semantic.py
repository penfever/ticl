import numpy as np
import torch
import random
import itertools
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array
from sklearn.utils import column_or_1d
from sklearn.utils.multiclass import check_classification_targets
import pandas as pd
from transformers import CLIPTokenizerFast

from ticl.utils import log_gpu_memory
from ticl.datasets.labeled_numeric_prior_data_loader import labeled_numeric_data, column_metadata


class SemanticAwareClassifierWrapper(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible wrapper for the SemanticAwareClassifier model.
    Uses an approach similar to TabPFNClassifier but adapted for the semantic-aware architecture.
    """

    def __init__(self, device='cpu', model=None, config=None, 
                 batch_size=32, verbose=False, 
                 N_ensemble_configurations=3,
                 seed=0,
                 semantic_column_indices=None,
                 semantic_class_descriptions=None,
                 feature_stats=None,
                 statistical_class_terms=None,
                 reserved_feature_indices=None,
                 table_metadata=None,
                 column_names=None):
        """
        Initialize the classifier wrapper for semantic-aware models.
        
        Parameters:
        -----------
        device : str
            Device to run the model on ('cpu', 'cuda', 'mps')
        model : nn.Module
            The pre-loaded semantic-aware model
        config : dict
            Configuration dictionary for the model
        batch_size : int
            Batch size for inference
        verbose : bool
            Whether to print verbose output
        N_ensemble_configurations : int
            Number of different ensemble views to use for prediction
        seed : int
            Random seed for reproducibility
        semantic_column_indices : list of int, optional
            Indices of semantic columns in the input data
        semantic_class_descriptions : dict or list, optional
            Class descriptions for semantic matching
        feature_stats : dict, optional
            Statistical information about numeric features
        statistical_class_terms : dict, optional
            Statistical terms describing each class
        reserved_feature_indices : list, optional
            Indices of reserved semantic features (always the last 3 semantic features)
        table_metadata : dict or str, optional
            Metadata about the table or problem domain
        column_names : list, optional
            Actual column names for the input data (for semantic feature enhancment)
        """
        # Set fitted flag to False initially
        self.is_fitted_ = False
        if model is None:
            raise ValueError("Model must be provided")
        if config is None:
            raise ValueError("Config must be provided")
            
        self.device = device
        self.model = model
        self.config = config
        self.batch_size = batch_size
        self.verbose = verbose
        self.N_ensemble_configurations = N_ensemble_configurations
        self.seed = seed
        self.semantic_column_indices = semantic_column_indices
        self.semantic_class_descriptions = semantic_class_descriptions
        
        # Add new statistical information parameters
        self.feature_stats = feature_stats
        self.statistical_class_terms = statistical_class_terms
        self.reserved_feature_indices = reserved_feature_indices
        self.table_metadata = table_metadata
        self.column_names = column_names
        
        # Get model's capabilities from config
        if "prior" in config:
            self.max_num_features = config['prior']['num_features']
            # If we have semantic features, we need to account for them
            if 'num_features_with_semantic' in config['prior']:
                self.max_num_features_with_semantic = config['prior']['num_features_with_semantic']
                if self.verbose:
                    print(f"Using max features with semantic: {self.max_num_features_with_semantic}")
            else:
                self.max_num_features_with_semantic = self.max_num_features
                
            if 'classification' in config['prior']:
                self.max_num_classes = config['prior']['classification']['max_num_classes']
            else:
                self.max_num_classes = 2
        else:
            self.max_num_features = config.get('num_features', 100)
            self.max_num_features_with_semantic = config.get('num_features_with_semantic', 
                                                           self.max_num_features)
            self.max_num_classes = config.get('max_num_classes', 2)
            
        # Check if model has semantic column metadata
        if hasattr(model, 'semantic_column_metadata') and model.semantic_column_metadata:
            if self.verbose:
                print(f"Using semantic column metadata from model with {len(model.semantic_column_metadata)} columns")
            self.semantic_column_metadata = model.semantic_column_metadata
            
        # Set model to eval mode during init
        self.model.eval()
        
        # Get default semantic feature probability from config
        if 'transformer' in config:
            self.semantic_feature_p = config['transformer'].get('semantic_feature_p', 0.0)
        elif 'linear_attention' in config:
            self.semantic_feature_p = config['linear_attention'].get('semantic_feature_p', 0.0)
        else:
            self.semantic_feature_p = 0.0
            
        # Log initialization
        if self.verbose:
            print(f"Initialized SemanticAwareClassifierWrapper with device={device}")
            print(f"Model type: {self.model.__class__.__name__}")
            print(f"Max features: {self.max_num_features}, Max features with semantic: {self.max_num_features_with_semantic}")
            print(f"Max classes: {self.max_num_classes}")
            print(f"Using {N_ensemble_configurations} ensemble configurations")
            print(f"Semantic feature probability: {self.semantic_feature_p}")
            
            # Log the expected output dimension of the model for debugging
            try:
                if hasattr(self.model, 'base_model') and hasattr(self.model.base_model, 'encoder'):
                    encoder_shape = self.model.base_model.encoder.weight.shape
                    print(f"Model encoder shape: {encoder_shape}")
            except Exception as e:
                print(f"Could not determine model encoder shape: {e}")
        
    def fit(self, X, y, semantic_column_indices=None, X_semantic_text=None, overwrite_warning=False):
        """
        Fit the classifier to the training data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        semantic_column_indices : list of int, optional
            Indices of columns that are semantic (text-based) features
        X_semantic_text : array-like or dict, optional
            Semantic text data corresponding to semantic columns
        overwrite_warning : bool, optional
            Whether to ignore warnings about dataset size limitations
            
        Returns:
        --------
        self : object
            Returns self
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self._validate_targets(y)
        
        # Keep track of semantic column indices if provided
        if semantic_column_indices is not None:
            self.semantic_column_indices = semantic_column_indices
            self.has_semantic_data = True
        else:
            # If not provided, assume no semantic features in X
            self.has_semantic_data = False
            
        # Store semantic text data if provided
        if X_semantic_text is not None:
            self.X_semantic_text = X_semantic_text
            self.has_semantic_data = True
            
        # Process input dimensions
        if X.shape[1] > self.max_num_features_with_semantic:
            if self.verbose:
                print(f"Warning: Input has {X.shape[1]} features, but model only supports {self.max_num_features_with_semantic}")
                
            # Determine which features to keep
            if self.semantic_column_indices:
                # Prioritize semantic features if we have any
                semantic_indices = set(self.semantic_column_indices)
                remaining_indices = set(range(X.shape[1])) - semantic_indices
                
                # Calculate how many non-semantic features we can keep
                remaining_slots = self.max_num_features_with_semantic - len(semantic_indices)
                
                if remaining_slots > 0:
                    # Set random seed for reproducibility
                    np.random.seed(self.seed)
                    # Choose from remaining indices
                    remaining_selected = sorted(np.random.choice(
                        list(remaining_indices), 
                        min(remaining_slots, len(remaining_indices)), 
                        replace=False
                    ))
                    # Combine selected indices
                    self.feature_indices = sorted(list(semantic_indices) + remaining_selected)
                else:
                    # Only room for some or all semantic features
                    self.feature_indices = sorted(list(semantic_indices))[:self.max_num_features_with_semantic]
            else:
                # No semantic features, just select random features
                np.random.seed(self.seed)
                self.feature_indices = sorted(np.random.choice(
                    X.shape[1], 
                    self.max_num_features_with_semantic, 
                    replace=False
                ))
                
            # Apply feature selection
            X = X[:, self.feature_indices]
            
            # Update semantic column indices if they exist
            if self.semantic_column_indices:
                # Create mapping from old indices to new positions
                old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(self.feature_indices)}
                # Update semantic column indices
                self.semantic_column_indices = [old_to_new[idx] for idx in self.semantic_column_indices 
                                             if idx in old_to_new]
                
        # Check class count
        if len(self.classes_) > self.max_num_classes:
            if self.verbose:
                print(f"Warning: Input has {len(self.classes_)} classes, but model only supports {self.max_num_classes}")
                print(f"Only using the first {self.max_num_classes} classes")
                
            # Get the most frequent classes
            unique_classes, counts = np.unique(y, return_counts=True)
            class_order = np.argsort(-counts)  # Sort by descending count
            top_classes = unique_classes[class_order[:self.max_num_classes]]
            
            # Create mask for examples belonging to top classes
            mask = np.isin(y, top_classes)
            
            # Filter X and y
            X = X[mask]
            y = y[mask]
            
            # Update classes
            self.classes_ = np.array(top_classes)
            
            # Re-encode labels to be consecutive integers
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
            self.label_map_ = dict(zip(range(len(label_encoder.classes_)), label_encoder.classes_))
        
        # Convert to PyTorch tensors
        if self.device == 'mps':
            # MPS doesn't support float64, explicitly convert to float32
            self.X_ = torch.tensor(X, dtype=torch.float32, device=self.device)
        else:
            self.X_ = torch.tensor(X, device=self.device).float()
            
        self.y_ = torch.tensor(y, dtype=torch.long, device=self.device)
        
        # Set model to eval mode
        self.model.eval()
        
        # Mark as fitted
        self.is_fitted_ = True
        
        return self
    
    def _validate_targets(self, y):
        """
        Validate and format target values y
        
        Parameters:
        -----------
        y : array-like
            Target values
            
        Returns:
        --------
        array
            Validated target values
        """
        y_ = column_or_1d(y, warn=True)
        check_classification_targets(y)
        cls, y = np.unique(y_, return_inverse=True)
        if len(cls) < 2:
            raise ValueError(
                "The number of classes has to be greater than one; got %d class"
                % len(cls)
            )
        
        self.classes_ = cls
        return np.asarray(y, dtype=np.float64, order="C")

    def get_ensemble_configurations(self):
        """
        Generate ensemble configurations for prediction.
        
        Returns:
        --------
        list
            List of ensemble configuration tuples
        """
        # Set random seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # Create different feature configurations
        # 1. Feature permutations 
        feature_configurations = [np.random.permutation(self.X_.shape[1]) for _ in range(min(5, self.N_ensemble_configurations))]
        
        # 2. Semantic variations if we have semantic columns
        semantic_configurations = []
        if self.semantic_column_indices and hasattr(self, 'has_semantic_data') and self.has_semantic_data:
            # Create different ways to sample semantic text
            semantic_configurations = [f"semantic_sample_{i}" for i in range(3)]
        
        # 3. Standardization variations
        preprocess_configurations = ["none", "standard", "robust"]
        
        # 4. Class permutations
        label_configurations = [np.random.permutation(len(self.classes_)) for _ in range(min(3, self.N_ensemble_configurations))]
        
        # Combine configurations
        all_configs = list(itertools.product(
            feature_configurations, 
            semantic_configurations if semantic_configurations else [None],
            preprocess_configurations,
            label_configurations
        ))
        
        # Shuffle and select up to N_ensemble_configurations
        random.shuffle(all_configs)
        return all_configs[:self.N_ensemble_configurations]
        
    def predict_proba(self, X, class_descriptions=None, normalize_with_test=False, return_logits=False):
        """
        Predict class probabilities for the input samples X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test samples
        class_descriptions : dict, optional
            Mapping of class labels to text descriptions
        normalize_with_test : bool, optional
            Whether to include test data when normalizing features
        return_logits : bool, optional
            Whether to return logits instead of probabilities
            
        Returns:
        --------
        y_proba : array-like of shape (n_samples, n_classes)
            Class probabilities for each sample
        """
        # Check if fit has been called
        check_is_fitted(self)
        
        # Input validation
        X = check_array(X)
        
        # Apply feature selection if needed
        if hasattr(self, 'feature_indices') and self.feature_indices is not None:
            X = X[:, self.feature_indices]
            
        # Check feature count alignment
        if self.X_.shape[1] != X.shape[1]:
            # Adjust X dimensions to match training data
            if X.shape[1] < self.X_.shape[1]:
                # Pad with zeros
                X_padded = np.zeros((X.shape[0], self.X_.shape[1]))
                X_padded[:, :X.shape[1]] = X
                X = X_padded
            else:
                # Truncate
                X = X[:, :self.X_.shape[1]]
                
        # Convert to torch tensor
        if self.device == 'mps':
            # MPS doesn't support float64, explicitly convert to float32
            X_test = torch.tensor(X, dtype=torch.float32, device=self.device)
        else:
            X_test = torch.tensor(X, device=self.device).float()
            
        # Combine training and test data for in-context learning
        X_full = torch.cat([self.X_, X_test], dim=0).unsqueeze(1)
        
        # Create zeros tensor for test labels
        test_zeros = torch.zeros(len(X_test), dtype=torch.long, device=self.device)
        
        # Combine train and test labels
        y_full = torch.cat([self.y_, test_zeros], dim=0).unsqueeze(1)
        
        # Get number of classes
        num_classes = len(torch.unique(self.y_))
        
        # Generate ensemble configurations
        ensemble_configs = self.get_ensemble_configurations()
        
        # Create a helper function to track and process semantic columns
        categorical_feats = []
        if self.semantic_column_indices:
            categorical_feats = self.semantic_column_indices
            
        # Store evaluation position (separate train from test)
        eval_pos = self.X_.shape[0]
        
        # Set random seed for reproducible predictions
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            
        # Initialize output storage
        output = None
        eval_xs_transformed = {}
        
        # Process each ensemble configuration
        for ensemble_idx, config in enumerate(ensemble_configs):
            # Unpack configuration
            feature_perm, semantic_config, preprocess_type, label_perm = config
            
            # Create a copy of input data for this configuration
            X_ensemble = X_full.clone()
            y_ensemble = y_full.clone()
            
            # Preprocess features
            if preprocess_type in eval_xs_transformed:
                # Reuse already preprocessed data
                X_processed = eval_xs_transformed[preprocess_type].clone()
            else:
                # Apply preprocessing
                X_processed = self._preprocess_input(
                    X_ensemble,
                    y_ensemble,
                    preprocess_transform=preprocess_type,
                    max_features=self.max_num_features_with_semantic,
                    normalize_with_test=normalize_with_test,
                    eval_position=eval_pos,
                    categorical_feats=categorical_feats,
                    device=self.device,
                    scale=True,
                    normalize_by_used_features=True
                )
                # Store for potential reuse
                eval_xs_transformed[preprocess_type] = X_processed
                
            # Check if we need to extend features to match model expectations
            expected_features = self.max_num_features_with_semantic if hasattr(self, 'max_num_features_with_semantic') else self.max_num_features
            if X_processed.shape[2] < expected_features:
                if self.verbose:
                    print(f"Padding features from {X_processed.shape[2]} to {expected_features}")
                # Extend with zeros to expected feature count
                padding = torch.zeros(
                    (X_processed.shape[0], X_processed.shape[1], expected_features - X_processed.shape[2]),
                    device=X_processed.device,
                    dtype=X_processed.dtype
                )
                X_processed = torch.cat([X_processed, padding], dim=2)
                
                # Update categorical features indices if needed
                if categorical_feats:
                    # No need to update indices as we're adding to the end
                    pass
                
            # Apply feature permutation if needed
            if len(feature_perm) > 1:  # Only permute if we have a meaningful permutation
                # Compute feature permutation
                n_features = X_processed.shape[2]
                # Create a permutation that respects categorical features
                if categorical_feats:
                    # Keep semantic columns in place, permute others
                    feature_indices = list(range(n_features))
                    semantic_set = set(categorical_feats)
                    non_semantic = [i for i in feature_indices if i not in semantic_set]
                    
                    # Permute non-semantic features
                    perm_indices = feature_perm % len(non_semantic)  # Ensure we don't go out of bounds
                    perm_non_semantic = [non_semantic[i] for i in perm_indices[:len(non_semantic)]]
                    
                    # Create full permutation
                    full_perm = list(categorical_feats) + perm_non_semantic
                else:
                    # Permute all features
                    perm_indices = feature_perm % n_features  # Ensure we don't go out of bounds
                    full_perm = [int(i) for i in perm_indices[:n_features]]
                    
                # Apply permutation
                if len(full_perm) > 0:  # Only apply if we have a valid permutation
                    # Handle case where full_perm might not cover all features
                    if len(full_perm) < X_processed.shape[2]:
                        # Include remaining features
                        remaining = list(set(range(X_processed.shape[2])) - set(full_perm))
                        full_perm = full_perm + remaining
                    # Apply permutation
                    X_processed = X_processed[:, :, full_perm]
                
            # Apply class label permutation if needed
            if len(label_perm) > 1:  # Only permute if we have a meaningful permutation
                # Permute class labels for training portion
                y_train = y_ensemble[:eval_pos].cpu()
                y_train_permuted = y_train.clone()
                
                # Apply permutation safely
                for i in range(len(y_train)):
                    orig_label = int(y_train[i].item())
                    if orig_label < len(label_perm):
                        y_train_permuted[i] = int(label_perm[orig_label])
                
                # Convert back to device
                y_train_permuted = y_train_permuted.to(self.device)
                
                # Keep test labels as zeros
                y_ensemble = torch.cat([y_train_permuted, y_ensemble[eval_pos:]], dim=0)
            
            # Forward pass through model
            with torch.no_grad():
                # Get model predictions
                outputs = self.model((X_processed, y_ensemble.float()), single_eval_pos=eval_pos)
                
                # Keep only outputs for classes we care about
                outputs = outputs[:, :, 0:num_classes]
                
                # Apply temperature scaling
                outputs = outputs / torch.exp(torch.tensor([0.8], device=self.device).log())
                
                # Convert to probabilities if not returning logits
                if not return_logits:
                    outputs = torch.nn.functional.softmax(outputs, dim=-1)
                    
                # Reverse label permutation if applied
                if len(label_perm) > 1:
                    # Create inverse permutation mapping
                    inv_perm = torch.zeros(len(label_perm), dtype=torch.long, device=self.device)
                    for i, p in enumerate(label_perm):
                        if p < len(inv_perm):
                            inv_perm[p] = i
                    
                    # Apply inverse permutation to outputs
                    permuted_outputs = outputs.clone()
                    for i in range(len(label_perm)):
                        if i < outputs.shape[2] and inv_perm[i] < outputs.shape[2]:
                            permuted_outputs[:, :, inv_perm[i]] = outputs[:, :, i]
                    
                    outputs = permuted_outputs
                    
                # Add to ensemble output
                if output is None:
                    output = outputs
                else:
                    output = output + outputs
        
        # Average ensemble outputs
        output = output / len(ensemble_configs)
        
        # Apply softmax if averaging logits
        if return_logits is False and len(ensemble_configs) > 1:
            output = torch.nn.functional.softmax(output, dim=-1)
            
        # Transpose to get [batch, samples, classes]
        output = torch.transpose(output, 0, 1)
        
        # Return numpy array
        return output.squeeze(0).cpu().numpy()
            
    def _preprocess_input(self, eval_xs, eval_ys, preprocess_transform, max_features, 
                         normalize_with_test, eval_position, categorical_feats, 
                         device=None, scale=True, normalize_by_used_features=True):
        """
        Preprocess input features using TabPFN approach but with semantic awareness
        
        Parameters:
        -----------
        eval_xs : torch.Tensor
            Input features
        eval_ys : torch.Tensor
            Target labels for training portion
        preprocess_transform : str
            Type of preprocessing to apply
        max_features : int
            Maximum number of features
        normalize_with_test : bool
            Whether to include test data when normalizing
        eval_position : int
            Position that separates training and test data
        categorical_feats : list
            Indices of categorical features (including semantic columns)
        device : str
            Device to use for computation
        scale : bool
            Whether to apply feature scaling
        normalize_by_used_features : bool
            Whether to adjust for number of features used
            
        Returns:
        --------
        torch.Tensor
            Preprocessed input features
        """
        import warnings
        from ticl.utils import normalize_data, remove_outliers, normalize_by_used_features_f
        from sklearn.preprocessing import PowerTransformer, QuantileTransformer, RobustScaler
        import numpy as np
        
        # Use provided device or default to the instance device
        device = device or self.device
        
        # Check batch dimension
        if eval_xs.shape[1] > 1:
            raise Exception("Transforms only allow one batch dim")
        
        # Add reserved feature indices to categorical features if they exist
        if self.reserved_feature_indices:
            # Make a copy of categorical_feats to avoid modifying the original
            if categorical_feats:
                categorical_feats = list(set(categorical_feats).union(set(self.reserved_feature_indices)))
            else:
                categorical_feats = self.reserved_feature_indices
            
        # Handle max feature count
        if eval_xs.shape[2] > max_features:
            # Randomly select max_features columns, preserving semantic columns if possible
            if categorical_feats:
                # Ensure semantic columns are included in selection
                semantic_indices = set(categorical_feats)
                remaining_indices = set(range(eval_xs.shape[2])) - semantic_indices
                
                # Fill remaining slots with random features
                remaining_slots = max_features - len(semantic_indices)
                if remaining_slots > 0 and len(remaining_indices) > 0:
                    # Set random seed for reproducibility if specified
                    if self.seed is not None:
                        np.random.seed(self.seed)
                    
                    # Choose from remaining indices
                    remaining_selected = sorted(np.random.choice(
                        list(remaining_indices), 
                        min(remaining_slots, len(remaining_indices)), 
                        replace=False
                    ))
                    # Combine selected indices
                    selected_indices = sorted(list(semantic_indices) + remaining_selected)
                else:
                    # Only have room for some semantic columns - prioritize reserved features if they exist
                    if self.reserved_feature_indices:
                        # Ensure reserved features are included first
                        priority_features = set(self.reserved_feature_indices)
                        remaining_semantic = list(semantic_indices - priority_features)
                        
                        # Calculate how many remaining slots we have after including reserved features
                        remaining_slots = max_features - len(priority_features)
                        if remaining_slots > 0:
                            # Use remaining slots for other semantic features
                            selected_semantic = remaining_semantic[:remaining_slots]
                            selected_indices = sorted(list(priority_features) + selected_semantic)
                        else:
                            # Only include reserved features
                            selected_indices = sorted(list(priority_features)[:max_features])
                    else:
                        # No reserved features, just take the first max_features semantic columns
                        selected_indices = sorted(list(semantic_indices)[:max_features])
            else:
                # No semantic columns, just select random features
                # Set random seed for reproducibility if specified
                if self.seed is not None:
                    np.random.seed(self.seed)
                    
                selected_indices = sorted(np.random.choice(eval_xs.shape[2], max_features, replace=False))
                
            # Apply selection
            eval_xs = eval_xs[:, :, selected_indices]
                
            # Update categorical_feats indices to match the new tensor dimensions
            if categorical_feats:
                # Create mapping from old indices to new positions
                index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_indices)}
                
                # Map the categorical feature indices to their new positions
                categorical_feats = [index_map[feat] for feat in categorical_feats if feat in index_map]
                
                if self.verbose and hasattr(self, 'semantic_column_indices') and self.semantic_column_indices:
                    # Check how many of the original semantic indices were preserved
                    preserved = [i for i in self.semantic_column_indices if i in selected_indices]
                
                # If we have reserved features, make sure they're updated in the object
                if self.reserved_feature_indices:
                    self.reserved_feature_indices = [index_map[feat] for feat in self.reserved_feature_indices 
                                                    if feat in index_map]
            
            # Check if we need to fill reserved features with statistical information
            if self.reserved_feature_indices and (self.feature_stats or self.statistical_class_terms):
                # We need to add statistical information to the reserved features
                self._fill_reserved_features_with_stats(eval_xs, device)
                
            # Fill all available semantic features with numerical feature semanticizations
            # If we have column names, we can use them to create better semanticizations
            if self.semantic_column_indices and len(self.semantic_column_indices) > (len(self.reserved_feature_indices or [])):
                self._fill_semantic_features_with_numeric_semanticization(eval_xs, device, categorical_feats)
        
        # Initialize preprocessing transformer based on type
        if preprocess_transform != 'none':
            if preprocess_transform == 'power' or preprocess_transform == 'power_all':
                pt = PowerTransformer(standardize=True)
            elif preprocess_transform == 'quantile' or preprocess_transform == 'quantile_all':
                pt = QuantileTransformer(output_distribution='normal')
            elif preprocess_transform == 'robust' or preprocess_transform == 'robust_all':
                pt = RobustScaler(unit_variance=True)
        
        # Apply normalization if scaling is enabled
        if scale:
            eval_xs = normalize_data(eval_xs, normalize_positions=-1 if normalize_with_test else eval_position)
        else:
            eval_xs = torch.clip(eval_xs, min=-100, max=100)
        
        # Remove batch dimension for column processing
        eval_xs = eval_xs[:, 0, :]
        
        # Filter empty or constant columns
        def check_col_values(col_tensor):
            # Check if this is a torch tensor
            if torch.is_tensor(col_tensor):
                # Use torch.isnan() for torch tensors
                nan_mask = torch.isnan(col_tensor)
                # Use ~ to invert the mask (get non-NaN values)
                return len(torch.unique(col_tensor[~nan_mask])) > 1
            else:
                # Fallback for numpy arrays or pandas Series
                import numpy as np
                nan_mask = np.isnan(col_tensor)
                return len(np.unique(col_tensor[~nan_mask])) > 1
            
        # Always include semantic columns
        include_mask = torch.zeros(eval_xs.shape[1], dtype=torch.bool, device=eval_xs.device)
        
        # Check which columns have variance
        for col in range(eval_xs.shape[1]):
            if col in categorical_feats:
                # Always include semantic columns
                include_mask[col] = True
            else:
                # Only include if it has variance
                include_mask[col] = check_col_values(eval_xs[0:eval_ys.shape[0], col])
                
        # Keep only non-empty columns
        eval_xs = eval_xs[:, include_mask]
        
        # Update categorical feature indices after filtering
        if categorical_feats:
            new_categorical_feats = []
            current_pos = 0
            for col in range(len(include_mask)):
                if include_mask[col]:
                    if col in categorical_feats:
                        new_categorical_feats.append(current_pos)
                    current_pos += 1
            categorical_feats = new_categorical_feats
            
            # Update reserved feature indices if they exist
            if self.reserved_feature_indices:
                # Create mapping from old indices to new positions after filtering
                old_to_new = {}
                new_pos = 0
                for old_pos, keep in enumerate(include_mask):
                    if keep:
                        old_to_new[old_pos] = new_pos
                        new_pos += 1
                        
                # Update reserved feature indices
                self.reserved_feature_indices = [old_to_new.get(idx, -1) for idx in self.reserved_feature_indices]
                # Filter out any that didn't make it through filtering
                self.reserved_feature_indices = [idx for idx in self.reserved_feature_indices if idx >= 0]
        
        # Apply feature transformation
        warnings.simplefilter('error')
        if preprocess_transform != 'none':
            # Convert to numpy for sklearn transformers
            eval_xs = eval_xs.cpu().numpy()
            
            # Determine which features to transform
            feats = set(range(eval_xs.shape[1])) if 'all' in preprocess_transform else \
                   set(range(eval_xs.shape[1])) - set(categorical_feats)
                   
            # Transform each column independently
            for col in feats:
                try:
                    with warnings.catch_warnings():
                        # Fit transformer on training data
                        pt.fit(eval_xs[0:eval_position, col:col + 1])
                        trans = pt.transform(eval_xs[:, col:col + 1])
                except KeyboardInterrupt:
                    raise KeyboardInterrupt
                except RuntimeWarning:
                    if self.verbose:
                        print('Power transform is not working, switching to robust transform...')
                    # Fallback to robust scaling
                    pt = RobustScaler(unit_variance=True)
                    pt.fit(eval_xs[0:eval_position, col:col + 1])
                    trans = pt.transform(eval_xs[:, col:col + 1])
                
                # Apply transformation
                eval_xs[:, col:col + 1] = trans
                
            # Convert back to torch tensor with proper dtype for device
            if device == 'mps':
                # MPS doesn't support float64, explicitly convert to float32
                eval_xs = torch.tensor(eval_xs, dtype=torch.float32, device=device)
            else:
                eval_xs = torch.tensor(eval_xs, device=device).float()
            
        # Reset warning filter
        warnings.simplefilter('default')
        
        # Restore batch dimension
        eval_xs = eval_xs.unsqueeze(1)
        
        # Remove outliers
        eval_xs = remove_outliers(eval_xs, normalize_positions=-1 if normalize_with_test else eval_position)
        
        # Rescale features if requested
        if normalize_by_used_features:
            eval_xs = normalize_by_used_features_f(eval_xs, eval_xs.shape[-1], max_features)
            
        return eval_xs.to(device)
        
    def _fill_reserved_features_with_stats(self, x, device):
        """
        Fill reserved semantic features with statistical information at inference time.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor with shape [samples, batch, features]
        device : str or torch.device
            Device to use for computation
            
        Returns:
        --------
        None (modifies x in-place)
        """
        if not self.reserved_feature_indices or (not self.feature_stats and not self.statistical_class_terms):
            # Nothing to do
            return
            
        try:
            # Initialize tokenizer
            tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
            
            # Feature 1: Row-wise semanticized representation of first numeric feature
            if len(self.reserved_feature_indices) >= 1 and self.feature_stats:
                feature_idx = self.reserved_feature_indices[0]
                
                # Find numerical features to semanticize
                numeric_features = []
                numeric_feature_stats = {}
                
                # Identify numeric features and gather their statistics
                for batch_idx, batch_stats in self.feature_stats.items():
                    for feat_idx, stats in batch_stats.items():
                        if stats.get('type') == 'numerical':
                            if feat_idx not in numeric_features:
                                numeric_features.append(feat_idx)
                                numeric_feature_stats[feat_idx] = stats
                
                # Ensure we found some numeric features
                if numeric_features:
                    # Sort to ensure consistent ordering
                    numeric_features.sort()
                    
                    # Select the first numeric feature to semanticize
                    first_numeric_feat = numeric_features[0]
                    stats = numeric_feature_stats.get(first_numeric_feat, {})
                    
                    # Get min/max/mean for normalization
                    feat_min = stats.get('min', 0)
                    feat_max = stats.get('max', 1)
                    feat_mean = stats.get('mean', (feat_min + feat_max) / 2)
                    
                    # Define thresholds for low/medium/high categorization
                    if 'quantiles' in stats:
                        quant_25 = stats['quantiles'].get('25%', feat_min + (feat_max - feat_min) * 0.25)
                        quant_75 = stats['quantiles'].get('75%', feat_min + (feat_max - feat_min) * 0.75)
                    else:
                        # Define even thresholds if no quantiles available
                        range_val = feat_max - feat_min
                        quant_25 = feat_min + range_val * 0.25
                        quant_75 = feat_min + range_val * 0.75
                    
                    # Process each row in the input tensor
                    for i in range(x.shape[0]):  # For each sample in batch
                        for j in range(x.shape[1]):  # For each batch
                            if first_numeric_feat < x.shape[2]:
                                # Get the actual value for this feature in this row
                                value = x[i, j, first_numeric_feat].item()
                                
                                # Determine if the value is low, medium, or high
                                token_value = 0  # Default token
                                if value <= quant_25:
                                    # Low value
                                    token_value = 10  # Arbitrary token for "low"
                                elif value <= quant_75:
                                    # Medium value
                                    token_value = 20  # Arbitrary token for "medium"
                                else:
                                    # High value
                                    token_value = 30  # Arbitrary token for "high"
                                
                                # Set the token value for this row
                                x[i, j, feature_idx] = token_value
                else:
                    # No numeric features found, use a placeholder
                    placeholder_text = "No numeric features to semanticize"
                    tokens = tokenizer(
                        placeholder_text, 
                        return_tensors="pt",
                        padding="max_length",
                        max_length=77,
                        truncation=True
                    ).input_ids[0].to(device)
                    
                    # Use a placeholder value for all rows
                    x[:, :, feature_idx] = tokens[0].to(x.dtype)
            
            # Feature 2: Row-wise semanticized representation of second numeric feature
            if len(self.reserved_feature_indices) >= 2 and self.feature_stats:
                feature_idx = self.reserved_feature_indices[1]
                
                # Find numerical features to semanticize (reuse from feature 1)
                numeric_features = []
                numeric_feature_stats = {}
                
                # Identify numeric features and gather their statistics
                for batch_idx, batch_stats in self.feature_stats.items():
                    for feat_idx, stats in batch_stats.items():
                        if stats.get('type') == 'numerical':
                            if feat_idx not in numeric_features:
                                numeric_features.append(feat_idx)
                                numeric_feature_stats[feat_idx] = stats
                
                # Ensure we found at least two numeric features
                if len(numeric_features) >= 2:
                    # Sort to ensure consistent ordering
                    numeric_features.sort()
                    
                    # Select the second numeric feature to semanticize
                    second_numeric_feat = numeric_features[1]
                    stats = numeric_feature_stats.get(second_numeric_feat, {})
                    
                    # Get min/max/mean for normalization
                    feat_min = stats.get('min', 0)
                    feat_max = stats.get('max', 1)
                    feat_mean = stats.get('mean', (feat_min + feat_max) / 2)
                    
                    # Define thresholds for low/medium/high categorization
                    if 'quantiles' in stats:
                        quant_25 = stats['quantiles'].get('25%', feat_min + (feat_max - feat_min) * 0.25)
                        quant_75 = stats['quantiles'].get('75%', feat_min + (feat_max - feat_min) * 0.75)
                    else:
                        # Define even thresholds if no quantiles available
                        range_val = feat_max - feat_min
                        quant_25 = feat_min + range_val * 0.25
                        quant_75 = feat_min + range_val * 0.75
                    
                    # Process each row in the input tensor
                    for i in range(x.shape[0]):  # For each sample in batch
                        for j in range(x.shape[1]):  # For each batch
                            if second_numeric_feat < x.shape[2]:
                                # Get the actual value for this feature in this row
                                value = x[i, j, second_numeric_feat].item()
                                
                                # Determine if the value is low, medium, or high
                                token_value = 0  # Default token
                                if value <= quant_25:
                                    # Low value
                                    token_value = 40  # Different token than feature 1
                                elif value <= quant_75:
                                    # Medium value
                                    token_value = 50  # Different token than feature 1
                                else:
                                    # High value
                                    token_value = 60  # Different token than feature 1
                                
                                # Set the token value for this row
                                x[i, j, feature_idx] = token_value
                elif len(numeric_features) == 1 and self.statistical_class_terms:
                    # Only have one numeric feature, use class information if available
                    
                    # Try to fill with class tokens based on statistical_class_terms
                    have_class_info = False
                    
                    # Check if we have classes to use
                    if hasattr(self, 'classes_'):
                        # We only have test data, so we can't know the class yet
                        # Just use a default token based on the number of classes
                        for i in range(x.shape[0]):  # For each sample in batch
                            for j in range(x.shape[1]):  # For each batch
                                # Use a default token value - we don't know the class yet
                                token_value = 70  # Base token for class info
                                x[i, j, feature_idx] = token_value
                        
                        have_class_info = True
                        
                    
                    # If we don't have class info, use a placeholder
                    if not have_class_info:
                        
                        placeholder_text = "No second feature or class info available"
                        tokens = tokenizer(
                            placeholder_text, 
                            return_tensors="pt",
                            padding="max_length",
                            max_length=77,
                            truncation=True
                        ).input_ids[0].to(device)
                        
                        # Use a placeholder value for all rows
                        x[:, :, feature_idx] = tokens[0].to(x.dtype)
                else:
                    
                    placeholder_text = "No second numeric feature to semanticize"
                    tokens = tokenizer(
                        placeholder_text, 
                        return_tensors="pt",
                        padding="max_length",
                        max_length=77,
                        truncation=True
                    ).input_ids[0].to(device)
                    
                    # Use a placeholder value for all rows
                    x[:, :, feature_idx] = tokens[0].to(x.dtype)
            
            # Feature 3: Table or problem domain metadata
            if len(self.reserved_feature_indices) >= 3:
                feature_idx = self.reserved_feature_indices[2]
                
                # Use provided metadata if available, otherwise use placeholder
                if self.table_metadata:
                    if isinstance(self.table_metadata, dict):
                        # Convert dictionary to text format
                        metadata_text = []
                        for key, value in self.table_metadata.items():
                            metadata_text.append(f"{key}: {value}")
                        metadata_str = ". ".join(metadata_text)
                    else:
                        # Assume it's already a string
                        metadata_str = str(self.table_metadata)
                else:
                    # Use placeholder if no metadata is provided
                    metadata_str = "Table metadata placeholder - no metadata provided at inference time"
                
                # Tokenize the metadata
                tokens = tokenizer(
                    metadata_str,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=77,
                    truncation=True
                ).input_ids[0].to(device)
                
                # Fill the third reserved feature with this information
                x[:, :, feature_idx] = tokens[0].to(x.dtype)
                
                    
            # Now fill any remaining semantic features (those that aren't reserved)
            self._fill_semantic_features_with_numeric_semanticization(x, device)
        
        except Exception as e:
            if self.verbose:
                print(f"Error filling reserved features with statistical information: {e}")
                import traceback
                traceback.print_exc()
    
    def _fill_semantic_features_with_numeric_semanticization(self, x, device, categorical_feats=None):
        """
        Fill all non-reserved semantic features with semanticized numeric features.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor with shape [samples, batch, features]
        device : str or torch.device
            Device to use for computation
        categorical_feats : list, optional
            List of categorical feature indices to exclude
            
        Returns:
        --------
        None (modifies x in-place)
        """
        if not self.semantic_column_indices:
            return
        
        # Get indices of semantic features that aren't reserved
        reserved_indices = set(self.reserved_feature_indices or [])
        semantic_indices = [idx for idx in self.semantic_column_indices if idx not in reserved_indices]
        
        if not semantic_indices:
            return
        
            
        try:
            # Initialize tokenizer
            tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
            
            # Find numeric columns in the input data
            if categorical_feats is None:
                categorical_feats = []
            
            # Identify numeric features (those not in categorical_feats)
            numeric_features = [i for i in range(x.shape[2]) if i not in categorical_feats and i not in semantic_indices]
            
            if not numeric_features:
                if self.verbose:
                    print("No numeric features found to semanticize")
                return
                
            # For each semantic feature, find a corresponding numeric feature to semanticize
            for i, sem_idx in enumerate(semantic_indices):
                # Select a numeric feature to semanticize
                if i < len(numeric_features):
                    num_idx = numeric_features[i]
                else:
                    # If we have more semantic features than numeric features, cycle through them
                    num_idx = numeric_features[i % len(numeric_features)]
                
                # Get column name if available, otherwise use a synthetic name
                column_name = None
                if self.column_names and num_idx < len(self.column_names):
                    column_name = self.column_names[num_idx]
                
                # If we don't have a real column name, use one from labeled_numeric_prior_data_loader
                if not column_name:
                    column_name = random.choice(list(labeled_numeric_data.keys()))
                
                # Get metadata for this column type
                col_metadata = column_metadata.get(column_name, {})
                col_description = col_metadata.get('description', f"Numeric feature {num_idx}")
                col_units = col_metadata.get('units', '')
                
                # Create a semantic tag for the feature
                if col_units:
                    semantic_tag = f"{column_name} ({col_description}, {col_units})"
                else:
                    semantic_tag = f"{column_name} ({col_description})"
                
                # Compute statistics for this numeric feature
                col_values = x[:, :, num_idx].flatten()
                valid_values = col_values[~torch.isnan(col_values)]
                
                if len(valid_values) > 0:
                    # Compute quantiles for classification
                    sorted_values, _ = torch.sort(valid_values)
                    q25_idx = max(0, min(int(len(sorted_values) * 0.25), len(sorted_values) - 1))
                    q75_idx = max(0, min(int(len(sorted_values) * 0.75), len(sorted_values) - 1))
                    
                    q25 = sorted_values[q25_idx].item()
                    q75 = sorted_values[q75_idx].item()
                    
                    # Process each row for semanticization
                    for i in range(x.shape[0]):
                        for j in range(x.shape[1]):
                            if num_idx < x.shape[2]:
                                value = x[i, j, num_idx].item()
                                
                                # Skip NaN values
                                if torch.isnan(torch.tensor(value)):
                                    continue
                                
                                # Determine the category (low, medium, high)
                                if value <= q25:
                                    category = "low"
                                    token_value = 10 + (i % 10)  # Vary slightly to prevent all rows having same token
                                elif value <= q75:
                                    category = "medium"
                                    token_value = 20 + (i % 10)
                                else:
                                    category = "high" 
                                    token_value = 30 + (i % 10)
                                
                                # Fill the semantic feature with the token value
                                x[i, j, sem_idx] = token_value
                    
                    
                else:
                    # No valid values, use a placeholder
                    
                    
                    # Tokenize the column name as a placeholder
                    tokens = tokenizer(
                        semantic_tag,
                        return_tensors="pt",
                        padding="max_length",
                        max_length=77,
                        truncation=True
                    ).input_ids[0].to(device)
                    
                    # Use a placeholder value for all rows
                    x[:, :, sem_idx] = tokens[0].to(x.dtype)
        
        except Exception as e:
            if self.verbose:
                print(f"Error filling semantic features with numeric semanticization: {e}")
                import traceback
                traceback.print_exc()
    
    def predict(self, X, class_descriptions=None):
        """
        Predict class labels for the input samples X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test samples
        class_descriptions : dict, optional
            Mapping of class labels to text descriptions
            
        Returns:
        --------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels
        """
        proba = self.predict_proba(X, class_descriptions=class_descriptions)
        y_pred = np.argmax(proba, axis=1)
        return self.classes_[y_pred]
    
    def predict_semantic(self, X, X_semantic_text=None, class_descriptions=None):
        """
        Predict class labels for the input samples X using semantic features.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test samples
        X_semantic_text : array-like or dict, optional
            Semantic text data for test samples
        class_descriptions : dict or list, optional
            Text descriptions for each class
            
        Returns:
        --------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels
        class_similarities : array-like of shape (n_samples, n_classes)
            Similarity scores between input and each class
        """
        # Check if fit has been called
        check_is_fitted(self)
        
        # Check if model has semantic capabilities
        if not hasattr(self.model, 'process_semantic_features'):
            raise NotImplementedError("The underlying model does not support semantic feature processing")
        
        # Input validation
        X = check_array(X)
        
        # Apply feature selection if needed
        if hasattr(self, 'feature_indices') and self.feature_indices is not None:
            X = X[:, self.feature_indices]
            
        # Check feature count alignment
        if self.X_.shape[1] != X.shape[1]:
            # Adjust X dimensions to match training data
            if X.shape[1] < self.X_.shape[1]:
                # Pad with zeros
                X_padded = np.zeros((X.shape[0], self.X_.shape[1]))
                X_padded[:, :X.shape[1]] = X
                X = X_padded
            else:
                # Truncate
                X = X[:, :self.X_.shape[1]]
            
        # Convert to torch tensor
        if self.device == 'mps':
            # MPS doesn't support float64, explicitly convert to float32
            X_test = torch.tensor(X, dtype=torch.float32, device=self.device)
        else:
            X_test = torch.tensor(X, device=self.device).float()
            
        # Combine training and test data for in-context learning
        X_full = torch.cat([self.X_, X_test], dim=0).unsqueeze(1)
        
        # Ensure we're using the right number of features
        expected_features = self.max_num_features_with_semantic if hasattr(self, 'max_num_features_with_semantic') else self.max_num_features
        if X_full.shape[2] < expected_features:
            if self.verbose:
                print(f"Padding features in semantic prediction from {X_full.shape[2]} to {expected_features}")
            # Pad to the model's expected feature count
            padding = torch.zeros(
                (X_full.shape[0], X_full.shape[1], expected_features - X_full.shape[2]),
                device=X_full.device,
                dtype=X_full.dtype
            )
            X_full = torch.cat([X_full, padding], dim=2)
        
        # Create zeros tensor for test labels
        test_zeros = torch.zeros(len(X_test), dtype=torch.long, device=self.device)
        
        # Combine train and test labels
        y_full = torch.cat([self.y_, test_zeros], dim=0).unsqueeze(1)
        
        # Process semantic text data if provided
        semantic_batch_info = None
        if X_semantic_text is not None:
            semantic_batch_info = self._prepare_semantic_batch_info(X_semantic_text, class_descriptions)
            
        # Process class descriptions if provided
        class_tokens = None
        if class_descriptions is not None:
            class_tokens = self._process_class_descriptions(class_descriptions)
            
        # Forward pass through model with semantic info
        with torch.no_grad():
            # Store evaluation position (separates train from test)
            eval_pos = self.X_.shape[0]
            
            # Get predictions with semantic information
            results = self.model(
                (X_full, y_full.float()), 
                single_eval_pos=eval_pos,
                semantic_batch_info=semantic_batch_info,
                class_tokens=class_tokens
            )
            
            # Extract semantic similarities if available
            if isinstance(results, dict) and 'class_semantic_similarities' in results:
                class_similarities = results['class_semantic_similarities'].cpu().numpy()
            else:
                # Generate dummy similarities based on logits
                logits = results[:, :, 0:len(self.classes_)]
                class_similarities = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()
                
            # Get class predictions
            pred_class_indices = np.argmax(class_similarities, axis=-1)
            pred_classes = self.classes_[pred_class_indices]
            
            # Return both predictions and similarity scores
            return pred_classes, class_similarities
            
    def _prepare_semantic_batch_info(self, X_semantic_text, class_descriptions=None):
        """
        Prepare semantic batch info from semantic text data
        
        Parameters:
        -----------
        X_semantic_text : array-like or dict
            Semantic text data for input samples
        class_descriptions : dict or list, optional
            Text descriptions for each class
            
        Returns:
        --------
        dict
            Batch info for semantic processing
        """
        # If no semantic text data, return None
        if X_semantic_text is None:
            return None
            
        # Basic structure for semantic batch info
        batch_info = {
            'semantic_targets': [],
            'class_token_patterns': []
        }
        
        # Process class descriptions if available
        if class_descriptions is not None:
            # Convert to standard format if needed
            if isinstance(class_descriptions, list):
                class_descriptions = {i: desc for i, desc in enumerate(class_descriptions)}
                
            # Create class token patterns
            for class_idx, description in class_descriptions.items():
                # Find corresponding numeric class if needed
                if isinstance(class_idx, str) and hasattr(self, 'classes_') and any(isinstance(c, str) for c in self.classes_):
                    # Try to find class index
                    try:
                        numeric_idx = np.where(self.classes_ == class_idx)[0][0]
                    except:
                        numeric_idx = -1  # Unknown class
                else:
                    numeric_idx = class_idx
                    
                # Create token pattern for this class
                token_pattern = {
                    'tokens': description,
                    'column_name': f"class_{numeric_idx}",
                    'class_name': str(class_idx),
                    'semantic_class': numeric_idx
                }
                
                batch_info['class_token_patterns'].append(token_pattern)
                
        # Process semantic text data
        if isinstance(X_semantic_text, dict):
            # Dictionary format with column names as keys
            for col_name, col_text in X_semantic_text.items():
                batch_info['semantic_targets'].append({
                    'column_name': col_name,
                    'text': col_text
                })
        elif isinstance(X_semantic_text, list):
            # List format with text entries
            for i, text in enumerate(X_semantic_text):
                batch_info['semantic_targets'].append({
                    'column_name': f"feature_{i}",
                    'text': text
                })
        else:
            # Unknown format, try to convert to string
            batch_info['semantic_targets'].append({
                'column_name': "text",
                'text': str(X_semantic_text)
            })
            
        return batch_info
        
    def _process_class_descriptions(self, class_descriptions):
        """
        Process class descriptions into tokens for model
        
        Parameters:
        -----------
        class_descriptions : dict or list
            Text descriptions for each class
            
        Returns:
        --------
        dict
            Processed tokens for each class
        """
        # Check if we need to process
        if class_descriptions is None:
            return None
            
        # Convert to standard format if needed
        if isinstance(class_descriptions, list):
            class_descriptions = {i: desc for i, desc in enumerate(class_descriptions)}
            
        # Initialize tokenizer
        try:
            from transformers import CLIPTokenizerFast
            tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
        except ImportError:
            if self.verbose:
                print("Could not import CLIPTokenizerFast, using raw text for class descriptions")
            # Return raw text without tokenization
            return class_descriptions
            
        # Process each class description
        class_tokens = {}
        for class_idx, description in class_descriptions.items():
            # Find corresponding numeric class if needed
            if isinstance(class_idx, str) and hasattr(self, 'classes_') and any(isinstance(c, str) for c in self.classes_):
                # Try to find class index
                try:
                    numeric_idx = np.where(self.classes_ == class_idx)[0][0]
                except:
                    # If class not found, use raw text
                    class_tokens[class_idx] = description
                    continue
            else:
                numeric_idx = class_idx
                
            # Tokenize description
            tokens = tokenizer(
                description,
                return_tensors="pt",
                padding="max_length",
                max_length=77,
                truncation=True
            ).to(self.device)
            
            class_tokens[numeric_idx] = tokens
            
        return class_tokens
    
    def _standardize_features(self, X_train, X_test, exclude_columns=None):
        """
        Standardize features to have zero mean and unit variance.
        
        Parameters:
        -----------
        X_train : torch.Tensor
            Training data
        X_test : torch.Tensor
            Test data
        exclude_columns : list
            Indices of columns to exclude from standardization
            
        Returns:
        --------
        tuple
            Standardized training and test data
        """
        exclude_set = set(exclude_columns or [])
        X_train_std = X_train.clone()
        X_test_std = X_test.clone()
        
        # Process each feature independently
        for col in range(X_train.shape[1]):
            if col in exclude_set:
                continue  # Skip semantic columns
                
            # Compute mean and std from training data
            col_mean = X_train[:, col].mean()
            col_std = X_train[:, col].std()
            
            # Avoid division by zero
            if col_std == 0:
                col_std = 1.0
                
            # Standardize both train and test
            X_train_std[:, col] = (X_train[:, col] - col_mean) / col_std
            X_test_std[:, col] = (X_test[:, col] - col_mean) / col_std
            
        return X_train_std, X_test_std
    
    def _robust_scale_features(self, X_train, X_test, exclude_columns=None):
        """
        Scale features using robust scaling (median and IQR).
        
        Parameters:
        -----------
        X_train : torch.Tensor
            Training data
        X_test : torch.Tensor
            Test data
        exclude_columns : list
            Indices of columns to exclude from scaling
            
        Returns:
        --------
        tuple
            Scaled training and test data
        """
        exclude_set = set(exclude_columns or [])
        X_train_scaled = X_train.clone()
        X_test_scaled = X_test.clone()
        
        # Process each feature independently
        for col in range(X_train.shape[1]):
            if col in exclude_set:
                continue  # Skip semantic columns
                
            # Get column values and sort
            col_values = X_train[:, col].cpu().numpy()
            
            # Compute median and IQR from training data
            median = np.median(col_values)
            q75, q25 = np.percentile(col_values, [75, 25])
            iqr = q75 - q25
            
            # Avoid division by zero
            if iqr == 0:
                iqr = 1.0
                
            # Scale both train and test
            X_train_scaled[:, col] = (X_train[:, col] - median) / iqr
            X_test_scaled[:, col] = (X_test[:, col] - median) / iqr
            
        return X_train_scaled, X_test_scaled
    
    def predict_from_text(self, X, text_description):
        """
        Make predictions directly from text descriptions if the model supports it.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test samples
        text_description : str
            Text description of the target class
            
        Returns:
        --------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels
        """
        if not hasattr(self.model, 'predict_from_text'):
            raise NotImplementedError("The underlying model does not support text-based prediction")
            
        # Check if fit has been called
        check_is_fitted(self)
        
        # Input validation
        X = check_array(X)
        
        # Apply feature selection if needed
        if self.feature_indices is not None:
            X = X[:, self.feature_indices]
        
        # Convert to torch tensor with proper dtype for device
        if self.device == 'mps':
            # MPS doesn't support float64, explicitly convert to float32
            X_test = torch.tensor(X, dtype=torch.float32, device=self.device)
        else:
            X_test = torch.tensor(X, device=self.device).float()
        
        # Check expected feature dimensions from model
        if hasattr(self.model.base_model, 'encoder') and hasattr(self.model.base_model.encoder, 'weight'):
            expected_input_dim = self.model.base_model.encoder.weight.shape[1]
            if X_test.shape[1] != expected_input_dim:
                if self.verbose:
                    print(f"Reshaping input data from {X_test.shape[1]} to {expected_input_dim} features for text-based prediction")
                
                # Pad or truncate data to match expected dimensions
                if X_test.shape[1] < expected_input_dim:
                    # Pad training data with zeros
                    X_train_pad = torch.zeros((self.X_.shape[0], expected_input_dim), 
                                              device=self.X_.device, dtype=self.X_.dtype)
                    X_train_pad[:, :self.X_.shape[1]] = self.X_
                    
                    # Pad test data with zeros 
                    X_test_pad = torch.zeros((X_test.shape[0], expected_input_dim), 
                                             device=X_test.device, dtype=X_test.dtype)
                    X_test_pad[:, :X_test.shape[1]] = X_test
                    
                    X_train = X_train_pad
                    X_test = X_test_pad
                else:
                    # Truncate if we have more features than expected
                    X_train = self.X_[:, :expected_input_dim]
                    X_test = X_test[:, :expected_input_dim]
            else:
                X_train = self.X_
        else:
            X_train = self.X_
        
        # Combine training and test data for in-context learning
        X_full = torch.cat([X_train, X_test], dim=0).unsqueeze(1)
        
        # Create zeros tensor for test labels with appropriate device
        test_zeros = torch.zeros(len(X_test), dtype=torch.long, device=self.device)
        
        # Ensure y labels are on the correct device
        if self.y_.device != self.device:
            y_train = self.y_.to(self.device)
        else:
            y_train = self.y_
            
        # Combine train and test labels
        y_full = torch.cat([y_train, test_zeros], dim=0).unsqueeze(1)
        
        # Call the model's text-based prediction method
        results = self.model.predict_from_text((X_full, y_full), text_description)
        
        # Extract predictions - the format depends on the implementation
        if isinstance(results, dict) and 'class_preds' in results:
            # Default return format from SemanticAwareClassifier
            preds = results['class_preds'].cpu().numpy()
            return self.classes_[preds]
        elif isinstance(results, dict) and 'mapped_class' in results:
            # Alternative format with a single mapped class
            mapped_class = results['mapped_class']
            # Return the same prediction for all samples
            return np.full(len(X_test), self.classes_[mapped_class])
        else:
            # Unknown format - try to interpret as class indices
            try:
                preds = np.array(results)
                return self.classes_[preds]
            except:
                # Last resort - return zeros
                return np.zeros(len(X_test), dtype=int)