import numpy as np
import torch
import random
import itertools
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array
import pandas as pd

from ticl.utils import log_gpu_memory


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
                 semantic_class_descriptions=None):
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
        """
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
        
        # Get model's capabilities from config
        if "prior" in config:
            self.max_num_features = config['prior']['num_features']
            if 'classification' in config['prior']:
                self.max_num_classes = config['prior']['classification']['max_num_classes']
            else:
                self.max_num_classes = 2
        else:
            self.max_num_features = config.get('num_features', 100)
            self.max_num_classes = config.get('max_num_classes', 2)
            
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
            print(f"Max features: {self.max_num_features}, Max classes: {self.max_num_classes}")
            print(f"Using {N_ensemble_configurations} ensemble configurations")
            print(f"Semantic feature probability: {self.semantic_feature_p}")
        
    def fit(self, X, y, semantic_column_indices=None, X_semantic_text=None):
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
        
        Returns:
        --------
        self : object
            Returns self
        """
        # Validate inputs
        X, y = check_X_y(X, y, force_all_finite=False)
        
        # Store classes seen during fit
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError("Need samples of at least 2 classes in the data, "
                             f"but the data contains only one class: {self.classes_[0]}")
        
        # Encode class labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Handle semantic columns
        if semantic_column_indices is not None:
            self.semantic_column_indices = semantic_column_indices
        
        # If we have semantic column indices but no textual data, try to infer it from X
        if self.semantic_column_indices and X_semantic_text is None:
            if self.verbose:
                print(f"Semantic columns identified: {self.semantic_column_indices}")
                print("No semantic text provided - will use numeric values for semantic columns")
            # In a real implementation, you might extract text from a pandas DataFrame here
            self.has_semantic_data = False
        elif X_semantic_text is not None:
            if self.verbose:
                print(f"Using provided semantic text data for {len(self.semantic_column_indices)} columns")
            self.X_semantic_text = X_semantic_text
            self.has_semantic_data = True
        else:
            if self.verbose:
                print("No semantic columns identified")
            self.has_semantic_data = False
            
        # Convert to torch tensors and store
        self.X_ = torch.tensor(X, device=self.device).float()
        self.y_ = torch.tensor(y_encoded, device=self.device).long()
        
        # Handle feature count constraints
        if X.shape[1] > self.max_num_features:
            if self.verbose:
                print(f"Warning: Input has {X.shape[1]} features, but model supports {self.max_num_features}. "
                      f"Random features will be selected.")
            # Randomly select features
            self.feature_indices = np.random.choice(X.shape[1], self.max_num_features, replace=False)
            self.X_ = self.X_[:, self.feature_indices]
            
            # Update semantic column indices if we have them
            if self.semantic_column_indices:
                self.semantic_column_indices = [i for i in self.semantic_column_indices if i in self.feature_indices]
        else:
            self.feature_indices = None
        
        # Prepare class descriptions if we have semantic aware model
        if self.semantic_class_descriptions is None and hasattr(self.model, 'generate_boundaries_from_text'):
            # If no descriptions provided, create simple class names
            self.semantic_class_descriptions = {
                f"Class {i}": f"This is class {self.label_encoder.inverse_transform([i])[0]}"
                for i in range(len(self.classes_))
            }
            if self.verbose:
                print(f"Created default class descriptions: {self.semantic_class_descriptions}")
        
        # Track if the model has been fitted
        self.is_fitted_ = True
        
        if self.verbose:
            print(f"Fitted model with {len(self.X_)} samples, {self.X_.shape[1]} features, {len(self.classes_)} classes")
            
        return self
    
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
        if self.semantic_column_indices and self.has_semantic_data:
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
        # Check if fit had been called
        check_is_fitted(self, ['is_fitted_'])
        
        # Input validation
        X = check_array(X, force_all_finite=False)
        
        # Apply feature selection if needed
        if self.feature_indices is not None:
            X = X[:, self.feature_indices]
            if self.verbose:
                print(f"Applied feature selection: {X.shape[1]} features")
        
        # Convert to tensors using TabPFN approach
        if self.verbose:
            print(f"Input shape: {X.shape}, training shape: {self.X_.shape}")
            
        # Concatenate training and test data
        if torch.is_tensor(self.X_) and torch.is_tensor(X):
            # Both are already tensors
            X_full = torch.cat((self.X_, torch.tensor(X, device=self.device)), dim=0).float().unsqueeze(1).to(self.device)
        else:
            # Convert both to tensors
            X_full = np.concatenate([self.X_, X], axis=0)
            X_full = torch.tensor(X_full, device=self.device).float().unsqueeze(1)
            
        # Create targets tensor
        y_full = np.concatenate([self.y_.cpu().numpy() if torch.is_tensor(self.y_) else self.y_, 
                                 np.zeros(shape=X.shape[0])], axis=0)
        y_full = torch.tensor(y_full, device=self.device).float().unsqueeze(1)
        
        # Position for evaluation
        eval_pos = len(self.X_)
        
        # Use the class descriptions if provided, otherwise use stored ones
        descriptions = class_descriptions or self.semantic_class_descriptions
        
        # Detect if we should use zero-padding from config (TabPFN approach)
        try:
            if hasattr(self.model.base_model, 'c') and 'prior' in self.model.base_model.c:
                extend_features = self.model.base_model.c['prior']['classification'].get('pad_zeros', True)
            elif hasattr(self.model, 'c') and 'prior' in self.model.c:
                extend_features = self.model.c['prior']['classification'].get('pad_zeros', True)
            elif hasattr(self.config, 'get') and 'prior' in self.config:
                extend_features = self.config['prior']['classification'].get('pad_zeros', True)
            else:
                extend_features = True
        except (KeyError, AttributeError):
            extend_features = True
            
        if self.verbose:
            print(f"Feature extension enabled: {extend_features}")
            
        # Get maximum number of features the model supports
        max_features = self.max_num_features
        if self.verbose:
            print(f"Max features: {max_features}")
        
        # Prepare prediction configurations
        preprocess_transform = 'none' if getattr(self, 'no_preprocess_mode', False) else 'mix'
        preprocess_transform_configurations = ['none', 'power_all'] if preprocess_transform == 'mix' else [preprocess_transform]
        
        # Determine categorical features (semantic features should be treated as categorical)
        categorical_feats = self.semantic_column_indices or []
        
        # Set random seed
        if self.seed is not None:
            torch.manual_seed(self.seed)
            random.seed(self.seed)
            np.random.seed(self.seed)
            
        # Create ensemble configurations (following TabPFN approach)
        feature_shift_configurations = torch.randperm(X_full.shape[2]) if getattr(self, 'feature_shift_decoder', True) else [0]
        class_shift_configurations = torch.randperm(len(torch.unique(y_full[:eval_pos]))) if getattr(self, 'multiclass_decoder', 'permutation') == 'permutation' else [0]
        
        ensemble_configurations = list(itertools.product(class_shift_configurations, feature_shift_configurations))
        
        # Shuffle configurations
        rng = random.Random(self.seed)
        rng.shuffle(ensemble_configurations)
        ensemble_configurations = list(itertools.product(ensemble_configurations, preprocess_transform_configurations))
        ensemble_configurations = ensemble_configurations[0:self.N_ensemble_configurations]
        
        if self.verbose:
            print(f"Using {len(ensemble_configurations)} ensemble configurations")
            
        # Start ensemble prediction
        output = None
        X_transformed = {}
        inputs, labels = [], []
        
        for ensemble_configuration in ensemble_configurations:
            (class_shift_configuration, feature_shift_configuration), preprocess_transform_configuration = ensemble_configuration
            
            X_, y_ = X_full.clone(), y_full.clone()
            
            # Check if we have already transformed this configuration
            if preprocess_transform_configuration in X_transformed:
                X_ = X_transformed[preprocess_transform_configuration].clone()
            else:
                # Use TabPFN's preprocess_input function with our semantic column awareness
                X_ = self._preprocess_input(
                    X_, 
                    y_full[:eval_pos], 
                    preprocess_transform=preprocess_transform_configuration, 
                    max_features=max_features,
                    normalize_with_test=normalize_with_test, 
                    eval_position=eval_pos, 
                    categorical_feats=categorical_feats,
                    scale=getattr(self, 'scale', True), 
                    normalize_by_used_features=extend_features
                )
                X_transformed[preprocess_transform_configuration] = X_
                
            # Apply class shift (label permutation)
            y_ = ((y_[:eval_pos] + class_shift_configuration) % len(self.classes_)).float()
            
            # Apply feature shift
            X_ = torch.cat([X_[..., feature_shift_configuration:], X_[..., :feature_shift_configuration]], dim=-1)
            
            # Extend features if needed (zero padding)
            if extend_features and X_.shape[2] < max_features:
                X_ = torch.cat(
                    [X_, torch.zeros((X_.shape[0], X_.shape[1], max_features - X_.shape[2])).to(self.device)], -1)
                
            inputs += [X_]
            labels += [y_]
            
        # Combine all configurations into batches
        inputs = torch.cat(inputs, 1)
        inputs = torch.split(inputs, self.batch_size, dim=1)
        labels = torch.cat(labels, 1)
        labels = torch.split(labels, self.batch_size, dim=1)
        
        # Run prediction on batches
        outputs = []
        num_classes = len(self.classes_)
        softmax_temperature = getattr(self, 'temperature', torch.log(torch.tensor([0.8], device=self.device)))

        # Process each batch
        with torch.no_grad():
            for batch_input, batch_label in zip(inputs, labels):
                # Get model prediction
                # Run the model in evaluation mode
                self.model.eval()
                
                # Handle case where the model yields different outputs
                if hasattr(self.model, 'generate_boundaries_from_text') and descriptions and len(descriptions) > 0:
                    # Use text-based boundary generation if available
                    model_output = self.model.generate_boundaries_from_text(
                        (batch_input, batch_label.float()), 
                        class_descriptions=descriptions
                    )
                    if isinstance(model_output, dict) and 'class_logits' in model_output:
                        output_batch = model_output['class_logits']
                    else:
                        # Get predictions from other fields if class_logits not available
                        output_batch = model_output.get('logits', model_output.get('class_preds', None))
                        if output_batch is None:
                            raise ValueError("Could not extract predictions from model output")
                else:
                    # Standard forward pass
                    output = self.model(
                        (batch_input, batch_label.float()),
                        single_eval_pos=eval_pos
                    )
                    
                    # Handle different output formats
                    if isinstance(output, dict) and 'class_logits' in output:
                        output_batch = output['class_logits']
                    elif isinstance(output, dict) and 'logits' in output:
                        output_batch = output['logits']
                    else:
                        # Direct tensor output (base model pass-through)
                        output_batch = output
                
                # Apply temperature scaling if needed
                output_batch = output_batch[:, :, 0:num_classes] / torch.exp(softmax_temperature)
                outputs.append(output_batch)
        
        # Combine all batch outputs
        if outputs:
            outputs = torch.cat(outputs, 1)
            
            # Process each ensemble configuration
            output = None
            for i, ensemble_configuration in enumerate(ensemble_configurations):
                (class_shift_configuration, feature_shift_configuration), preprocess_transform_configuration = ensemble_configuration
                output_ = outputs[:, i:i+1, :]
                
                # Reverse class shift
                output_ = torch.cat([output_[..., class_shift_configuration:], output_[..., :class_shift_configuration]], dim=-1)
                
                # Average or convert to probabilities
                if not return_logits:
                    # Apply softmax to convert to probabilities
                    output_ = torch.nn.functional.softmax(output_, dim=-1)
                
                # Add to ensemble output
                output = output_ if output is None else output + output_
            
            # Average ensemble outputs
            output = output / len(ensemble_configurations)
            
            # Apply final softmax if needed
            if not return_logits:
                # Already applied softmax to each configuration
                pass
            
            # Get test set predictions (transpose to match sklearn format)
            output = torch.transpose(output, 0, 1)
            prediction = output.detach().cpu().numpy()
            
            # Return prediction probabilities for all classes
            return prediction
        else:
            # Fallback if no predictions could be generated
            return np.zeros((X.shape[0], len(self.classes_)))
            
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
        
        # Use provided device or default to the instance device
        device = device or self.device
        
        # Check batch dimension
        if eval_xs.shape[1] > 1:
            raise Exception("Transforms only allow one batch dim")
            
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
                    # Choose from remaining indices
                    remaining_selected = sorted(np.random.choice(
                        list(remaining_indices), 
                        min(remaining_slots, len(remaining_indices)), 
                        replace=False
                    ))
                    # Combine selected indices
                    selected_indices = sorted(list(semantic_indices) + remaining_selected)
                else:
                    # Only have room for some semantic columns
                    selected_indices = sorted(list(semantic_indices)[:max_features])
            else:
                # No semantic columns, just select random features
                selected_indices = sorted(np.random.choice(eval_xs.shape[2], max_features, replace=False))
                
            # Apply selection
            eval_xs = eval_xs[:, :, selected_indices]
            
            if self.verbose:
                print(f"Selected {len(selected_indices)} features for preprocessing")
        
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
            return len(torch.unique(col_tensor[~col_tensor.isnan()])) > 1
            
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
                
            # Convert back to torch tensor
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
        # Get probability predictions
        proba = self.predict_proba(X, class_descriptions=class_descriptions)
        
        # Get the most likely class
        indices = np.argmax(proba, axis=1)
        
        # Map indices back to original classes
        return self.classes_[indices]
    
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
        check_is_fitted(self, ['is_fitted_'])
        
        # Input validation
        X = check_array(X, force_all_finite=False)
        
        # Apply feature selection if needed
        if self.feature_indices is not None:
            X = X[:, self.feature_indices]
        
        # Convert to torch tensor
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
        y_full = torch.cat([self.y_, torch.zeros(len(X_test), dtype=torch.long, device=self.device)], dim=0).unsqueeze(1)
        
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