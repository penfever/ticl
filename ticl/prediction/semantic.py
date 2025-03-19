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
        
    def predict_proba(self, X, class_descriptions=None):
        """
        Predict class probabilities for the input samples X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test samples
        class_descriptions : dict, optional
            Mapping of class labels to text descriptions
            
        Returns:
        --------
        y_proba : array-like of shape (n_samples, n_classes)
            Class probabilities for each sample
        """
        # Check if fit has been called
        check_is_fitted(self, ['is_fitted_'])
        
        # Input validation
        X = check_array(X, force_all_finite=False)
        
        # Apply feature selection if needed
        if self.feature_indices is not None:
            X = X[:, self.feature_indices]
        
        # Convert to torch tensor
        X_test = torch.tensor(X, device=self.device).float()
        
        # Evaluation position
        eval_pos = len(self.X_)
        
        # Use the class descriptions if provided, otherwise use stored ones
        descriptions = class_descriptions or self.semantic_class_descriptions
        
        # Generate ensemble configurations
        ensemble_configurations = self.get_ensemble_configurations()
        if self.verbose:
            print(f"Using {len(ensemble_configurations)} ensemble configurations for prediction")
        
        # Run ensemble predictions
        all_probs = []
        for config_idx, config in enumerate(ensemble_configurations):
            feature_perm, semantic_config, preprocess_config, label_perm = config
            
            if self.verbose:
                print(f"Ensemble configuration {config_idx+1}/{len(ensemble_configurations)}")
            
            # Apply feature permutation
            X_train_perm = self.X_[:, feature_perm].clone()
            X_test_perm = X_test[:, feature_perm].clone()
            
            # Apply label permutation
            y_train_perm = torch.tensor(
                [(self.y_[i].item() + label_perm[self.y_[i].item()]) % len(self.classes_) 
                 for i in range(len(self.y_))],
                device=self.device, dtype=self.y_.dtype
            )
            
            # Preprocess data according to config
            if preprocess_config == "standard":
                # Standardize features (excluding semantic ones)
                X_train_perm, X_test_perm = self._standardize_features(
                    X_train_perm, X_test_perm, exclude_columns=self.semantic_column_indices
                )
            elif preprocess_config == "robust":
                # Robust scaling (excluding semantic ones)
                X_train_perm, X_test_perm = self._robust_scale_features(
                    X_train_perm, X_test_perm, exclude_columns=self.semantic_column_indices
                )
            
            # Combine data for in-context learning
            X_full = torch.cat([X_train_perm, X_test_perm], dim=0).unsqueeze(1)
            y_full = torch.cat([y_train_perm, torch.zeros(len(X_test_perm), dtype=torch.long, device=self.device)], dim=0).unsqueeze(1)
            
            # Run inference in batches
            batch_probs = []
            with torch.no_grad():
                for i in range(0, len(X_test_perm), self.batch_size):
                    # Prepare batch
                    batch_end = min(i + self.batch_size, len(X_test_perm))
                    batch_size = batch_end - i
                    
                    # Select relevant slice of data
                    batch_X = torch.cat([X_train_perm, X_test_perm[i:batch_end]], dim=0).unsqueeze(1)
                    batch_y = torch.cat([y_train_perm, torch.zeros(batch_size, dtype=torch.long, device=self.device)], dim=0).unsqueeze(1)
                    
                    # Run inference - with class descriptions if supported
                    if hasattr(self.model, 'generate_boundaries_from_text') and descriptions and semantic_config:
                        # Use semantic zero-shot capabilities
                        # Randomly sample text descriptions to create diversity
                        shuffled_descriptions = dict(descriptions)
                        if semantic_config.startswith("semantic_sample"):
                            # Add some randomness to the descriptions for this ensemble member
                            shuffled_descriptions = {
                                k: v + f" (variation {semantic_config.split('_')[-1]})"
                                for k, v in descriptions.items()
                            }
                        
                        # Use the text-based prediction method
                        model_output = self.model.generate_boundaries_from_text(
                            (batch_X, batch_y.float()), 
                            class_descriptions=shuffled_descriptions
                        )
                        logits = model_output.get('class_logits', model_output.get('logits', None))
                    else:
                        # Standard forward pass
                        model_output = self.model((batch_X, batch_y.float()), single_eval_pos=eval_pos)
                        
                        # Handle various output formats from semantic model
                        if isinstance(model_output, dict) and 'class_logits' in model_output:
                            # Standard dictionary output from SemanticAwareClassifier
                            logits = model_output['class_logits']
                        elif isinstance(model_output, dict) and 'logits' in model_output:
                            # Alternative dictionary format
                            logits = model_output['logits']
                        else:
                            # Direct tensor output (base model pass-through)
                            logits = model_output
                    
                    # Apply softmax to get probabilities
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    
                    # Extract test set probabilities
                    test_probs = probs[eval_pos:].cpu().numpy()
                    batch_probs.append(test_probs)
                    
                    # Clean up batch tensors
                    del batch_X, batch_y, model_output, logits, probs
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # Combine batch results for this configuration
            config_probs = np.concatenate(batch_probs, axis=0)
            
            # Reverse the label permutation
            inv_label_perm = np.zeros_like(label_perm)
            for i, p in enumerate(label_perm):
                inv_label_perm[p] = i
            config_probs = config_probs[:, inv_label_perm]
            
            # Add to ensemble
            all_probs.append(config_probs)
            
            # Clean up
            del X_train_perm, X_test_perm
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Average ensemble predictions
        if all_probs:
            ensemble_probs = sum(all_probs) / len(all_probs)
            
            # Handle potential class count mismatch
            if ensemble_probs.shape[1] < len(self.classes_):
                # Pad with zeros if model outputs fewer classes than we've seen
                padding = np.zeros((ensemble_probs.shape[0], len(self.classes_) - ensemble_probs.shape[1]))
                ensemble_probs = np.hstack([ensemble_probs, padding])
            elif ensemble_probs.shape[1] > len(self.classes_):
                # Truncate if model outputs more classes than we've seen
                ensemble_probs = ensemble_probs[:, :len(self.classes_)]
            
            return ensemble_probs
        else:
            # Fallback if no predictions were generated
            return np.zeros((len(X_test), len(self.classes_)))
    
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
        
        # Combine training and test data for in-context learning
        X_full = torch.cat([self.X_, X_test], dim=0).unsqueeze(1)
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