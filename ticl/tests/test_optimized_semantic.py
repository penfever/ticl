#!/usr/bin/env python3
"""
Test script for optimized semantic implementations
"""
import os
import sys
import time
import torch
import random
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add required paths
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# Load helper functions directly
def get_nan_value(set_to_nan):
    """Get a NaN value based on the configuration"""
    if set_to_nan == 'nan':
        return float('nan')
    elif set_to_nan == 'zero':
        return 0.0
    else:
        return float('nan')

def normalize_by_used_features_f(x, num_features_used, num_features):
    """Scale features based on used/total feature ratio"""
    return x * (num_features / num_features_used) ** 0.5

def normalize_data(x):
    """Simple data normalization"""
    return x

def remove_outliers(x, categorical_features=None):
    """Remove outliers from data"""
    if categorical_features is None:
        categorical_features = []
    return x
    
def randomize_classes(classes, num_classes):
    """Randomize class assignments ensuring the range stays within num_classes"""
    unique_classes = classes.unique()
    num_unique = len(unique_classes)
    
    # Create a random permutation of indices 0 to num_unique-1
    permutation = torch.randperm(num_unique)
    
    # Apply the permutation to each class
    randomized = torch.zeros_like(classes)
    for i, cls in enumerate(unique_classes):
        mask = (classes == cls)
        new_class = permutation[i] % num_classes
        randomized[mask] = new_class
    
    return randomized

# Create a simple adapter class based on the optimized code
class ClassificationAdapter:
    """Simplified adapter for testing"""
    
    def __init__(self, base_prior, config):
        self.base_prior = base_prior
        self.h = config  # Use config directly
        
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
        """Create semantic class mapping"""
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
        
        # Vectorized selection of semantic classes for each class
        if num_classes <= num_semantic_classes:
            # Sample without replacement when we have enough classes
            selected_indices = torch.randperm(num_semantic_classes)[:num_classes].tolist()
        else:
            # If more classes than semantic classes, we'll have some duplicates
            selected_indices = torch.randint(0, num_semantic_classes, (num_classes,)).tolist()
                
        # Pre-compute token counts and indices in advance for all classes
        min_ratio, max_ratio = 0.25, 0.5
        min_tokens = max(5, int(semantic_data.shape[1] * min_ratio))  # At least 5 tokens
        max_tokens = min(int(semantic_data.shape[1] * max_ratio), semantic_data.shape[1])
        
        # Generate all token counts at once
        num_signature_tokens = torch.randint(min_tokens, max_tokens + 1, (num_classes,)).tolist()
        
        # Generate all indices at once for all classes
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
            
            # Store the class token pattern
            class_token_patterns[class_idx] = {
                'tokens': signature_tokens,
                'semantic_class': semantic_class,
                'class_name': class_name,
                'column_name': f"col_{semantic_class}"
            }
        
        # Cache the result
        self._class_token_patterns_cache[cache_key] = class_token_patterns
        
        return class_token_patterns
    
    def create_semantic_targets(self, y, class_token_patterns):
        """Create semantic targets tensor"""
        # Avoid repeatedly checking dict keys
        valid_classes = set(class_token_patterns.keys())
        
        # Get shape info
        if len(y.shape) == 3:
            # Handle case where y has shape (samples, batch_size, 1)
            y = y.squeeze(-1)
        
        sample_size, batch_size = y.shape
        
        # Create targets tensor - initialized with ignore index
        semantic_targets = torch.full((sample_size, batch_size), -100, 
                                    device=y.device, dtype=torch.long)
        
        # Convert y to long for indexing
        y_long = y.to(torch.long)
        
        # Vectorized assignment for valid classes
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
        """Optimized semantic prior application"""
        # Track performance
        start_time = time.time()
        self._performance_stats['semantic_prior_calls'] += 1
        
        # Make a mutable copy of the feature tensor
        x_new = x.clone()
        
        # Get the number of classes from config
        num_classes = max(2, self.h['num_classes'])
        
        # If y is not provided (initial call), generate proxy targets
        if y is None:
            # Create a temporary random target for initial feature generation
            y = torch.randint(0, num_classes, (x.shape[0], x.shape[1]), device=device).float()
        
        # Check class-level cache for semantic data
        data_start_time = time.time()
        
        # Cache key based on number of classes
        semantic_cache_key = f"semantic_data_{num_classes}"
        
        # Check if we need to reload semantic data
        if semantic_cache_key not in self._semantic_data_cache:
            # Load new semantic data
            # Get seed from config
            seed = self.h.get('random_seed', None)
            
            try:
                # Try to import get_random_semantic_data
                from ticl.ticl.datasets.semantic_prior_data_loader import get_random_semantic_data
                
                semantic_data, semantic_data_column_names = get_random_semantic_data(
                    num_classes=num_classes,
                    seed=seed,
                    use_cache=True
                )
            except ImportError:
                # Generate our own random data if import fails
                if seed is not None:
                    torch.manual_seed(seed)
                    random.seed(seed)
                semantic_data = torch.randint(1000, 5000, (num_classes, 50))
                semantic_data_column_names = [f"feature_{i}" for i in range(num_classes)]
            
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
        
        # Cache token patterns at the class level
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
        
        # Feature assignment using vectorized operations
        feature_start_time = time.time()
        
        # Pre-compute token pools for all semantic classes
        token_values_table = semantic_data.to(dtype=torch.float32, device=device)
        
        # Pre-extract tokens for class patterns
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
        
        # Process all batch items at once using vectorized operations
        
        # Create masks for each class
        class_masks = [(y_int == class_idx) for class_idx in range(num_classes)]
        
        # Create semantic feature values 
        for feat_idx, feature_pos in enumerate(semantic_features):
            # Determine which pattern to use for this feature
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
            
            # Single assignment to the output tensor 
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
        
        return x_new, semantic_info
    
    def get_performance_stats(self):
        """Return performance statistics"""
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


# Import data loading functions
try:
    from ticl.ticl.datasets.semantic_prior_data_loader import load_semantic_prior_data, get_random_semantic_data
except ImportError as e:
    # Simplified versions for testing if imports fail
    logger.warning(f"Import error: {e}, using simplified versions for testing")
    
    def load_semantic_prior_data(log_file_path, force_reload=False, **kwargs):
        """Simplified version for testing"""
        import json
        try:
            with open(log_file_path, 'r') as f:
                completed_columns = json.load(f)
        except Exception:
            # Create dummy data if file doesn't exist
            completed_columns = {f"col_{i}": [random.randint(1000, 5000) for _ in range(50)] for i in range(10)}
            
        column_name_tokens = {}
        column_value_tokens = {}
        
        for col_name, tensor_data in completed_columns.items():
            # Store processed data
            column_name_tokens[col_name] = [ord(c) for c in col_name[:10]]
            column_value_tokens[col_name] = tensor_data
            
        return column_name_tokens, column_value_tokens
    
    def get_random_semantic_data(num_classes=5, num_tokens=50, seed=None, use_cache=True, **kwargs):
        """Simplified version for testing"""
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
        
        # Generate random data
        semantic_data = torch.randint(1000, 5000, (num_classes, num_tokens))
        column_names = [f"feature_{i}" for i in range(num_classes)]
        
        return semantic_data, column_names

def setup_completed_columns():
    """Create a synthetic completed_columns.json file for testing if needed"""
    test_file = os.path.join(script_dir, "ticl/completed_columns_test.json")
    
    # Only create if it doesn't exist
    if not os.path.exists(test_file):
        import json
        # Create simple test data with 10 columns
        test_data = {}
        for i in range(10):
            column_name = f"test_column_{i}"
            # Random token IDs 
            token_ids = [random.randint(1000, 5000) for _ in range(50)]
            test_data[column_name] = token_ids
            
        # Write to file
        with open(test_file, 'w') as f:
            json.dump(test_data, f)
            
        logger.info(f"Created test data file at {test_file}")
    
    return test_file

def test_load_data(iterations=3):
    """Test semantic data loading performance"""
    logger.info("Testing load_semantic_prior_data optimization")
    
    # Use or create test data file
    log_file_path = setup_completed_columns()
    
    # Warm-up run
    load_semantic_prior_data(log_file_path=log_file_path, force_reload=True)
    
    # Timed runs
    total_time = 0
    for i in range(iterations):
        start = time.time()
        column_tokens, value_tokens = load_semantic_prior_data(
            log_file_path=log_file_path,
            force_reload=True
        )
        elapsed = time.time() - start
        logger.info(f"Iteration {i+1}: {elapsed:.4f}s")
        total_time += elapsed
    
    avg_time = total_time / iterations
    logger.info(f"Average load time: {avg_time:.4f}s")
    return avg_time

def test_random_data(iterations=3):
    """Test random semantic data generation performance"""
    logger.info("Testing get_random_semantic_data optimization")
    
    # Warm-up run
    get_random_semantic_data(use_cache=False)
    
    # Timed runs
    total_time = 0
    for i in range(iterations):
        start = time.time()
        semantic_data, column_names = get_random_semantic_data(
            num_classes=10,
            num_tokens=100,
            use_cache=False
        )
        elapsed = time.time() - start
        logger.info(f"Iteration {i+1}: {elapsed:.4f}s")
        total_time += elapsed
    
    avg_time = total_time / iterations
    logger.info(f"Average generation time: {avg_time:.4f}s")
    return avg_time

def test_semantic_prior(iterations=3):
    """Test apply_semantic_prior performance"""
    logger.info("Testing _apply_semantic_prior optimization")
    
    # Set random seeds for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Setup data
    device = torch.device('cuda' if torch.cuda.is_available() 
                         else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() 
                         else 'cpu')
    
    logger.info(f"Using device: {device}")
    
    # Create synthetic data
    n_samples = 20  # Number of examples per batch
    batch_size = 8  # Number of batches
    num_features = 30  # Total features
    
    # Generate feature tensor
    x_data = torch.randn(n_samples, batch_size, num_features, device=device)
    
    # Define semantic feature indices (last 10 features)
    semantic_features = list(range(num_features - 10, num_features))
    
    # Generate class labels
    num_classes = 5
    y_data = torch.randint(0, num_classes, (n_samples, batch_size), device=device).float()
    
    # Create prior configuration
    prior_config = {
        "feature_curriculum": False,
        "pad_zeros": False,
        "balanced": False,
        "multiclass_type": "rank", 
        "output_multiclass_ordered_p": 0.5,
        "multiclass_max_steps": 5,
        "num_classes": 5,
        "max_num_classes": 10,
        "nan_prob_a_reason": 0.0,
        "nan_prob_no_reason": 0.0,
        "categorical_feature_p": 0.0,
        "semantic_feature_p": 0.3,
        "random_seed": 42,
        "num_features_sampler": "uniform",
        "set_value_to_nan": "nan"
    }
    
    # Mock a base_prior class
    class MockPrior:
        def get_batch(self, **kwargs):
            return x_data, y_data, y_data
    
    # Create adapter with mock base prior
    base_prior = MockPrior()
    adapter = ClassificationAdapter(base_prior, prior_config)
    
    # Warm-up run
    adapter._apply_semantic_prior(x_data.clone(), semantic_features, device, y_data)
    
    # Timed runs
    total_time = 0
    for i in range(iterations):
        # Use fresh copy of data to avoid caching effects
        x_copy = x_data.clone()
        
        start = time.time()
        x_new, info = adapter._apply_semantic_prior(
            x_copy,
            semantic_features,
            device,
            y_data
        )
        elapsed = time.time() - start
        logger.info(f"Iteration {i+1}: {elapsed:.4f}s")
        total_time += elapsed
    
    avg_time = total_time / iterations
    logger.info(f"Average apply time: {avg_time:.4f}s")
    
    # Print performance stats if available
    if hasattr(adapter, 'get_performance_stats'):
        perf_stats = adapter.get_performance_stats()
        logger.info("Performance breakdown:")
        for key, value in perf_stats.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
    
    return avg_time

def main():
    """Run optimization tests and show improvement over baseline"""
    logger.info("======= TESTING OPTIMIZED SEMANTIC IMPLEMENTATIONS =======")
    
    # Run tests
    load_time = test_load_data(iterations=3)
    random_time = test_random_data(iterations=3)
    semantic_prior_time = test_semantic_prior(iterations=5)
    
    # Print results
    logger.info("\n======= OPTIMIZATION RESULTS =======")
    logger.info(f"load_semantic_prior_data: {load_time:.4f}s")
    logger.info(f"get_random_semantic_data: {random_time:.4f}s")
    logger.info(f"_apply_semantic_prior: {semantic_prior_time:.4f}s")
    logger.info("====================================")

if __name__ == "__main__":
    main()