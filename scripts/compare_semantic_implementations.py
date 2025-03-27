#!/usr/bin/env python3
"""
Comparison script for semantic data implementation optimizations.
Tests both original and optimized implementations of:
1. load_semantic_prior_data
2. _apply_semantic_prior
"""
import os
import sys
import torch
import random
import time
import logging
import numpy as np
from contextlib import contextmanager
from unittest.mock import patch, MagicMock

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add ticl to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import original implementations
from ticl.datasets.semantic_prior_data_loader import load_semantic_prior_data, get_random_semantic_data
from ticl.priors.classification_adapter import ClassificationAdapter

# Import optimized implementations
from ticl.datasets.semantic_prior_data_loader_optimized import (
    load_semantic_prior_data as load_semantic_prior_data_optimized,
    get_random_semantic_data as get_random_semantic_data_optimized
)
from ticl.priors.classification_adapter_optimized import ClassificationAdapterOptimized

class TimingContextManager:
    """Context manager for timing operations with proper return value"""
    def __init__(self, name="Operation"):
        self.name = name
        self.elapsed = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.time() - self.start_time
        logger.info(f"{self.name} took {self.elapsed:.4f} seconds")
        
def timing(name="Operation"):
    """Function that returns a timing context manager"""
    return TimingContextManager(name)

def setup_test_data(device):
    """Setup test data for benchmark comparisons"""
    # Set random seeds for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    
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
        "nan_prob_a_reason": 0.0,
        "nan_prob_no_reason": 0.0,
        "max_num_classes": 10,
        "categorical_feature_p": 0.0,
        "semantic_feature_p": 0.3,
        "random_seed": 42
    }
    
    return {
        "x_data": x_data,
        "y_data": y_data,
        "semantic_features": semantic_features,
        "prior_config": prior_config,
        "num_classes": num_classes
    }

def create_test_json(filename, num_columns=20):
    """Create a test JSON file with synthetic column data"""
    import json
    
    # Create mock semantic data
    mock_data = {}
    for i in range(num_columns):
        column_name = f"test_column_{i}"
        # Create random token IDs between 1000-2000 for each column
        token_ids = [random.randint(1000, 2000) for _ in range(50)]
        mock_data[column_name] = token_ids
        
    # Write to file
    with open(filename, 'w') as f:
        json.dump(mock_data, f)
    
    return filename

def compare_load_semantic_prior_data(filename, iterations=3):
    """Compare original and optimized load_semantic_prior_data"""
    logger.info("=" * 60)
    logger.info("COMPARING load_semantic_prior_data")
    logger.info("=" * 60)
    
    # Run original implementation
    original_times = []
    for i in range(iterations):
        logger.info(f"Original implementation - iteration {i+1}/{iterations}")
        with timing("Original implementation") as timer:
            column_tokens, value_tokens = load_semantic_prior_data(
                log_file_path=filename,
                force_reload=True,
                use_cache=False
            )
        original_times.append(timer.elapsed)
    
    # Run optimized implementation
    optimized_times = []
    for i in range(iterations):
        logger.info(f"Optimized implementation - iteration {i+1}/{iterations}")
        with timing("Optimized implementation") as timer:
            column_tokens_opt, value_tokens_opt = load_semantic_prior_data_optimized(
                log_file_path=filename,
                force_reload=True,
                use_cache=False
            )
        optimized_times.append(timer.elapsed)
    
    # Verify results match
    assert len(column_tokens) == len(column_tokens_opt), "Column token counts don't match"
    assert len(value_tokens) == len(value_tokens_opt), "Value token counts don't match"
    
    # Calculate statistics
    avg_original = sum(original_times) / len(original_times)
    avg_optimized = sum(optimized_times) / len(optimized_times)
    improvement = (avg_original - avg_optimized) / avg_original * 100
    
    logger.info(f"Original implementation: {avg_original:.4f}s average")
    logger.info(f"Optimized implementation: {avg_optimized:.4f}s average")
    logger.info(f"Improvement: {improvement:.2f}%")
    
    return {
        "original_times": original_times,
        "optimized_times": optimized_times,
        "avg_original": avg_original,
        "avg_optimized": avg_optimized,
        "improvement": improvement
    }

def compare_get_random_semantic_data(iterations=5):
    """Compare original and optimized get_random_semantic_data"""
    logger.info("=" * 60)
    logger.info("COMPARING get_random_semantic_data")
    logger.info("=" * 60)
    
    # Run original implementation
    original_times = []
    for i in range(iterations):
        logger.info(f"Original implementation - iteration {i+1}/{iterations}")
        with timing("Original implementation") as timer:
            # Force reload by not using cache
            semantic_data, column_names = get_random_semantic_data(
                num_classes=5,
                use_cache=False
            )
        original_times.append(timer.elapsed)
    
    # Run optimized implementation
    optimized_times = []
    for i in range(iterations):
        logger.info(f"Optimized implementation - iteration {i+1}/{iterations}")
        with timing("Optimized implementation") as timer:
            # Force reload by not using cache
            semantic_data_opt, column_names_opt = get_random_semantic_data_optimized(
                num_classes=5, 
                use_cache=False
            )
        optimized_times.append(timer.elapsed)
    
    # Verify results have expected form
    assert semantic_data.shape[0] == 5, "Expected 5 semantic classes"
    assert semantic_data_opt.shape[0] == 5, "Expected 5 semantic classes"
    
    # Calculate statistics
    avg_original = sum(original_times) / len(original_times)
    avg_optimized = sum(optimized_times) / len(optimized_times)
    improvement = (avg_original - avg_optimized) / avg_original * 100
    
    logger.info(f"Original implementation: {avg_original:.4f}s average")
    logger.info(f"Optimized implementation: {avg_optimized:.4f}s average")
    logger.info(f"Improvement: {improvement:.2f}%")
    
    return {
        "original_times": original_times,
        "optimized_times": optimized_times,
        "avg_original": avg_original,
        "avg_optimized": avg_optimized,
        "improvement": improvement
    }

def compare_apply_semantic_prior(test_data, iterations=3):
    """Compare original and optimized _apply_semantic_prior"""
    logger.info("=" * 60)
    logger.info("COMPARING _apply_semantic_prior")
    logger.info("=" * 60)
    
    x_data = test_data["x_data"]
    y_data = test_data["y_data"]
    semantic_features = test_data["semantic_features"]
    prior_config = test_data["prior_config"]
    device = x_data.device
    
    # Create mock priors
    prior_bag = MagicMock()
    
    # Create original adapter
    orig_adapter = ClassificationAdapter(prior_bag, prior_config)
    
    # Create optimized adapter
    opt_adapter = ClassificationAdapterOptimized(prior_bag, prior_config)
    
    # Patch get_random_semantic_data to return consistent data
    mock_semantic_data = torch.randint(1000, 2000, (5, 50))
    mock_names = [f"feature_{i}" for i in range(5)]
    
    # Run original implementation
    original_times = []
    with patch('ticl.datasets.semantic_prior_data_loader.get_random_semantic_data',
              return_value=(mock_semantic_data, mock_names)):
        for i in range(iterations):
            logger.info(f"Original implementation - iteration {i+1}/{iterations}")
            # Create fresh copy of data to avoid caching effects
            x_copy = x_data.clone()
            
            with timing("Original implementation") as timer:
                x_new, info = orig_adapter._apply_semantic_prior(
                    x_copy,
                    semantic_features,
                    device,
                    y_data
                )
            original_times.append(timer.elapsed)
    
    # Run optimized implementation
    optimized_times = []
    with patch('ticl.datasets.semantic_prior_data_loader.get_random_semantic_data',
              return_value=(mock_semantic_data, mock_names)):
        for i in range(iterations):
            logger.info(f"Optimized implementation - iteration {i+1}/{iterations}")
            # Create fresh copy of data to avoid caching effects
            x_copy = x_data.clone()
            
            with timing("Optimized implementation") as timer:
                x_new_opt, info_opt = opt_adapter._apply_semantic_prior(
                    x_copy,
                    semantic_features,
                    device,
                    y_data
                )
            optimized_times.append(timer.elapsed)
    
    # Calculate statistics
    avg_original = sum(original_times) / len(original_times)
    avg_optimized = sum(optimized_times) / len(optimized_times)
    improvement = (avg_original - avg_optimized) / avg_original * 100
    
    # Print detailed stats if available
    if hasattr(opt_adapter, 'get_performance_stats'):
        perf_stats = opt_adapter.get_performance_stats()
        logger.info("Detailed performance breakdown:")
        for key, value in perf_stats.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
    
    logger.info(f"Original implementation: {avg_original:.4f}s average")
    logger.info(f"Optimized implementation: {avg_optimized:.4f}s average")
    logger.info(f"Improvement: {improvement:.2f}%")
    
    return {
        "original_times": original_times,
        "optimized_times": optimized_times,
        "avg_original": avg_original,
        "avg_optimized": avg_optimized,
        "improvement": improvement
    }

def main():
    """Run all comparison tests"""
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() 
                         else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() 
                         else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Setup test data
    test_data = setup_test_data(device)
    
    # Create test JSON file
    test_json = create_test_json("test_columns.json", num_columns=30)
    
    try:
        # Compare loading semantic prior data
        load_results = compare_load_semantic_prior_data(test_json, iterations=3)
        
        # Compare random semantic data generation
        random_results = compare_get_random_semantic_data(iterations=3)
        
        # Compare semantic prior application
        prior_results = compare_apply_semantic_prior(test_data, iterations=5)
        
        # Print summary of all results
        logger.info("=" * 60)
        logger.info("OPTIMIZATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"load_semantic_prior_data: {load_results['improvement']:.2f}% improvement")
        logger.info(f"get_random_semantic_data: {random_results['improvement']:.2f}% improvement")
        logger.info(f"_apply_semantic_prior: {prior_results['improvement']:.2f}% improvement")
        
    finally:
        # Cleanup
        if os.path.exists(test_json):
            os.remove(test_json)

if __name__ == "__main__":
    main()