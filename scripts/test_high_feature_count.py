import torch
import time
import logging
import argparse
import numpy as np
import psutil
import os
import gc
from torch.cuda import memory_allocated, memory_reserved, memory_summary

from ticl.priors.classification_adapter import ClassificationAdapter
from ticl.distributions import sample_distributions, parse_distributions

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DummyPrior:
    """Simple dummy prior that generates random data with specified features"""
    def __init__(self):
        pass
        
    def get_batch(self, batch_size, n_samples, num_features, device, **kwargs):
        # Generate random inputs and outputs 
        x = torch.randn(n_samples, batch_size, num_features, device=device)
        y = torch.randint(0, 5, (n_samples, batch_size, 1), device=device).float()
        return x, y, y

def track_memory_usage(device='cpu'):
    """Track memory usage for the current process"""
    process = psutil.Process(os.getpid())
    cpu_memory = process.memory_info().rss / (1024 * 1024)  # MB
    
    gpu_memory = 0
    if device == 'cuda' and torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
        
    return {
        'cpu_memory_mb': cpu_memory,
        'gpu_memory_mb': gpu_memory
    }

def test_high_feature_count(
    feature_counts=[50, 100, 150, 200, 250],
    batch_size=16, 
    n_samples=100, 
    semantic_feature_p=0.5,
    num_classes=5,
    device='cpu', 
    num_runs=3
):
    """
    Test performance and memory usage with various feature counts
    and semantic features enabled.
    """
    logger.info(f"Testing with {len(feature_counts)} different feature counts")
    logger.info(f"Batch size: {batch_size}, Samples: {n_samples}, Device: {device}")
    logger.info(f"Semantic feature probability: {semantic_feature_p}")
    
    results = {}
    
    for num_features in feature_counts:
        logger.info(f"\n{'='*40}\nTesting with {num_features} features\n{'='*40}")
        
        # Create a base prior for testing
        base_prior = DummyPrior()
        
        # Configuration for the classification adapter with semantic features
        config = {
            'num_features_sampler': 'uniform',
            'feature_curriculum': False,
            'pad_zeros': False,
            'balanced': False,
            'multiclass_type': 'rank',
            'output_multiclass_ordered_p': 0.5,
            'multiclass_max_steps': 5,
            'nan_prob_a_reason': 0.0,
            'nan_prob_no_reason': 0.0,
            'categorical_feature_p': 0.0,
            'max_num_classes': 10,
            'num_classes': num_classes,
            'set_value_to_nan': 'nan',
            'semantic_feature_p': semantic_feature_p,
            'random_seed': 42
        }
        
        # Initialize the adapter
        adapter = ClassificationAdapter(base_prior, config)
        
        # Parameters for the test
        single_eval_pos = n_samples // 2  # Split point for train/test
        
        # Record initial memory usage
        initial_memory = track_memory_usage(device)
        logger.info(f"Initial memory usage: {initial_memory['cpu_memory_mb']:.2f} MB CPU, {initial_memory['gpu_memory_mb']:.2f} MB GPU")
        
        # Warm-up run
        logger.info("Running warm-up batch...")
        try:
            _ = adapter(batch_size, n_samples, num_features, device, single_eval_pos=single_eval_pos)
            logger.info("Warm-up completed successfully")
        except RuntimeError as e:
            logger.error(f"Error during warm-up: {str(e)}")
            results[num_features] = {
                'status': 'error',
                'error': str(e),
                'stage': 'warm-up'
            }
            continue
            
        # Force garbage collection
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
        
        # Record post-warmup memory usage
        post_warmup_memory = track_memory_usage(device)
        logger.info(f"Memory after warm-up: {post_warmup_memory['cpu_memory_mb']:.2f} MB CPU, {post_warmup_memory['gpu_memory_mb']:.2f} MB GPU")
        logger.info(f"Memory increase: {post_warmup_memory['cpu_memory_mb'] - initial_memory['cpu_memory_mb']:.2f} MB CPU, {post_warmup_memory['gpu_memory_mb'] - initial_memory['gpu_memory_mb']:.2f} MB GPU")
        
        # Time the performance
        times = []
        max_memory = {'cpu_memory_mb': 0, 'gpu_memory_mb': 0}
        tensor_shapes = []
        run_status = 'success'
        run_error = None
        
        logger.info(f"Running {num_runs} performance tests...")
        
        for i in range(num_runs):
            try:
                # Force GC to avoid accumulating memory
                gc.collect()
                if device == 'cuda':
                    torch.cuda.empty_cache()
                
                # Measure time
                start_time = time.time()
                x, y, _, info = adapter(batch_size, n_samples, num_features, device, single_eval_pos=single_eval_pos)
                end_time = time.time()
                
                # Measure memory right after run
                current_memory = track_memory_usage(device)
                max_memory['cpu_memory_mb'] = max(max_memory['cpu_memory_mb'], current_memory['cpu_memory_mb'])
                max_memory['gpu_memory_mb'] = max(max_memory['gpu_memory_mb'], current_memory['gpu_memory_mb'])
                
                # Record run time
                run_time = end_time - start_time
                times.append(run_time)
                
                # Record output tensor shape
                tensor_shapes.append(x.shape)
                
                logger.info(f"Run {i+1}/{num_runs}: {run_time:.4f} seconds")
                logger.info(f"  - Output tensor shape: {x.shape}")
                logger.info(f"  - Semantic features: {len(info['semantic_features'])} features")
                logger.info(f"  - Memory usage: {current_memory['cpu_memory_mb']:.2f} MB CPU, {current_memory['gpu_memory_mb']:.2f} MB GPU")
                
                # Check tensor properties
                if info['semantic_features']:
                    nonzero_semantic = (x[:, :, info['semantic_features']] != 0).sum().item()
                    total_semantic = np.prod(x[:, :, info['semantic_features']].shape)
                    logger.info(f"  - Semantic feature fill rate: {nonzero_semantic/total_semantic*100:.1f}%")
                
            except RuntimeError as e:
                logger.error(f"Error during run {i+1}: {str(e)}")
                run_status = 'error'
                run_error = str(e)
                break
        
        # Calculate statistics
        if times:
            avg_time = np.mean(times)
            std_dev = np.std(times) if len(times) > 1 else 0
            
            logger.info(f"Average time per run: {avg_time:.4f} seconds (±{std_dev:.4f})")
            logger.info(f"Max memory usage: {max_memory['cpu_memory_mb']:.2f} MB CPU, {max_memory['gpu_memory_mb']:.2f} MB GPU")
            
            # Store results
            results[num_features] = {
                'status': run_status,
                'avg_time': avg_time,
                'std_dev': std_dev,
                'tensor_shapes': tensor_shapes,
                'max_memory': max_memory,
                'memory_increase': {
                    'cpu_memory_mb': max_memory['cpu_memory_mb'] - initial_memory['cpu_memory_mb'],
                    'gpu_memory_mb': max_memory['gpu_memory_mb'] - initial_memory['gpu_memory_mb']
                }
            }
        else:
            results[num_features] = {
                'status': run_status,
                'error': run_error
            }
    
    # Display summary of results
    logger.info("\n\n" + "="*60)
    logger.info("SUMMARY OF HIGH FEATURE COUNT TESTS")
    logger.info("="*60)
    
    logger.info(f"{'Features':<10} {'Status':<10} {'Avg Time (s)':<15} {'CPU Memory (MB)':<20} {'GPU Memory (MB)':<20}")
    logger.info("-"*75)
    
    for num_features in feature_counts:
        result = results[num_features]
        status = result.get('status', 'unknown')
        
        if status == 'success':
            avg_time = f"{result['avg_time']:.4f}"
            cpu_memory = f"{result['max_memory']['cpu_memory_mb']:.2f}"
            gpu_memory = f"{result['max_memory']['gpu_memory_mb']:.2f}"
        else:
            avg_time = "N/A"
            cpu_memory = "N/A"
            gpu_memory = "N/A"
            
        logger.info(f"{num_features:<10} {status:<10} {avg_time:<15} {cpu_memory:<20} {gpu_memory:<20}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test high feature counts with semantic features')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--n-samples', type=int, default=100, help='Number of samples')
    parser.add_argument('--feature-min', type=int, default=50, help='Minimum number of features to test')
    parser.add_argument('--feature-max', type=int, default=300, help='Maximum number of features to test')
    parser.add_argument('--feature-step', type=int, default=50, help='Step size for feature count')
    parser.add_argument('--semantic-feature-p', type=float, default=0.5, help='Probability of using semantic features')
    parser.add_argument('--num-classes', type=int, default=5, help='Number of classes')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='Device to use')
    parser.add_argument('--num-runs', type=int, default=3, help='Number of runs to average')
    
    args = parser.parse_args()
    
    # Use 'cuda' if available and requested
    if args.device == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
    elif args.device == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    # Generate feature counts to test
    feature_counts = list(range(args.feature_min, args.feature_max + 1, args.feature_step))
    
    # Run the test
    test_high_feature_count(
        feature_counts=feature_counts,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
        semantic_feature_p=args.semantic_feature_p,
        num_classes=args.num_classes,
        device=device,
        num_runs=args.num_runs
    )