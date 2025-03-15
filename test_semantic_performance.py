import torch
import time
import logging
import argparse
import numpy as np
import cProfile
import pstats
from ticl.priors.classification_adapter import ClassificationAdapter
from ticl.distributions import sample_distributions, parse_distributions

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DummyPrior:
    """Simple dummy prior that generates random data"""
    def __init__(self):
        pass
        
    def get_batch(self, batch_size, n_samples, num_features, device, **kwargs):
        # Generate random inputs and outputs 
        x = torch.randn(n_samples, batch_size, num_features, device=device)
        y = torch.randint(0, 5, (n_samples, batch_size, 1), device=device).float()
        return x, y, y

def test_semantic_prior_performance(batch_size=16, 
                                   n_samples=100, 
                                   num_features=20, 
                                   num_classes=5,
                                   device='cpu', 
                                   profile=False,
                                   num_runs=5):
    """Test performance of semantic prior feature generation"""
    
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
        'semantic_feature_p': 0.5,  # Enable semantic features
        'random_seed': 42
    }
    
    # Initialize the adapter
    adapter = ClassificationAdapter(base_prior, config)
    
    # Parameters for the test
    single_eval_pos = n_samples // 2  # Split point for train/test
    
    # Warm-up run
    logger.info("Running warm-up batch...")
    _ = adapter(batch_size, n_samples, num_features, device, single_eval_pos=single_eval_pos)
    
    # Profile if requested
    if profile:
        logger.info("Running with profiler...")
        profiler = cProfile.Profile()
        profiler.enable()
        
        # Do a single run with profiling
        x, y, _, info = adapter(batch_size, n_samples, num_features, device, single_eval_pos=single_eval_pos)
        
        profiler.disable()
        
        # Create stats object and sort by cumulative time
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        
        # Print the 20 functions that took the most cumulative time
        logger.info("Top 20 functions by cumulative time:")
        stats.print_stats(20)
        
        # Save profile results to file
        stats.dump_stats('semantic_prior_profile.prof')
        logger.info("Profile saved to 'semantic_prior_profile.prof'")
        
        return
    
    # Time the performance
    total_time = 0
    times = []
    
    logger.info(f"Running {num_runs} performance tests...")
    
    for i in range(num_runs):
        # Force GC to avoid measuring garbage collection time
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        start_time = time.time()
        x, y, _, info = adapter(batch_size, n_samples, num_features, device, single_eval_pos=single_eval_pos)
        end_time = time.time()
        
        run_time = end_time - start_time
        total_time += run_time
        times.append(run_time)
        
        logger.info(f"Run {i+1}/{num_runs}: {run_time:.4f} seconds")
        logger.info(f"  - Output tensor shape: {x.shape}")
        logger.info(f"  - Semantic features: {len(info['semantic_features'])} features")
        logger.info(f"  - Number of classes: {config['num_classes']}")
    
    avg_time = total_time / num_runs
    std_dev = np.std(times) if len(times) > 1 else 0
    
    logger.info(f"Average time per run: {avg_time:.4f} seconds (±{std_dev:.4f})")
    
    # Check tensor properties
    nonzero_semantic = (x[:, :, info['semantic_features']] != 0).sum().item()
    total_semantic = np.prod(x[:, :, info['semantic_features']].shape)
    logger.info(f"Semantic feature fill rate: {nonzero_semantic}/{total_semantic} " 
                f"({nonzero_semantic/total_semantic*100:.1f}%)")
    
    # Check semantic targets
    if info['semantic_targets'] is not None:
        valid_targets = (info['semantic_targets'] != -100).sum().item()
        total_targets = info['semantic_targets'].numel()
        logger.info(f"Valid semantic targets: {valid_targets}/{total_targets} "
                    f"({valid_targets/total_targets*100:.1f}%)")
    
    return avg_time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test semantic prior performance')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--n-samples', type=int, default=100, help='Number of samples')
    parser.add_argument('--num-features', type=int, default=20, help='Number of base features')
    parser.add_argument('--num-classes', type=int, default=5, help='Number of classes')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], 
                       help='Device to use')
    parser.add_argument('--profile', action='store_true', help='Run with profiler')
    parser.add_argument('--num-runs', type=int, default=5, help='Number of runs to average')
    
    args = parser.parse_args()
    
    # Use 'cuda' if available and requested
    if args.device == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
    elif args.device == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    # Run the test
    test_semantic_prior_performance(
        batch_size=args.batch_size,
        n_samples=args.n_samples,
        num_features=args.num_features,
        num_classes=args.num_classes,
        device=device,
        profile=args.profile,
        num_runs=args.num_runs
    )