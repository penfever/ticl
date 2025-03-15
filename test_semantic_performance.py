import torch
import time
import logging
import numpy as np
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
        y = torch.randn(n_samples, batch_size, 1, device=device)
        return x, y, y

def test_semantic_prior_performance():
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
        'num_classes': 5,
        'set_value_to_nan': 'nan',
        'semantic_feature_p': 0.5,  # Enable semantic features
        'random_seed': 42
    }
    
    # Initialize the adapter
    adapter = ClassificationAdapter(base_prior, config)
    
    # Parameters for the test
    batch_size = 16
    n_samples = 100
    num_features = 20
    device = 'cpu'  # Change to 'cuda' if available
    single_eval_pos = n_samples // 2  # Split point for train/test
    
    # Warm-up run
    logger.info("Running warm-up batch...")
    _ = adapter(batch_size, n_samples, num_features, device, single_eval_pos=single_eval_pos)
    
    # Time the performance
    num_runs = 5
    total_time = 0
    
    logger.info(f"Running {num_runs} performance tests...")
    
    for i in range(num_runs):
        start_time = time.time()
        x, y, _, info = adapter(batch_size, n_samples, num_features, device, single_eval_pos=single_eval_pos)
        end_time = time.time()
        
        run_time = end_time - start_time
        total_time += run_time
        
        logger.info(f"Run {i+1}/{num_runs}: {run_time:.4f} seconds")
        logger.info(f"  - Output tensor shape: {x.shape}")
        logger.info(f"  - Semantic features: {len(info['semantic_features'])} features")
        logger.info(f"  - Number of classes: {config['num_classes']}")
    
    avg_time = total_time / num_runs
    logger.info(f"Average time per run: {avg_time:.4f} seconds")
    
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
    test_semantic_prior_performance()