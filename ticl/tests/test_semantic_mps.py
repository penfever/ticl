#!/usr/bin/env python3
"""
Simplified script for profiling TabPFN model performance with and without semantic features.
This standalone script measures performance differences on MPS devices.
"""

import os
import sys
import time
import torch
import numpy as np
import cProfile
import pstats
from datetime import datetime
from io import StringIO
import argparse
import logging

# Setup logging
log_dir = "logs/profiling"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(f"{log_dir}/mps_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add ticl to path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import ticl components
try:
    from ticl.models.semantic_aware_model import SemanticConsistencyLoss, SemanticAwareClassifier, create_semantic_aware_model
    from ticl.models.tabpfn import TabPFN
    from ticl.models.encoders import Linear
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Make sure ticl is in your PYTHONPATH")
    sys.exit(1)

def setup_model(emsize=128, nlayers=2, semantic_feature_p=0.0, device="cpu"):
    """Set up a TabPFN model with or without semantic features."""
    
    # Base TabPFN config
    config = {
        "emsize": emsize, 
        "nhead": 2,
        "d_inner": 512,
        "nlayers": nlayers,
        "dropout": 0.0,
        "n_out": 5,  # Number of output classes
        "prior": {
            "type": "classification",
            "classification": {
                "num_features_sampler": "uniform",
                "feature_curriculum": False,
                "pad_zeros": False,
                "balanced": False,
                "multiclass_type": "rank",
                "output_multiclass_ordered_p": 0.5,
                "multiclass_max_steps": 5,
                "nan_prob_a_reason": 0.0,
                "nan_prob_no_reason": 0.0,
                "categorical_feature_p": 0.0,
                "max_num_classes": 10,
                "num_classes": 5,
                "set_value_to_nan": "nan",
                "semantic_feature_p": semantic_feature_p,
                "random_seed": 42
            }
        }
    }
    
    # Create base TabPFN model
    model = TabPFN(
        n_out=config["n_out"],
        emsize=config["emsize"], 
        nhead=config["nhead"],
        nhid_factor=config["d_inner"] // config["emsize"], 
        nlayers=config["nlayers"],
        n_features=20,  # For simple profiling, fixed size
        dropout=config["dropout"],
        classification_task=True,
        semantic_feature_p=semantic_feature_p,
        config=config,
        y_encoder_layer=Linear(1, config["emsize"])  # Required for TabPFN
    ).to(device)
    
    # Apply semantic wrapper if required
    if semantic_feature_p > 0.0:
        # Wrap in semantic-aware model
        model = create_semantic_aware_model(model, num_semantic_classes=5, freeze_clip=False)
        criterion = SemanticConsistencyLoss(semantic_weight=0.2)
    else:
        # Standard cross-entropy loss
        criterion = torch.nn.CrossEntropyLoss()
    
    # Return model, criterion, and config
    return model, criterion, config

def get_dummy_batch(n_samples=20, batch_size=4, num_features=20, device="cpu", 
                   semantic_feature_p=0.0):
    """Generate a dummy batch for profiling."""
    
    # Generate random data in typical TabPFN format
    x = torch.randn(n_samples, batch_size, num_features, device=device)
    y = torch.randint(0, 5, (n_samples, batch_size, 1), device=device).float()
    
    # Create eval positions
    single_eval_pos = n_samples // 2
    
    # If semantic features enabled, add semantic data
    if semantic_feature_p > 0.0:
        # Create batch info with semantic data
        batch_info = {
            "semantic_features": list(range(num_features-10, num_features)),
            "semantic_targets": torch.randint(0, 5, (batch_size,), device=device)
        }
        
        # Add class token patterns to simulate CLIP encoder inputs using commonly used CLIP token IDs
        batch_info["class_token_patterns"] = {}
        for i in range(5):  # 5 classes
            # Use specific token IDs from CLIP's vocabulary to avoid issues with random tokens
            # Start with CLIP's BOS token (49406), then use common token IDs, and end with EOS token (49407)
            # Use long datatype explicitly for token IDs
            tokens = torch.tensor([49406] + [i+1000 for _ in range(10)] + [49407] + [0]*37, 
                                 dtype=torch.long, device=device)
            
            batch_info["class_token_patterns"][i] = {
                "semantic_class": i,
                "class_name": f"Class {i}",
                "column_name": f"feature_{i}",
                "tokens": tokens
            }
        
        return (x, y), y, single_eval_pos, batch_info
    else:
        # Standard data without semantic features
        return (x, y), y, single_eval_pos, {}

def profile_forward_pass(model, batch, semantic_feature_p=0.0, n_iterations=10, output_file=None):
    """Profile forward pass through the model."""
    
    # Extract batch data
    (x, y), targets, single_eval_pos, batch_info = batch
    
    # Get device for checking MPS compatibility
    device = x.device
    is_mps = device.type == 'mps'
    
    # Add workaround for MPS with CLIP if needed
    if is_mps and semantic_feature_p > 0.0:
        logger.info("Using MPS device with semantic features, adding CLIP compatibility layer")
        # Modify the model to handle MPS-specific issues with CLIP
        if hasattr(model, 'clip_text_model'):
            # Try to move CLIP model to CPU for text encoding
            try:
                model.clip_text_model_device = model.clip_text_model.device
                model.original_process_semantic_tokens = model._process_semantic_tokens
                
                # Monkey patch with a version that moves tensors to CPU for CLIP processing
                def mps_safe_process_tokens(self, semantic_tokens):
                    # Move tokens to CPU for processing
                    cpu_tokens = semantic_tokens.to('cpu')
                    
                    # Create attention mask on CPU
                    attention_mask = (cpu_tokens != -100).long()
                    
                    # Replace -100 values with pad token IDs for the CLIP model
                    input_ids = torch.where(cpu_tokens == -100, 
                                         torch.tensor(self.tokenizer.pad_token_id, device='cpu'), 
                                         cpu_tokens)
                    
                    # Ensure proper shapes                 
                    if attention_mask.dim() == 1:
                        attention_mask = attention_mask.unsqueeze(0)
                        
                    if input_ids.dim() == 1:
                        input_ids = input_ids.unsqueeze(0)
                    
                    # Create input dict for CLIP
                    token_dict = {
                        'input_ids': input_ids,
                        'attention_mask': attention_mask
                    }
                    
                    # Process with CLIP text encoder on CPU
                    with torch.set_grad_enabled(False):
                        outputs = self.clip_text_model(**token_dict)
                    
                    # Move results back to original device
                    return outputs.pooler_output.to(device)
                
                # Replace the method for safe MPS operation
                import types
                model._process_semantic_tokens = types.MethodType(mps_safe_process_tokens, model)
                logger.info("Added MPS compatibility wrapper for CLIP processing")
            except Exception as e:
                logger.warning(f"Failed to add MPS compatibility for CLIP: {e}")
    
    # Warm-up iterations
    logger.info(f"Running {2} warm-up iterations...")
    for _ in range(2):
        try:
            if semantic_feature_p > 0.0:
                # Semantic model forward pass
                outputs = model((x, y), single_eval_pos=single_eval_pos, batch_info=batch_info)
            else:
                # Standard model forward pass
                outputs = model((x, y), single_eval_pos=single_eval_pos)
        except Exception as e:
            logger.warning(f"Warning: Warm-up iteration failed: {e}")
            # Continue despite errors in warm-up
    
    # Set up profiler
    pr = cProfile.Profile()
    
    # Run profiled iterations
    logger.info(f"Running {n_iterations} profiled iterations...")
    pr.enable()
    start_time = time.time()
    
    for i in range(n_iterations):
        if semantic_feature_p > 0.0:
            # Semantic model forward pass
            outputs = model((x, y), single_eval_pos=single_eval_pos, batch_info=batch_info)
        else:
            # Standard model forward pass
            outputs = model((x, y), single_eval_pos=single_eval_pos)
        
        # Clear cache periodically to prevent OOM
        if i % 5 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    elapsed_time = time.time() - start_time
    pr.disable()
    
    # Calculate average time per iteration
    avg_time = elapsed_time / n_iterations
    
    # Save profiling results
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        pr.dump_stats(output_file)
    
    # Create string buffer for stats
    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
    ps.print_stats(20)  # Top 20 functions
    stats_text = s.getvalue()
    
    return avg_time, stats_text, pr

def format_stats(stats_text, top_n=10):
    """Format profiler output for better readability."""
    lines = stats_text.strip().split('\n')
    # Extract header and stats
    header = lines[0]
    if len(lines) > 4:
        stats = lines[4:4+top_n]
        return '\n'.join([header] + stats)
    return stats_text

def compare_profiles(baseline_profile, semantic_profile, baseline_time, semantic_time, num_functions=10):
    """Compare two profiles to identify performance differences."""
    baseline_stats = pstats.Stats(baseline_profile)
    semantic_stats = pstats.Stats(semantic_profile)
    
    # Get functions sorted by cumulative time
    baseline_functions = [(func, stats) for func, stats in baseline_stats.stats.items()]
    baseline_functions.sort(key=lambda x: x[1][3], reverse=True)  # Sort by cumtime
    
    semantic_functions = [(func, stats) for func, stats in semantic_stats.stats.items()]
    semantic_functions.sort(key=lambda x: x[1][3], reverse=True)  # Sort by cumtime
    
    # Print comparison header
    logger.info(f"{'=' * 80}")
    logger.info(f"PERFORMANCE COMPARISON: Baseline ({baseline_time:.4f}s) vs Semantic ({semantic_time:.4f}s)")
    logger.info(f"Slowdown factor: {semantic_time/baseline_time:.2f}x")
    logger.info(f"{'=' * 80}")
    
    # Print top functions from each profile
    logger.info(f"\nTop {num_functions} functions by cumtime in BASELINE run:")
    logger.info(f"{'Function':<50} {'Calls':<10} {'Time (s)':<12} {'Per call (ms)':<15} {'% Total':<10}")
    logger.info(f"{'-' * 100}")
    
    for i, (func, stats) in enumerate(baseline_functions[:num_functions]):
        # Extract stats (nc=ncalls, cc=primitive calls, tt=tottime, ct=cumtime)
        nc, cc, tt, ct, callers = stats
        func_name = f"{func[2]}:{func[1]}({func[0]})" 
        if len(func_name) > 50:
            func_name = "..." + func_name[-47:]
        percent = ct / baseline_time * 100
        logger.info(f"{func_name:<50} {nc:<10} {ct:12.4f} {ct*1000/nc:15.4f} {percent:9.1f}%")
    
    logger.info(f"\nTop {num_functions} functions by cumtime in SEMANTIC run:")
    logger.info(f"{'Function':<50} {'Calls':<10} {'Time (s)':<12} {'Per call (ms)':<15} {'% Total':<10}")
    logger.info(f"{'-' * 100}")
    
    for i, (func, stats) in enumerate(semantic_functions[:num_functions]):
        # Extract stats
        nc, cc, tt, ct, callers = stats
        func_name = f"{func[2]}:{func[1]}({func[0]})"
        if len(func_name) > 50:
            func_name = "..." + func_name[-47:]
        percent = ct / semantic_time * 100
        logger.info(f"{func_name:<50} {nc:<10} {ct:12.4f} {ct*1000/nc:15.4f} {percent:9.1f}%")
    
    # Find functions with biggest difference in cumulative time
    logger.info(f"\nFunctions with biggest ABSOLUTE time difference (semantic - baseline):")
    logger.info(f"{'Function':<50} {'Semantic (s)':<15} {'Baseline (s)':<15} {'Diff (s)':<12} {'Ratio':<10}")
    logger.info(f"{'-' * 100}")
    
    # Combine all functions from both profiles
    all_funcs = set([func for func, _ in baseline_functions] + [func for func, _ in semantic_functions])
    time_diffs = []
    
    for func in all_funcs:
        base_time = baseline_stats.stats.get(func, [0, 0, 0, 0, {}])[3]  # cumtime
        sem_time = semantic_stats.stats.get(func, [0, 0, 0, 0, {}])[3]   # cumtime
        diff = sem_time - base_time
        
        # Skip very small differences
        if abs(diff) < 0.01:
            continue
            
        ratio = sem_time / max(base_time, 0.0001)  # Avoid division by zero
        time_diffs.append((func, sem_time, base_time, diff, ratio))
    
    # Sort by absolute difference
    time_diffs.sort(key=lambda x: abs(x[3]), reverse=True)
    
    for i, (func, sem_time, base_time, diff, ratio) in enumerate(time_diffs[:num_functions]):
        func_name = f"{func[2]}:{func[1]}({func[0]})"
        if len(func_name) > 50:
            func_name = "..." + func_name[-47:]
        logger.info(f"{func_name:<50} {sem_time:15.4f} {base_time:15.4f} {diff:12.4f} {ratio:10.2f}x")
    
    # Find functions with biggest ratio of time increase
    logger.info(f"\nFunctions with biggest RATIO increase (semantic/baseline, min 0.1s in semantic):")
    logger.info(f"{'Function':<50} {'Semantic (s)':<15} {'Baseline (s)':<15} {'Diff (s)':<12} {'Ratio':<10}")
    logger.info(f"{'-' * 100}")
    
    # Filter for functions that take significant time and have high ratio
    significant_diffs = [x for x in time_diffs if x[1] > 0.1]  # At least 0.1s in semantic
    significant_diffs.sort(key=lambda x: x[4], reverse=True)  # Sort by ratio
    
    for i, (func, sem_time, base_time, diff, ratio) in enumerate(significant_diffs[:num_functions]):
        func_name = f"{func[2]}:{func[1]}({func[0]})"
        if len(func_name) > 50:
            func_name = "..." + func_name[-47:]
        logger.info(f"{func_name:<50} {sem_time:15.4f} {base_time:15.4f} {diff:12.4f} {ratio:10.2f}x")

def main(args):
    """Main function for profiling model performance."""
    logger.info(f"Starting performance profiling with args: {args}")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Get device
    if args.device == 'auto':
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    else:
        device = args.device
    logger.info(f"Using device: {device}")
    
    # Profile baseline (no semantic features)
    logger.info(f"{'=' * 80}")
    logger.info(f"PROFILING BASELINE MODEL (semantic_feature_p=0.0)")
    logger.info(f"{'=' * 80}")
    
    baseline_model, baseline_criterion, baseline_config = setup_model(
        emsize=args.emsize,
        nlayers=args.nlayers,
        semantic_feature_p=0.0,
        device=device
    )
    
    # Generate dummy batch for baseline
    baseline_batch = get_dummy_batch(
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        num_features=args.num_features,
        device=device,
        semantic_feature_p=0.0
    )
    
    # Warm-up and profile baseline
    baseline_profile_file = f"{log_dir}/baseline_mps_profile.prof"
    baseline_time, baseline_stats, baseline_profile = profile_forward_pass(
        baseline_model,
        baseline_batch,
        semantic_feature_p=0.0,
        n_iterations=args.n_iterations,
        output_file=baseline_profile_file
    )
    
    logger.info(f"Baseline profile saved to {baseline_profile_file}")
    logger.info(f"Baseline average iteration time: {baseline_time:.4f}s")
    logger.info(f"Baseline top functions:\n{format_stats(baseline_stats)}")
    
    # Now profile with semantic features
    logger.info(f"\n{'=' * 80}")
    logger.info(f"PROFILING SEMANTIC MODEL (semantic_feature_p={args.semantic_feature_p})")
    logger.info(f"{'=' * 80}")
    
    semantic_model, semantic_criterion, semantic_config = setup_model(
        emsize=args.emsize,
        nlayers=args.nlayers,
        semantic_feature_p=args.semantic_feature_p,
        device=device
    )
    
    # Generate dummy batch for semantic
    semantic_batch = get_dummy_batch(
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        num_features=args.num_features,
        device=device,
        semantic_feature_p=args.semantic_feature_p
    )
    
    # Profile semantic
    semantic_profile_file = f"{log_dir}/semantic_mps_profile.prof"
    try:
        semantic_time, semantic_stats, semantic_profile = profile_forward_pass(
            semantic_model,
            semantic_batch,
            semantic_feature_p=args.semantic_feature_p,
            n_iterations=args.n_iterations,
            output_file=semantic_profile_file
        )
        
        logger.info(f"Semantic profile saved to {semantic_profile_file}")
        logger.info(f"Semantic average iteration time: {semantic_time:.4f}s")
        logger.info(f"Semantic top functions:\n{format_stats(semantic_stats)}")
        
        # Compare profiles
        compare_profiles(
            baseline_profile, 
            semantic_profile, 
            baseline_time * args.n_iterations, 
            semantic_time * args.n_iterations,
            num_functions=args.num_functions
        )
        
        # Final summary
        logger.info(f"\n{'=' * 80}")
        logger.info(f"PERFORMANCE SUMMARY")
        logger.info(f"{'=' * 80}")
        logger.info(f"Baseline average iteration time:  {baseline_time:.4f}s")
        logger.info(f"Semantic average iteration time:  {semantic_time:.4f}s")
        logger.info(f"Slowdown factor:                  {semantic_time/baseline_time:.2f}x")
        
        # Save summary to file
        summary_file = f"{log_dir}/mps_performance_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"PERFORMANCE SUMMARY\n")
            f.write(f"{'=' * 80}\n")
            f.write(f"Baseline average iteration time:  {baseline_time:.4f}s\n")
            f.write(f"Semantic average iteration time:  {semantic_time:.4f}s\n")
            f.write(f"Slowdown factor:                  {semantic_time/baseline_time:.2f}x\n")
            f.write(f"Profile run at:                   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Parameters:\n")
            f.write(f"  device:            {device}\n")
            f.write(f"  batch_size:        {args.batch_size}\n")
            f.write(f"  n_samples:         {args.n_samples}\n")
            f.write(f"  num_features:      {args.num_features}\n")
            f.write(f"  emsize:            {args.emsize}\n")
            f.write(f"  nlayers:           {args.nlayers}\n")
            f.write(f"  n_iterations:      {args.n_iterations}\n")
            f.write(f"  semantic_feature_p: {args.semantic_feature_p}\n\n")
            
            # Add profile file paths
            f.write(f"Profile files:\n")
            f.write(f"  Baseline profile:  {baseline_profile_file}\n")
            f.write(f"  Semantic profile:  {semantic_profile_file}\n\n")
            
            # Add command to view profiles
            f.write(f"View profiles with:\n")
            f.write(f"  python -m pstats {baseline_profile_file}\n")
            f.write(f"  python -m pstats {semantic_profile_file}\n")
        
        logger.info(f"Performance summary saved to {summary_file}")
        
    except Exception as e:
        logger.error(f"Error in semantic profiling: {e}")
        import traceback
        logger.error(traceback.format_exc())
        if args.continue_on_error:
            logger.info("Continuing despite error...")
        else:
            raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile TabPFN with semantic features on MPS")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--n-samples", type=int, default=20, help="Number of samples")
    parser.add_argument("--num-features", type=int, default=20, help="Number of features")
    parser.add_argument("--emsize", type=int, default=128, help="Embedding size")
    parser.add_argument("--nlayers", type=int, default=2, help="Number of layers")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"], 
                       help="Device to use")
    parser.add_argument("--semantic-feature-p", type=float, default=0.3, 
                       help="Probability of using semantic features")
    parser.add_argument("--n-iterations", type=int, default=10, 
                       help="Number of iterations to profile")
    parser.add_argument("--num-functions", type=int, default=15, 
                       help="Number of top functions to analyze")
    parser.add_argument("--continue-on-error", action="store_true", 
                       help="Continue execution if semantic model fails")
    
    args = parser.parse_args()
    main(args)