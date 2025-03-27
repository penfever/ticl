#!/usr/bin/env python3
"""
Script for profiling TabPFN model performance with and without semantic features.
This script measures performance differences and identifies bottlenecks.
"""

import os
import sys
import time
import torch
import cProfile
import pstats
import argparse
import logging
import numpy as np
from datetime import datetime
from functools import partial
from io import StringIO

# Setup logging
log_dir = "logs/profiling"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(f"{log_dir}/performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add ticl to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import ticl components
from ticl.models.tabpfn import TabPFN
from ticl.models.semantic_aware_model import create_semantic_aware_model
from ticl.priors.classification_adapter import ClassificationAdapter
from ticl.fit_model import main as fit_model_main
from ticl.train import train, train_epoch
from ticl.models.semantic_aware_model import SemanticConsistencyLoss

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
    
    # Create simple Linear encoder for y values
    from ticl.models.encoders import Linear
    
    # Create base TabPFN model with required y_encoder
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
        y_encoder_layer=Linear(1, config["emsize"]) # Required for TabPFN
    ).to(device)
    
    # Apply semantic wrapper if required
    if semantic_feature_p > 0.0:
        # Wrap in semantic-aware model
        model = create_semantic_aware_model(model, freeze_clip=False)
        criterion = SemanticConsistencyLoss(semantic_weight=0.2)
    else:
        # Standard cross-entropy loss
        criterion = torch.nn.CrossEntropyLoss()
    
    # Return both model and criterion
    return model, criterion, config

def setup_dataloader(batch_size=4, n_samples=20, num_features=20, device="cpu",
                    config=None, num_batches=5):
    """Set up a dataloader for training."""
    
    # Simple dataloader class for testing
    class SimpleDataLoader:
        def __init__(self, batch_size, n_samples, num_features, device, config, num_batches):
            self.batch_size = batch_size
            self.n_samples = n_samples
            self.num_features = num_features
            self.device = device
            self.config = config
            self.num_batches = num_batches
            self.model = None  # Will be set by training loop
            
        def __len__(self):
            return self.num_batches
            
        def __iter__(self):
            # Iterator that returns random data
            for _ in range(self.num_batches):
                # Generate random data in typical TabPFN format
                x = torch.randn(self.n_samples, self.batch_size, self.num_features, device=self.device)
                y = torch.randint(0, 5, (self.n_samples, self.batch_size, 1), device=self.device).float()
                single_eval_pos = self.n_samples // 2
                
                # If semantic features enabled, add some dummy semantic data
                semantic_feature_p = self.config.get("prior", {}).get("classification", {}).get("semantic_feature_p", 0)
                if semantic_feature_p > 0:
                    # Create dummy batch info with semantic data
                    batch_info = {
                        "semantic_features": list(range(self.num_features-10, self.num_features)),
                        "semantic_targets": torch.randint(0, 5, (self.n_samples, self.batch_size), device=self.device)
                    }
                    # Set some values to -100 to simulate ignore indices
                    batch_info["semantic_targets"][0:5] = -100
                    
                    # Add class token patterns to simulate CLIP encoder inputs
                    batch_info["class_token_patterns"] = {}
                    for i in range(5):  # 5 classes
                        batch_info["class_token_patterns"][i] = {
                            "semantic_class": i,
                            "class_name": f"Class {i}",
                            "column_name": f"feature_{i}",
                            "tokens": torch.randint(1, 1000, (50,), device=self.device)  # Dummy tokens
                        }
                        
                    yield (x, y), y, single_eval_pos, batch_info
                else:
                    # Standard data without semantic features
                    yield (x, y), y, single_eval_pos
    
    # Return the simple dataloader
    return SimpleDataLoader(batch_size, n_samples, num_features, device, config, num_batches)

def profile_train_epoch(model, dl, criterion, optimizer, device, 
                      semantic_feature_p, output_file=None):
    """Profile a single training epoch."""
    
    # Set up profiler
    pr = cProfile.Profile()
    pr.enable()
    
    # Run a single training epoch
    epoch_start = time.time()
    loss, nan_share, ignore_share = train_epoch(
        model, 
        aggregate_k_gradients=1,
        using_dist=False,
        scaler=None,
        dl=dl,
        device=device,
        optimizer=optimizer,
        criterion=criterion,
        n_out=model.n_out if hasattr(model, 'n_out') else model.base_model.n_out,
        progress_bar=False,
        batch_monitor=None,
        semantic_batch_monitoring=False,
        skip_bad_semantic_batches=False,
        train_mixed_precision=False
    )
    epoch_time = time.time() - epoch_start
    
    # Disable profiler
    pr.disable()
    
    # Save and analyze profile results
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        pr.dump_stats(output_file)
    
    # Create a string buffer to capture output
    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
    ps.print_stats(20)  # Print top 20 functions
    
    # Extract the stats for analysis
    stats_text = s.getvalue()
    
    return loss, epoch_time, stats_text, pr

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
    logger.info(f"PERFORMANCE COMPARISON: Baseline ({baseline_time:.2f}s) vs Semantic ({semantic_time:.2f}s)")
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
    
    return {
        "baseline_time": baseline_time,
        "semantic_time": semantic_time,
        "slowdown_factor": semantic_time/baseline_time,
        "top_baseline": baseline_functions[:num_functions],
        "top_semantic": semantic_functions[:num_functions],
        "top_differences": time_diffs[:num_functions],
        "top_ratios": significant_diffs[:num_functions]
    }

def main(args):
    """Main function for profiling model performance."""
    logger.info(f"Starting performance profiling with args: {args}")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Get device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
    
    baseline_dl = setup_dataloader(
        batch_size=args.batch_size,
        n_samples=args.n_samples,
        num_features=args.num_features,
        device=device,
        config=baseline_config,
        num_batches=args.num_batches
    )
    
    # Create optimizer directly
    baseline_optimizer = torch.optim.AdamW(
        baseline_model.parameters(), 
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Warm-up run
    logger.info("Running baseline warm-up epoch...")
    _ = train_epoch(
        baseline_model, 
        aggregate_k_gradients=1,
        using_dist=False,
        scaler=None,
        dl=baseline_dl,
        device=device,
        optimizer=baseline_optimizer,
        criterion=baseline_criterion,
        n_out=baseline_model.n_out,
        progress_bar=False,
        batch_monitor=None,
        semantic_batch_monitoring=False,
        skip_bad_semantic_batches=False,
        train_mixed_precision=False
    )
    
    # Profile baseline
    logger.info("Profiling baseline epoch...")
    baseline_profile_file = f"{log_dir}/baseline_profile.prof"
    baseline_loss, baseline_time, baseline_stats, baseline_profile = profile_train_epoch(
        baseline_model,
        baseline_dl,
        baseline_criterion,
        baseline_optimizer,
        device,
        semantic_feature_p=0.0,
        output_file=baseline_profile_file
    )
    
    logger.info(f"Baseline profile saved to {baseline_profile_file}")
    logger.info(f"Baseline epoch time: {baseline_time:.4f}s with loss: {baseline_loss:.4f}")
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
    
    semantic_dl = setup_dataloader(
        batch_size=args.batch_size,
        n_samples=args.n_samples,
        num_features=args.num_features,
        device=device,
        config=semantic_config,
        num_batches=args.num_batches
    )
    
    # Create optimizer directly
    semantic_optimizer = torch.optim.AdamW(
        semantic_model.parameters(), 
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Warm-up run
    logger.info("Running semantic warm-up epoch...")
    try:
        _ = train_epoch(
            semantic_model, 
            aggregate_k_gradients=1,
            using_dist=False,
            scaler=None,
            dl=semantic_dl,
            device=device,
            optimizer=semantic_optimizer,
            criterion=semantic_criterion,
            n_out=semantic_model.base_model.n_out,
            progress_bar=False,
            batch_monitor=None,
            semantic_batch_monitoring=False,
            skip_bad_semantic_batches=False,
            train_mixed_precision=False
        )
    except Exception as e:
        logger.error(f"Error in semantic warm-up epoch: {e}")
        if args.continue_on_error:
            logger.info("Continuing despite error...")
        else:
            raise
    
    # Profile semantic
    logger.info("Profiling semantic epoch...")
    semantic_profile_file = f"{log_dir}/semantic_profile.prof"
    try:
        semantic_loss, semantic_time, semantic_stats, semantic_profile = profile_train_epoch(
            semantic_model,
            semantic_dl,
            semantic_criterion,
            semantic_optimizer,
            device,
            semantic_feature_p=args.semantic_feature_p,
            output_file=semantic_profile_file
        )
        
        logger.info(f"Semantic profile saved to {semantic_profile_file}")
        logger.info(f"Semantic epoch time: {semantic_time:.4f}s with loss: {semantic_loss:.4f}")
        logger.info(f"Semantic top functions:\n{format_stats(semantic_stats)}")
        
        # Compare profiles
        results = compare_profiles(
            baseline_profile, 
            semantic_profile, 
            baseline_time, 
            semantic_time,
            num_functions=args.num_functions
        )
        
        # Final summary
        logger.info(f"\n{'=' * 80}")
        logger.info(f"PERFORMANCE SUMMARY")
        logger.info(f"{'=' * 80}")
        logger.info(f"Baseline epoch time:  {baseline_time:.4f}s")
        logger.info(f"Semantic epoch time:  {semantic_time:.4f}s")
        logger.info(f"Slowdown factor:      {semantic_time/baseline_time:.2f}x")
        logger.info(f"Recommendations for optimization:")
        
        # Make specific recommendations based on the profiling results
        if semantic_time/baseline_time > 5:
            logger.info("1. The semantic model is significantly slower - focus on optimizing the top functions")
        
        # Check if CLIP text encoder is taking a lot of time
        clip_overhead = 0
        for func, stats in semantic_profile.stats.items():
            if 'clip' in func[2].lower() or 'transformer' in func[2].lower():
                clip_overhead += stats[3]  # cumtime
        
        if clip_overhead > semantic_time * 0.3:
            logger.info(f"2. CLIP text encoder is taking {clip_overhead:.2f}s ({clip_overhead/semantic_time*100:.1f}% of time)")
            logger.info("   - Consider using float16 precision for CLIP")
            logger.info("   - Freeze CLIP weights and pre-compute embeddings")
            logger.info("   - Cache repeated token sequences")
        
        # Check for tokenization overhead
        token_overhead = 0
        for func, stats in semantic_profile.stats.items():
            if 'tokenize' in func[2].lower() or 'token' in func[2].lower():
                token_overhead += stats[3]  # cumtime
        
        if token_overhead > semantic_time * 0.1:
            logger.info(f"3. Tokenization is taking {token_overhead:.2f}s ({token_overhead/semantic_time*100:.1f}% of time)")
            logger.info("   - Implement batched tokenization")
            logger.info("   - Cache tokenization results for repeated patterns")
        
        # Check for data movement overhead
        data_overhead = 0
        for func, stats in semantic_profile.stats.items():
            if 'to(' in func[2].lower() or 'cuda' in func[2].lower() or 'copy' in func[2].lower():
                data_overhead += stats[3]  # cumtime
        
        if data_overhead > semantic_time * 0.1:
            logger.info(f"4. Data movement is taking {data_overhead:.2f}s ({data_overhead/semantic_time*100:.1f}% of time)")
            logger.info("   - Reduce CPU-GPU transfers")
            logger.info("   - Keep more tensors on GPU consistently")
        
        # Save summary to file
        summary_file = f"{log_dir}/performance_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"PERFORMANCE SUMMARY\n")
            f.write(f"{'=' * 80}\n")
            f.write(f"Baseline epoch time:  {baseline_time:.4f}s\n")
            f.write(f"Semantic epoch time:  {semantic_time:.4f}s\n")
            f.write(f"Slowdown factor:      {semantic_time/baseline_time:.2f}x\n")
            f.write(f"Profile run at:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Parameters:\n")
            f.write(f"  batch_size:        {args.batch_size}\n")
            f.write(f"  n_samples:         {args.n_samples}\n")
            f.write(f"  num_features:      {args.num_features}\n")
            f.write(f"  emsize:            {args.emsize}\n")
            f.write(f"  nlayers:           {args.nlayers}\n")
            f.write(f"  device:            {device}\n")
            f.write(f"  semantic_feature_p: {args.semantic_feature_p}\n\n")
            
            # Add profile file paths
            f.write(f"Profile files:\n")
            f.write(f"  Baseline profile:  {baseline_profile_file}\n")
            f.write(f"  Semantic profile:  {semantic_profile_file}\n\n")
            
            # Add command to view profiles
            f.write(f"View profiles with:\n")
            f.write(f"  python -m cProfile -s cumtime {baseline_profile_file}\n")
            f.write(f"  python -m cProfile -s cumtime {semantic_profile_file}\n")
        
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
    parser = argparse.ArgumentParser(description="Profile TabPFN with and without semantic features")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--n-samples", type=int, default=20, help="Number of samples")
    parser.add_argument("--num-features", type=int, default=20, help="Number of features")
    parser.add_argument("--emsize", type=int, default=128, help="Embedding size")
    parser.add_argument("--nlayers", type=int, default=2, help="Number of layers")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"], 
                       help="Device to use")
    parser.add_argument("--semantic-feature-p", type=float, default=0.3, 
                       help="Probability of using semantic features")
    parser.add_argument("--learning-rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--num-batches", type=int, default=5, help="Number of batches per epoch")
    parser.add_argument("--num-functions", type=int, default=15, 
                       help="Number of top functions to analyze")
    parser.add_argument("--continue-on-error", action="store_true", 
                       help="Continue execution if semantic model fails")
    
    args = parser.parse_args()
    main(args)