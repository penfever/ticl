#!/usr/bin/env python
"""
Test script for semantic model mixed precision implementation.

This script benchmarks the performance and memory usage of the mixed precision
CLIP text processor across different devices (CUDA, CPU, MPS) and precision settings.
"""
import os
import sys
import time
import argparse
import torch
import logging
import numpy as np
from collections import defaultdict
from contextlib import nullcontext

# Path handling is done after the imports

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_semantic_precision")

# Fix import path - we need to adjust for the directory structure
import sys
from pathlib import Path

# Add the parent directory to sys.path (if needed)
script_dir = Path(__file__).resolve().parent
ticl_dir = script_dir.parent
sys.path.insert(0, str(ticl_dir))

# Import the module to test
from ticl.models.semantic_model_precision import MixedPrecisionCLIPProcessor, optimize_clip_model

def get_available_device():
    """Determine the best available device for testing"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    
    # Check for MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    
    # Fallback to CPU
    return torch.device('cpu')

def load_clip_model(device):
    """Load the CLIP text model for testing"""
    try:
        from transformers import CLIPTokenizerFast, CLIPTextModel
        
        logger.info(f"Loading CLIP text model on {device}")
        tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
        clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        
        return tokenizer, clip_model
    except Exception as e:
        logger.error(f"Failed to load CLIP model: {e}")
        raise

def generate_test_tokens(tokenizer, device, batch_size=4, seq_length=50):
    """Generate sample tokens for testing"""
    texts = [
        "This is a sample text for testing the CLIP model efficiency",
        "Tabular data processing requires specialized models",
        "Mixed precision can significantly improve performance",
        "Apple Silicon requires specific optimizations for best performance"
    ]
    
    # Repeat texts if needed for larger batch sizes
    while len(texts) < batch_size:
        texts.extend(texts[:batch_size - len(texts)])
    
    # Truncate to requested batch size
    texts = texts[:batch_size]
    
    # Tokenize texts
    tokens = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=seq_length,
        return_tensors="pt"
    )
    
    # Move to device
    tokens = {k: v.to(device) for k, v in tokens.items()}
    
    return tokens

def measure_memory_usage(callable_fn, *args, **kwargs):
    """Measure peak memory usage of a function call"""
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.memory_allocated()
    
    # Run the function
    result = callable_fn(*args, **kwargs)
    
    # Measure memory usage
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated()
        mem_usage = peak_mem - start_mem
        return result, mem_usage
    else:
        # For CPU or MPS, return a dummy memory usage
        return result, 0

def benchmark_processor(processor, token_dict, batch_sizes, seq_length, num_runs=10):
    """Benchmark the processor with different batch sizes"""
    results = defaultdict(list)
    tokenizer, clip_model = load_clip_model(processor.device)
    
    logger.info(f"Testing with precision: {processor.dtype}")
    
    for batch_size in batch_sizes:
        logger.info(f"Testing batch size: {batch_size}")
        
        # Generate test tokens for this batch size
        test_tokens = generate_test_tokens(tokenizer, processor.device, batch_size, seq_length)
        
        # Warm-up run
        processor.process_tokens(test_tokens)
        
        # Benchmark runs
        durations = []
        memory_usages = []
        
        for i in range(num_runs):
            # Clear memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Time the function and measure memory
            start_time = time.time()
            output, mem_usage = measure_memory_usage(processor.process_tokens, test_tokens)
            end_time = time.time()
            
            # Record stats
            duration = end_time - start_time
            durations.append(duration)
            memory_usages.append(mem_usage)
            
            # Check for issues
            if torch.isnan(output).any() or torch.isinf(output).any():
                logger.warning(f"Run {i+1}: NaN or Inf detected in output!")
            
            # Log progress
            if (i + 1) % 5 == 0:
                logger.info(f"Completed {i+1}/{num_runs} runs")
        
        # Store results
        results[batch_size] = {
            'mean_time': np.mean(durations),
            'std_time': np.std(durations),
            'min_time': np.min(durations),
            'max_time': np.max(durations),
            'mean_memory': np.mean(memory_usages),
            'has_nan': torch.isnan(output).any().item(),
            'has_inf': torch.isinf(output).any().item(),
        }
    
    return results

def test_all_precision_configs(device, batch_sizes=[1, 4, 8, 16], seq_length=50, num_runs=5):
    """Test all precision configurations on the given device"""
    tokenizer, clip_model = load_clip_model(device)
    
    # Define precision configurations to test
    precision_configs = [
        {'name': 'float32', 'enable_mixed_precision': False},
        {'name': 'auto', 'enable_mixed_precision': True},
    ]
    
    # Add explicit precision settings if device supports them
    if device.type == 'cuda':
        precision_configs.extend([
            {'name': 'float16', 'precision': 'float16', 'enable_mixed_precision': True},
            {'name': 'bfloat16', 'precision': 'bfloat16', 'enable_mixed_precision': True},
        ])
    elif device.type == 'mps':
        precision_configs.append(
            {'name': 'float16', 'precision': 'float16', 'enable_mixed_precision': True}
        )
    
    all_results = {}
    
    for config in precision_configs:
        config_name = config.pop('name')
        logger.info(f"Testing configuration: {config_name}")
        
        # Create processor with this configuration
        processor = MixedPrecisionCLIPProcessor(
            clip_model=clip_model,
            device=device,
            **config
        )
        
        # Benchmark
        results = benchmark_processor(
            processor=processor,
            token_dict=None,  # Will be generated within benchmark function
            batch_sizes=batch_sizes,
            seq_length=seq_length,
            num_runs=num_runs
        )
        
        all_results[config_name] = results
    
    return all_results

def print_results(all_results):
    """Print the benchmark results in a readable format"""
    print("\n===== BENCHMARK RESULTS =====")
    print("-" * 80)
    
    for config_name, results in all_results.items():
        print(f"\nConfiguration: {config_name}")
        print("-" * 40)
        print("| Batch Size | Mean Time (ms) | Memory (MB) | NaN/Inf |")
        print("|------------|---------------|-------------|---------|")
        
        for batch_size, stats in results.items():
            mean_time_ms = stats['mean_time'] * 1000
            memory_mb = stats['mean_memory'] / (1024 * 1024) if stats['mean_memory'] > 0 else "N/A"
            has_issues = "Yes" if stats['has_nan'] or stats['has_inf'] else "No"
            
            if isinstance(memory_mb, str):
                memory_str = memory_mb
            else:
                memory_str = f"{memory_mb:.2f}"
                
            print(f"| {batch_size:^10} | {mean_time_ms:^13.2f} | {memory_str:^11} | {has_issues:^7} |")
    
    print("-" * 80)

def compare_results(full_precision_results, mixed_precision_results, device_type):
    """Compare full precision and mixed precision results to show speedup"""
    print("\n===== SPEEDUP ANALYSIS =====")
    print("-" * 80)
    print("| Batch Size | Full Precision (ms) | Mixed Precision (ms) | Speedup (x) |")
    print("|------------|---------------------|----------------------|-------------|")
    
    for batch_size in full_precision_results.keys():
        fp_time = full_precision_results[batch_size]['mean_time'] * 1000
        mp_time = mixed_precision_results[batch_size]['mean_time'] * 1000
        speedup = fp_time / mp_time if mp_time > 0 else 0
        
        print(f"| {batch_size:^10} | {fp_time:^19.2f} | {mp_time:^20.2f} | {speedup:^11.2f} |")
    
    print("-" * 80)
    print(f"Device: {device_type}")

def main():
    """Main testing function"""
    parser = argparse.ArgumentParser(description="Test semantic model mixed precision implementation")
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 4, 8], 
                        help='Batch sizes to test')
    parser.add_argument('--seq-length', type=int, default=50, 
                        help='Sequence length for test tokens')
    parser.add_argument('--num-runs', type=int, default=5, 
                        help='Number of runs per configuration')
    args = parser.parse_args()
    
    # Get device
    device = get_available_device()
    logger.info(f"Testing on device: {device}")
    
    # Run tests
    try:
        all_results = test_all_precision_configs(
            device=device,
            batch_sizes=args.batch_sizes,
            seq_length=args.seq_length,
            num_runs=args.num_runs
        )
        
        # Print results
        print_results(all_results)
        
        # Compare full precision vs auto mixed precision
        if 'float32' in all_results and 'auto' in all_results:
            compare_results(
                all_results['float32'],
                all_results['auto'],
                device.type
            )
        
        logger.info("Testing completed successfully!")
        return 0
    
    except Exception as e:
        logger.error(f"Testing failed: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())