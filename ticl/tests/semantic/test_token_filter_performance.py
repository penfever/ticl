"""
Benchmark script for token filtering performance.
"""

import time
import torch
from ticl.datasets.semantic_prior_data_loader import get_random_semantic_data

def benchmark_token_filtering(num_classes=10, tensor_size=500, runs=3):
    """Benchmark token filtering performance."""
    print("=" * 80)
    print("BENCHMARKING TOKEN FILTERING PERFORMANCE")
    print("=" * 80)
    print(f"Parameters: num_classes={num_classes}, tensor_size={tensor_size}, runs={runs}")
    
    # Warm-up run
    print("\nWarming up...")
    _ = get_random_semantic_data(
        num_classes=num_classes,
        tensor_size=tensor_size,
        filter_tokens=False
    )
    _ = get_random_semantic_data(
        num_classes=num_classes,
        tensor_size=tensor_size,
        filter_tokens=True
    )
    
    # Benchmark runs
    unfiltered_times = []
    filtered_times = []
    
    print("\nRunning benchmark...")
    for i in range(runs):
        print(f"Run {i+1}/{runs}")
        
        # Time unfiltered generation
        start_time = time.time()
        _ = get_random_semantic_data(
            num_classes=num_classes,
            tensor_size=tensor_size,
            filter_tokens=False
        )
        unfiltered_time = time.time() - start_time
        unfiltered_times.append(unfiltered_time)
        print(f"  Unfiltered: {unfiltered_time:.4f}s")
        
        # Time filtered generation
        start_time = time.time()
        _ = get_random_semantic_data(
            num_classes=num_classes,
            tensor_size=tensor_size,
            filter_tokens=True
        )
        filtered_time = time.time() - start_time
        filtered_times.append(filtered_time)
        print(f"  Filtered:   {filtered_time:.4f}s")
    
    # Calculate statistics
    avg_unfiltered = sum(unfiltered_times) / len(unfiltered_times)
    avg_filtered = sum(filtered_times) / len(filtered_times)
    overhead_ratio = avg_filtered / avg_unfiltered
    overhead_percent = (avg_filtered / avg_unfiltered - 1) * 100
    
    print("\nResults:")
    print(f"  Average unfiltered time: {avg_unfiltered:.4f}s")
    print(f"  Average filtered time:   {avg_filtered:.4f}s")
    print(f"  Overhead ratio:          {overhead_ratio:.2f}x")
    print(f"  Overhead percentage:     {overhead_percent:.1f}%")
    
    print("\nAnalysis:")
    if overhead_percent < 10:
        print("✅ EXCELLENT: Token filtering adds minimal overhead (<10%)")
    elif overhead_percent < 30:
        print("✅ GOOD: Token filtering adds acceptable overhead (<30%)")
    elif overhead_percent < 50:
        print("⚠️ MODERATE: Token filtering adds noticeable overhead (<50%)")
    else:
        print("❌ HIGH: Token filtering adds significant overhead (>50%)")
    
    print("=" * 80)
    
if __name__ == "__main__":
    # Use larger values for a more accurate benchmark
    benchmark_token_filtering(num_classes=30, tensor_size=2000, runs=5)