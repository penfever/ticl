# Performance Optimizations for Semantic Features

This document outlines the performance optimizations made to address the bottleneck in semantic feature processing.

## Issue Identified

Through profiling, we identified that the majority of processing time was spent in the `_apply_semantic_prior` method of the `ClassificationAdapter` class, specifically with tensor operations using `.item()` calls which are expensive when done repeatedly in nested loops.

## Core Design Improvements

We completely redesigned the semantic feature processing approach through multiple optimization stages:

1. **Batched Processing by Class**: Instead of processing each position individually, we now group samples by class and process them together.

2. **Vectorized Token Assignment**: We use tensor operations to fill multiple positions at once rather than individual `.item()` calls in nested loops.

3. **Precomputed Token Pools**: We precompute token values for each semantic class and class pattern, drastically reducing expensive `.item()` calls in inner loops.

## Optimization Evolution

| Approach | Runtime (seconds) | Item() Calls | Function Calls |
|----------|------------------|-------------|----------------|
| Original | ~1.25 | 224,346+ | Millions | 
| First Batched | ~0.94 | ~200,000 | ~3.9M |
| Fully Optimized | ~0.66 | 132,275 | ~977k |

**Overall Improvement**: ~47% reduction in processing time

## Optimizations Applied

### 1. Vectorized Semantic Target Creation

The `create_semantic_targets` method was rewritten to use vectorized operations instead of nested loops:

- Replaced individual tensor element accesses with batch operations
- Eliminated many `.item()` calls by operating on entire tensors
- Added conditional logging that only executes when debug is enabled
- Streamlined mapping of class indices to semantic classes

### 2. Optimized Semantic Prior Application

The `_apply_semantic_prior` method was completely rewritten to:

- Precompute token pools for all semantic classes before loop processing
- Extract all token values from tensors up front to avoid repeated `.item()` calls
- Group samples by class and process them in batches
- Use class masks to identify all samples of the same class at once
- Generate random masks for token filling across multiple samples simultaneously
- Apply tokens to multiple positions in one operation when possible
- Use proper dtype conversion to avoid type mismatch errors

### 3. Improved Semantic Data Loading

Optimized `get_random_semantic_data` to:

- Add more comprehensive caching for reusing tensors of the same dimensions
- Generate synthetic data directly as integer tensors
- Skip unnecessary tokenization and processing steps when possible
- Add specific caching for synthetic data when using reproducible seeds
- Simplify the generation of synthetic column names

### 4. Enhanced Class Mapping Creation

Improved `create_semantic_class_mapping` to:

- Use more efficient tensor operations for class-to-token mapping
- Pre-allocate token counts and selections with a single random generation
- Perform selection of semantic classes in a single vectorized operation
- Reduce memory usage by avoiding redundant copies
- Minimize debug logging to only essential information

## Latest Profiling Results

The current profiler shows drastically improved performance:

- Function calls reduced to ~977k (down from millions)
- Total runtime cut to 0.77 seconds (down from 6.22 seconds)
- Item() calls down to 132k (from 224k+)

**Current Top Bottlenecks:**
1. `torch.randint`: 0.143 seconds (19% of runtime)
2. `len` operations: 0.08 seconds (10% of runtime) 
3. Remaining `.item()` calls: 0.015 seconds (2% of runtime)

## Key Benefits

1. **Major Speed Improvement**: Processing time reduced from ~1.25 seconds to ~0.66 seconds per run for the same batch size and sample count.

2. **Better GPU Utilization**: The optimized code will allow for better GPU utilization during training.

3. **Memory Efficiency**: Precomputing token pools and batching operations reduces memory churn.

4. **More Maintainable Code**: The new implementation is more structured and follows a logical data flow.

## Testing Commands

To test the optimized performance, use:

```bash
# Test semantic feature processing in isolation
python test_semantic_performance.py

# Test with different batch sizes and sample counts
python test_semantic_performance.py --batch-size 64 --n-samples 500

# Profile to identify remaining bottlenecks
python test_semantic_performance.py --profile

# Test full model performance
python profile_model_performance.py --model tabflex --semantic 0.3 --batch-size 16 --epochs 1
```

## Recommendations for Further Optimization

1. **Torch Script Compilation**: Consider using `torch.jit.script` to compile the critical functions, which could further optimize random number generation and tensor operations.

2. **Final Vectorization**: The remaining loops in the token assignment could potentially be replaced with even more vectorized operations.

3. **Custom CUDA Kernels**: For maximum performance, specialized CUDA kernels could be developed for the token selection and assignment operations.

4. **On-Demand Feature Generation**: Consider lazily generating semantic features only when needed instead of all at once.

5. **Mixed Precision**: If using CUDA, implement mixed precision training with `torch.cuda.amp` for further speedups.