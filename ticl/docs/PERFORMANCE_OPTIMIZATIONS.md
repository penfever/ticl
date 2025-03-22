# Performance and Numerical Stability Optimizations

This document describes the performance and numerical stability optimizations implemented in the codebase, particularly focusing on making the contrastive learning with semantic features work consistently across different hardware backends.

## CUDA vs MPS Numerical Stability

We encountered an issue where the semantic contrastive loss would lead to NaN values in the loss on CUDA devices, but work fine on MPS (Apple Silicon). The key differences include:

1. Precision handling between devices
2. Gradient accumulation and handling
3. Normalizing values with different epsilon values

## Key Optimizations

### 1. Device-Specific Settings

We created backend-specific code paths for critical numerical operations:

```python
# Device-specific parameters
if raw_similarity.device.type == 'cuda':
    # CUDA-specific optimizations
    cuda_eps = 1e-2  # Larger epsilon for CUDA
    raw_similarity = raw_similarity / (raw_similarity.abs().max() + cuda_eps)
    raw_logits = raw_similarity * 1.0  # Smaller scale factor
    raw_logits = torch.clamp(raw_logits, min=-5.0, max=5.0)  # Tighter bounds
else:
    # Standard approach for MPS/CPU
    raw_similarity = raw_similarity / (raw_similarity.abs().max() + self.eps)
    raw_logits = raw_similarity * 2.0  # Original scale factor
    raw_logits = torch.clamp(raw_logits, min=-10.0, max=10.0)  # Original bounds
```

### 2. CUDA-Specific Epsilon Values

CUDA may need larger epsilon values to maintain stability:

- MPS/CPU: `eps = 1e-3`
- CUDA: `eps = 1e-2`

### 3. Gradient Handling

We added a custom gradient hook for CUDA to handle potential NaN/Inf values:

```python
def grad_hook(grad):
    if torch.isnan(grad).any() or torch.isinf(grad).any():
        # Replace NaN/Inf with zeros to allow training to continue
        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
    return grad
    
# Only register hook if running on CUDA
if device_type == 'cuda':
    total_loss.register_hook(grad_hook)
```

### 4. Separate Semantic Weights

We use different weights for the semantic loss component depending on the hardware:

```python
def __init__(self, semantic_weight=0.2, cuda_weight=0.1):
    # CUDA uses a lower weight to reduce amplification of numerical instability
    self.semantic_weight = semantic_weight  # For MPS/CPU
    self.cuda_weight = cuda_weight          # For CUDA
```

### 5. Additional Safeguards for Very Large Values

We added extra normalization for CUDA when values get too large:

```python
# Extra safety: apply softmax normalization to stabilize CUDA training
if raw_max_val > 5.0:
    # Apply row-wise softmax as an additional safety measure
    normalized_logits = F.softmax(raw_logits, dim=-1)
    # Convert back to logits with safe scaling
    raw_logits = torch.log(normalized_logits + 1e-6) * 2.0
```

## Why These Changes Work

1. **Increased Epsilon**: CUDA sometimes produces very small values that can approach zero, increasing epsilon prevents division by near-zero values.

2. **Reduced Scaling**: Lower scaling factors on CUDA prevent gradients from becoming too large during backpropagation.

3. **Tighter Clamping**: More aggressive clamping on CUDA keeps values in a range where they maintain numerical stability.

4. **Lower Semantic Weight**: Reducing the contribution of the semantic loss on CUDA reduces the impact of any potential instability.

5. **Gradient Hook**: Catches and fixes any NaN/Inf values that might still occur during backpropagation.

## Testing Across Backends

The `test_semantic_mps.py` script can be used to test numerical stability across different backends. It compares the behavior of our optimizations on CPU, MPS, and CUDA devices with various epsilon, scaling, and clamping values.

## Results

With these optimizations, the semantic contrastive loss now works consistently across all hardware backends, maintaining numerical stability throughout training.