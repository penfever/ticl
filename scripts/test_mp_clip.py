#!/usr/bin/env python
"""
Quick test for semantic model mixed precision implementation.
"""
import os
import sys
import time
import torch
import logging
import numpy as np
from pathlib import Path

# Setup path
script_dir = Path(__file__).resolve().parent
ticl_dir = script_dir.parent
sys.path.insert(0, str(ticl_dir))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_mp_clip")

# Import transformers
try:
    from transformers import CLIPTokenizerFast, CLIPTextModel
except ImportError:
    logger.error("Transformers package not found. Install with: pip install transformers")
    sys.exit(1)

# Get device
device = torch.device('mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() 
                     else 'cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Mixed Precision CLIP Processor implementation
class MixedPrecisionCLIPProcessor:
    """Handles mixed precision processing for CLIP text models."""
    
    def __init__(self, clip_model, enable_mixed_precision=True, precision='auto', device=None):
        self.clip_model = clip_model
        self.enable_mixed_precision = enable_mixed_precision
        self.precision = precision
        self.device = device or self._detect_device()
        
        # Set the appropriate dtype
        self.dtype = self._get_optimal_dtype()
        
        if self.enable_mixed_precision:
            self._convert_model_precision()
    
    def _detect_device(self):
        """Detect the current device of the CLIP model"""
        for param in self.clip_model.parameters():
            return param.device
        return torch.device('cpu')
    
    def _get_optimal_dtype(self):
        """Determine the optimal dtype based on device and precision setting"""
        if self.precision == 'float32':
            return torch.float32
            
        if self.precision == 'float16':
            return torch.float16
            
        if self.precision == 'bfloat16':
            if hasattr(torch, 'bfloat16'):
                return torch.bfloat16
            else:
                logger.warning("bfloat16 requested but not available, falling back to float16")
                return torch.float16
        
        # Auto-detect best precision
        device_type = self.device.type
        
        if device_type == 'cuda':
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            else:
                return torch.float16
                
        elif device_type == 'mps':
            return torch.float16
            
        elif device_type == 'cpu':
            if hasattr(torch, 'bfloat16'):
                return torch.bfloat16
            else:
                return torch.float32
        
        return torch.float32
    
    def _convert_model_precision(self):
        """Convert the CLIP model to the appropriate precision"""
        if not self.enable_mixed_precision:
            return
            
        model_dtype = next(self.clip_model.parameters()).dtype
        if model_dtype == self.dtype:
            logger.info(f"CLIP model already in {self.dtype} precision, skipping conversion")
            return
            
        logger.info(f"Converting CLIP model from {model_dtype} to {self.dtype} precision")
        self.clip_model = self.clip_model.to(dtype=self.dtype)
    
    def get_autocast_context(self):
        """Get the appropriate autocast context for the current device and precision"""
        if not self.enable_mixed_precision:
            return torch.no_grad()
            
        device_type = self.device.type
        
        # Skip autocast for MPS as it can cause issues
        if device_type == 'mps':
            return torch.no_grad()
            
        # Create appropriate autocast context
        if device_type in ['cuda', 'cpu']:
            return torch.autocast(device_type=device_type, dtype=self.dtype)
            
        # Fallback
        return torch.no_grad()
    
    def process_tokens(self, token_dict, return_all_outputs=False):
        """Process tokens using the CLIP text model with appropriate precision handling."""
        # Ensure input tensors are on the right device
        input_ids = token_dict['input_ids'].to(self.device)
        attention_mask = token_dict['attention_mask'].to(self.device)
        
        # Create input dictionary
        model_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        
        # Process with appropriate precision
        with torch.no_grad():
            with self.get_autocast_context():
                outputs = self.clip_model(**model_inputs)
        
        # Return either just the pooler output or all outputs
        if return_all_outputs:
            return outputs
        else:
            return outputs.pooler_output

def run_full_precision_test(clip_model, tokenizer, device, texts, n_runs=5):
    """Run test with full precision."""
    logger.info("Testing with FULL PRECISION")
    
    # Prepare tokens
    tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    
    # Warm-up run
    with torch.no_grad():
        _ = clip_model(**tokens)
    
    # Benchmark
    start_time = time.time()
    for _ in range(n_runs):
        with torch.no_grad():
            outputs = clip_model(**tokens)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / n_runs
    logger.info(f"Full precision - Avg time: {avg_time*1000:.2f}ms")
    
    # Check for issues
    has_nan = torch.isnan(outputs.pooler_output).any().item()
    has_inf = torch.isinf(outputs.pooler_output).any().item()
    if has_nan or has_inf:
        logger.warning("Full precision has NaN/Inf values!")
    
    return avg_time, outputs.pooler_output

def run_mixed_precision_test(processor, tokenizer, device, texts, n_runs=5):
    """Run test with mixed precision."""
    logger.info(f"Testing with MIXED PRECISION ({processor.dtype})")
    
    # Prepare tokens
    tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    
    # Warm-up run
    _ = processor.process_tokens(tokens)
    
    # Benchmark
    start_time = time.time()
    for _ in range(n_runs):
        outputs = processor.process_tokens(tokens)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / n_runs
    logger.info(f"Mixed precision - Avg time: {avg_time*1000:.2f}ms")
    
    # Check for issues
    has_nan = torch.isnan(outputs).any().item()
    has_inf = torch.isinf(outputs).any().item()
    if has_nan or has_inf:
        logger.warning("Mixed precision has NaN/Inf values!")
    
    return avg_time, outputs

def main():
    # Load models
    logger.info(f"Loading CLIP model on {device}")
    tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    
    # Create test data
    texts = [
        "This is a sample text for testing the CLIP model efficiency",
        "Tabular data processing requires specialized models",
        "Mixed precision can significantly improve performance",
        "Apple Silicon requires specific optimizations for best performance",
        "Large language models are revolutionizing AI applications"
    ]
    
    # Run full precision test
    fp_time, fp_output = run_full_precision_test(clip_model, tokenizer, device, texts)
    
    # Create and run mixed precision processor
    processor = MixedPrecisionCLIPProcessor(
        clip_model=clip_model, 
        enable_mixed_precision=True,
        device=device
    )
    
    mp_time, mp_output = run_mixed_precision_test(processor, tokenizer, device, texts)
    
    # Compare results
    speedup = fp_time / mp_time if mp_time > 0 else 0
    logger.info(f"Speedup: {speedup:.2f}x")
    
    # Check output similarity
    output_diff = torch.abs(fp_output - mp_output).mean().item()
    logger.info(f"Output difference (mean abs diff): {output_diff:.6f}")
    
    # Output comparison
    if output_diff < 0.01:
        logger.info("✅ Outputs are very similar - mixed precision is working well!")
    elif output_diff < 0.1:
        logger.info("⚠️ Outputs have moderate differences - acceptable for most use cases")
    else:
        logger.info("❌ Outputs have significant differences - mixed precision may cause issues")
    
    # Print summary
    print("\n===== SUMMARY =====")
    print(f"Device: {device.type}")
    print(f"Mixed precision dtype: {processor.dtype}")
    print(f"Full precision time: {fp_time*1000:.2f}ms")
    print(f"Mixed precision time: {mp_time*1000:.2f}ms")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Output difference: {output_diff:.6f}")
    print("=" * 20)

if __name__ == "__main__":
    main()