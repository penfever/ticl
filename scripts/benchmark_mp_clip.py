#!/usr/bin/env python
"""
Benchmark mixed precision CLIP processing across different batch sizes.
"""
import os
import sys
import time
import torch
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from contextlib import nullcontext

# Setup path
script_dir = Path(__file__).resolve().parent
ticl_dir = script_dir.parent
sys.path.insert(0, str(ticl_dir))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("benchmark_mp_clip")

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
            return nullcontext()
            
        device_type = self.device.type
        
        # Skip autocast for MPS as it can cause issues
        if device_type == 'mps':
            return nullcontext()
            
        # Create appropriate autocast context
        if device_type in ['cuda', 'cpu']:
            return torch.autocast(device_type=device_type, dtype=self.dtype)
            
        # Fallback
        return nullcontext()
    
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

def generate_text_batch(batch_size):
    """Generate a batch of sample texts for testing."""
    base_texts = [
        "This is a sample text for testing CLIP model performance",
        "Tabular data processing requires specialized models",
        "Mixed precision can significantly improve performance", 
        "Apple Silicon requires specific optimizations",
        "Large language models are transforming AI applications",
        "Neural networks excel at pattern recognition tasks",
        "Transformer models have revolutionized NLP",
        "Computer vision systems benefit from deep learning",
        "Machine learning algorithms need sufficient training data",
        "Data preprocessing is crucial for model performance"
    ]
    
    # Repeat texts as needed to reach the desired batch size
    texts = []
    while len(texts) < batch_size:
        texts.extend(base_texts[:min(batch_size - len(texts), len(base_texts))])
    
    return texts[:batch_size]

def benchmark_batch_sizes(batch_sizes, n_runs=5):
    """Benchmark processing with different batch sizes."""
    # Load models
    logger.info(f"Loading CLIP model on {device}")
    tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    
    # Create mixed precision processor
    processor = MixedPrecisionCLIPProcessor(
        clip_model=clip_model, 
        enable_mixed_precision=True,
        device=device
    )
    
    results = []
    
    for batch_size in batch_sizes:
        logger.info(f"Testing batch size: {batch_size}")
        
        # Generate text batch
        texts = generate_text_batch(batch_size)
        
        # Tokenize for full precision
        tokens_fp = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
        
        # Tokenize for mixed precision 
        tokens_mp = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
        
        # Warm-up runs
        with torch.no_grad():
            _ = clip_model(**tokens_fp)
        _ = processor.process_tokens(tokens_mp)
        
        # Full precision benchmark
        fp_times = []
        for _ in range(n_runs):
            start_time = time.time()
            with torch.no_grad():
                outputs_fp = clip_model(**tokens_fp)
            fp_times.append(time.time() - start_time)
        
        fp_avg_time = np.mean(fp_times) * 1000  # Convert to ms
        
        # Mixed precision benchmark
        mp_times = []
        for _ in range(n_runs):
            start_time = time.time()
            outputs_mp = processor.process_tokens(tokens_mp)
            mp_times.append(time.time() - start_time)
        
        mp_avg_time = np.mean(mp_times) * 1000  # Convert to ms
        
        # Calculate speedup
        speedup = fp_avg_time / mp_avg_time if mp_avg_time > 0 else 0
        
        # Calculate output difference
        output_diff = torch.abs(outputs_fp.pooler_output - outputs_mp).mean().item()
        
        # Store results
        results.append({
            'batch_size': batch_size,
            'fp_time_ms': fp_avg_time,
            'mp_time_ms': mp_avg_time,
            'speedup': speedup,
            'output_diff': output_diff
        })
        
        logger.info(f"Batch {batch_size}: FP={fp_avg_time:.2f}ms, MP={mp_avg_time:.2f}ms, Speedup={speedup:.2f}x")
    
    return results

def plot_results(results):
    """Plot benchmark results."""
    df = pd.DataFrame(results)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot times
    ax1.plot(df['batch_size'], df['fp_time_ms'], 'o-', label='Full Precision')
    ax1.plot(df['batch_size'], df['mp_time_ms'], 's-', label='Mixed Precision')
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Processing Time by Batch Size')
    ax1.legend()
    ax1.grid(True)
    
    # Plot speedup
    ax2.plot(df['batch_size'], df['speedup'], 'o-')
    ax2.axhline(y=1.0, color='r', linestyle='--')
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Speedup (x)')
    ax2.set_title('Mixed Precision Speedup by Batch Size')
    ax2.grid(True)
    
    # Add device and dtype info
    plt.suptitle(f'CLIP Text Model Mixed Precision Benchmark ({device.type}, {results[0].get("dtype", "auto")})')
    plt.tight_layout()
    
    # Save figure
    plt_path = os.path.join(ticl_dir, 'docs', 'benchmark_mp_clip.png')
    plt.savefig(plt_path)
    logger.info(f"Plot saved to {plt_path}")
    
    # Print results as markdown table
    print("\n## Mixed Precision Benchmark Results\n")
    print("| Batch Size | Full Precision (ms) | Mixed Precision (ms) | Speedup | Output Diff |")
    print("|------------|---------------------|----------------------|---------|-------------|")
    
    for row in results:
        print(f"| {row['batch_size']:^10} | {row['fp_time_ms']:^19.2f} | {row['mp_time_ms']:^20.2f} | "
              f"{row['speedup']:^7.2f}x | {row['output_diff']:^11.6f} |")

def main():
    """Main benchmark function."""
    # Set batch sizes to test
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    
    # Run benchmark
    logger.info(f"Starting benchmark on {device} with batch sizes: {batch_sizes}")
    results = benchmark_batch_sizes(batch_sizes, n_runs=5)
    
    # Add dtype info
    for result in results:
        result['dtype'] = 'float16' if device.type == 'mps' else 'auto'
    
    # Plot and print results
    logger.info("Benchmark complete, generating plots and reports")
    plot_results(results)
    
    # Print summary
    best_result = max(results, key=lambda x: x['speedup'])
    logger.info(f"Best speedup: {best_result['speedup']:.2f}x at batch size {best_result['batch_size']}")
    
    worst_result = min(results, key=lambda x: x['speedup'])
    logger.info(f"Worst speedup: {worst_result['speedup']:.2f}x at batch size {worst_result['batch_size']}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())