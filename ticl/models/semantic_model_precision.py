"""
Precision utilities for semantic model processing.
This module focuses on enabling mixed precision for CLIP text models.
"""
import torch
import torch.nn as nn
import logging
import contextlib
from typing import Optional, Dict, Any, Tuple, Union

# Configure logging
logger = logging.getLogger(__name__)

class MixedPrecisionCLIPProcessor:
    """
    Handles mixed precision processing for CLIP text models.
    
    This class ensures that CLIP text models can run efficiently with mixed precision,
    handling device-specific optimizations and proper precision conversion.
    """
    
    def __init__(
        self, 
        clip_model, 
        enable_mixed_precision: bool = True,
        precision: str = 'auto',
        device: Optional[torch.device] = None
    ):
        """
        Initialize the mixed precision processor.
        
        Parameters:
        -----------
        clip_model : CLIPTextModel
            The CLIP text model to process with mixed precision
        enable_mixed_precision : bool
            Whether to enable mixed precision (default: True)
        precision : str
            Precision to use: 'auto', 'float16', 'bfloat16', or 'float32'
        device : torch.device, optional
            Device to use (default: auto-detect)
        """
        self.clip_model = clip_model
        self.enable_mixed_precision = enable_mixed_precision
        self.precision = precision
        self.device = device or self._detect_device()
        
        # Set the appropriate dtype based on precision setting and device capabilities
        self.dtype = self._get_optimal_dtype()
        
        # Convert model to appropriate precision if mixed precision is enabled
        if self.enable_mixed_precision:
            self._convert_model_precision()
    
    def _detect_device(self) -> torch.device:
        """Detect the current device of the CLIP model"""
        # Check if any parameter exists to get device
        for param in self.clip_model.parameters():
            return param.device
        
        # Fallback to CPU if no parameters found
        return torch.device('cpu')
    
    def _get_optimal_dtype(self) -> torch.dtype:
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
        
        # CUDA devices generally support float16 well
        if device_type == 'cuda':
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            else:
                return torch.float16
                
        # Apple Silicon (MPS) works well with float16
        elif device_type == 'mps':
            return torch.float16
            
        # CPU generally works better with bfloat16 when available
        elif device_type == 'cpu':
            if hasattr(torch, 'bfloat16'):
                return torch.bfloat16
            else:
                # Generally avoid float16 on CPU as it's often emulated
                return torch.float32
        
        # Default to float32 for any other device
        return torch.float32
    
    def _convert_model_precision(self):
        """Convert the CLIP model to the appropriate precision"""
        if not self.enable_mixed_precision:
            return
            
        # Skip if already in the right precision
        model_dtype = next(self.clip_model.parameters()).dtype
        if model_dtype == self.dtype:
            logger.debug(f"CLIP model already in {self.dtype} precision, skipping conversion")
            return
            
        logger.info(f"Converting CLIP model from {model_dtype} to {self.dtype} precision")
        
        # Use to() method to convert the model
        self.clip_model = self.clip_model.to(dtype=self.dtype)
    
    def get_autocast_context(self):
        """Get the appropriate autocast context for the current device and precision"""
        if not self.enable_mixed_precision:
            return contextlib.nullcontext()
            
        device_type = self.device.type
        
        # Skip autocast for MPS as it can cause issues
        if device_type == 'mps':
            return contextlib.nullcontext()
            
        # Create appropriate autocast context
        if device_type in ['cuda', 'cpu']:
            return torch.autocast(device_type=device_type, dtype=self.dtype)
            
        # Fallback to nullcontext for unsupported devices
        return contextlib.nullcontext()
    
    def process_tokens(
        self, 
        token_dict: Dict[str, torch.Tensor], 
        return_all_outputs: bool = False
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Process tokens using the CLIP text model with appropriate precision handling.
        
        Parameters:
        -----------
        token_dict : Dict[str, torch.Tensor]
            Dictionary containing 'input_ids' and 'attention_mask'
        return_all_outputs : bool
            Whether to return all model outputs or just the pooler_output
            
        Returns:
        --------
        torch.Tensor or Dict[str, Any]
            The model outputs, either just pooler_output or the full output dictionary
        """
        # Ensure input tensors are on the right device
        input_ids = token_dict['input_ids'].to(self.device)
        attention_mask = token_dict['attention_mask'].to(self.device)
        
        # Create input dictionary
        model_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        
        # Determine whether to track gradients
        requires_grad = any(p.requires_grad for p in self.clip_model.parameters())
        
        # Process with appropriate precision
        with torch.set_grad_enabled(requires_grad):
            with self.get_autocast_context():
                outputs = self.clip_model(**model_inputs)
        
        # Check for NaN/Inf values in embeddings
        if torch.isnan(outputs.pooler_output).any() or torch.isinf(outputs.pooler_output).any():
            logger.warning("NaN or Inf detected in CLIP embeddings")
            # Sanitize output for stability
            pooler_output = torch.nan_to_num(
                outputs.pooler_output, 
                nan=0.0, 
                posinf=1.0, 
                neginf=-1.0
            )
        else:
            pooler_output = outputs.pooler_output
        
        # Return either just the pooler output or all outputs
        if return_all_outputs:
            # Replace pooler_output with sanitized version
            outputs.pooler_output = pooler_output
            return outputs
        else:
            return pooler_output

def optimize_clip_model(clip_model, enable_mixed_precision=True, precision='auto', device=None):
    """
    Factory function to create a mixed precision CLIP processor.
    
    Parameters:
    -----------
    clip_model : CLIPTextModel
        The CLIP text model to optimize
    enable_mixed_precision : bool
        Whether to enable mixed precision (default: True)
    precision : str
        Precision to use: 'auto', 'float16', 'bfloat16', or 'float32'
    device : torch.device, optional
        Device to use (default: auto-detect)
        
    Returns:
    --------
    MixedPrecisionCLIPProcessor
        Processor for working with the CLIP model in optimized precision
    """
    return MixedPrecisionCLIPProcessor(
        clip_model=clip_model,
        enable_mixed_precision=enable_mixed_precision,
        precision=precision,
        device=device
    )