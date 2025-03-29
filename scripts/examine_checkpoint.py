#!/usr/bin/env python3
"""
Examine the structure of a model checkpoint to locate CLIP weights.
"""

import os
import sys
import torch
import logging
import argparse
from pprint import pprint

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("examine_checkpoint")

def examine_checkpoint(checkpoint_path, detailed=False):
    """
    Examine the structure of a checkpoint file.
    
    Parameters:
    -----------
    checkpoint_path : str
        Path to the checkpoint file
    detailed : bool
        Whether to print detailed information about tensors
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Basic info
    logger.info(f"Checkpoint type: {type(checkpoint)}")
    
    # If checkpoint is a dict, examine its keys
    if isinstance(checkpoint, dict):
        logger.info(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        # Check for state_dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            logger.info(f"State dict has {len(state_dict)} keys")
            
            # Look for CLIP related keys
            clip_keys = [k for k in state_dict.keys() if 'clip' in k.lower() or 'text_model' in k.lower() or 'semantic' in k.lower()]
            logger.info(f"Found {len(clip_keys)} potential CLIP related keys: {clip_keys[:10]}...")
            
            # Group keys by prefix
            prefixes = {}
            for k in state_dict.keys():
                parts = k.split('.')
                prefix = parts[0]
                if len(parts) > 1:
                    prefix = '.'.join(parts[:2])
                
                if prefix not in prefixes:
                    prefixes[prefix] = []
                prefixes[prefix].append(k)
            
            logger.info(f"Key prefixes: {list(prefixes.keys())}")
            
            # If detailed, print some sample tensors
            if detailed:
                logger.info("Sample tensors:")
                for prefix, keys in prefixes.items():
                    if len(keys) > 0:
                        key = keys[0]
                        tensor = state_dict[key]
                        logger.info(f"{key}: {tensor.shape}, {tensor.dtype}")
                        
                        # Print more keys for this prefix
                        if len(keys) > 1:
                            logger.info(f"More keys with prefix {prefix}: {keys[1:5]}...")
        
        # Check for other potentially useful keys
        other_keys = [k for k in checkpoint.keys() if k != 'state_dict']
        for k in other_keys:
            value = checkpoint[k]
            logger.info(f"Key: {k}, Type: {type(value)}")
            
            if isinstance(value, dict):
                logger.info(f"  Subkeys: {list(value.keys())}")
            elif isinstance(value, (list, tuple)):
                logger.info(f"  Length: {len(value)}")
                if len(value) > 0:
                    logger.info(f"  First element type: {type(value[0])}")
    elif isinstance(checkpoint, (list, tuple)):
        # If checkpoint is a list or tuple
        logger.info(f"Checkpoint is a {type(checkpoint).__name__} with {len(checkpoint)} elements")
        
        # Examine each element
        for i, element in enumerate(checkpoint):
            logger.info(f"Element {i} type: {type(element)}")
            
            if isinstance(element, dict):
                logger.info(f"  Dict with {len(element)} keys: {list(element.keys())}")
                
                # Look for state dict or model components
                if 'state_dict' in element:
                    state_dict = element['state_dict']
                    logger.info(f"  State dict has {len(state_dict)} keys")
                    
                    # Look for CLIP related keys
                    clip_keys = [k for k in state_dict.keys() if 'clip' in k.lower() or 'text_model' in k.lower() or 'semantic' in k.lower()]
                    logger.info(f"  Found {len(clip_keys)} potential CLIP related keys: {clip_keys[:10]}...")
                
                # Print a few sample keys and their types if detailed
                if detailed and len(element) > 0:
                    for j, (k, v) in enumerate(element.items()):
                        if j >= 5:  # Limit to first 5 items
                            break
                        logger.info(f"    {k}: {type(v)}")
                        if isinstance(v, torch.Tensor):
                            logger.info(f"      Shape: {v.shape}, Dtype: {v.dtype}")
            
            elif isinstance(element, torch.Tensor):
                logger.info(f"  Tensor shape: {element.shape}, dtype: {element.dtype}")
            
            elif hasattr(element, 'state_dict') and callable(getattr(element, 'state_dict')):
                logger.info(f"  Has state_dict method, might be a model")
                try:
                    state_dict = element.state_dict()
                    logger.info(f"  State dict has {len(state_dict)} keys")
                except Exception as e:
                    logger.error(f"  Error getting state dict: {e}")
            
            elif element is None:
                logger.info("  Element is None")
            
            else:
                logger.info(f"  Other type: {type(element)}")
                
    else:
        # If checkpoint is some other type
        logger.info(f"Checkpoint is an unsupported type: {type(checkpoint)}")

def main():
    parser = argparse.ArgumentParser(description="Examine a PyTorch checkpoint file")
    parser.add_argument("checkpoint_path", type=str, help="Path to checkpoint file")
    parser.add_argument("--detailed", action="store_true", help="Print detailed information")
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.checkpoint_path):
        logger.error(f"File not found: {args.checkpoint_path}")
        return 1
    
    # Examine checkpoint
    examine_checkpoint(args.checkpoint_path, args.detailed)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())