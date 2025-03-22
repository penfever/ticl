#!/usr/bin/env python3
"""
Test script to verify semantic feature processing in different precision modes,
focusing on the data loading and model forward pass pipeline.
"""

import sys
import torch
import logging
import argparse
import numpy as np
import os
from pathlib import Path

from ticl.models.semantic_aware_model import SemanticConsistencyLoss, SemanticAwareClassifier, create_semantic_aware_model
from ticl.model_builder import get_model, get_criterion
from ticl.model_configs import get_model_default_config
from ticl.utils import get_autocast_context
from ticl.dataloader import get_dataloader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelOutputHook:
    """Hook to capture outputs from forward pass."""
    
    def __init__(self):
        self.outputs = None
    
    def __call__(self, module, input, output):
        if isinstance(output, dict):
            self.outputs = output
        else:
            logger.warning(f"Hook received non-dict output of type: {type(output)}")
            self.outputs = {"raw_output": output}

class DataBatchHook:
    """Hook to capture data batches from the data loader."""
    
    def __init__(self):
        self.batches = []
        self.max_batches = 3  # Store up to 3 batches to avoid memory issues
    
    def __call__(self, batch_data):
        """Process and store batch data."""
        if len(self.batches) < self.max_batches:
            # Clone tensors to detach from computation graph and avoid memory issues
            if isinstance(batch_data, tuple) or isinstance(batch_data, list):
                processed_batch = []
                for item in batch_data:
                    if isinstance(item, torch.Tensor):
                        processed_batch.append(item.detach().clone())
                    elif isinstance(item, dict):
                        processed_dict = {}
                        for k, v in item.items():
                            if isinstance(v, torch.Tensor):
                                processed_dict[k] = v.detach().clone()
                            else:
                                processed_dict[k] = v
                        processed_batch.append(processed_dict)
                    else:
                        processed_batch.append(item)
                self.batches.append(tuple(processed_batch))
            else:
                self.batches.append(batch_data)

def test_model_with_real_dataloader(mixed_precision=False):
    """
    Test semantic feature processing with actual dataloader in different precision modes.
    
    This test simulates the actual training process by:
    1. Creating a real dataloader with the same configuration as training
    2. Using the dataloader to get real batches
    3. Processing those batches through the model with mixed precision on/off
    4. Checking if semantic features are properly generated in both modes
    """
    logger.info(f"\n{'='*40}\nTesting dataloader with mixed_precision={mixed_precision}\n{'='*40}")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device configuration (use CPU for consistent testing)
    device = torch.device("cpu")
    
    try:
        # Create a minimal but realistic config for TabPFN with semantic features
        config = {
            'model_type': 'tabpfn',
            'device': device,
            'prior': {
                'num_features': 10,
                'n_samples': 32,
                'prior_type': 'prior_bag',  # Required for dataloader
                'gp': {                    # Required for prior_bag type
                    'outputscale': 1.0,
                    'lengthscale': 1.0,
                    'noise': 0.0001
                },
                'mlp': {                   # Required for prior_bag type
                    'add_uninformative_features': False,
                    'sampling': 'normal',
                    'num_layers': 2,
                    'prior_mlp_hidden_dim': 16,
                    'prior_mlp_dropout_prob': 0.1,
                    'init_std': 0.1,
                    'noise_std': 0.01,
                    'num_causes': 2,
                    'is_causal': False,
                    'pre_sample_weights': True,
                    'y_is_effect': False,
                    'prior_mlp_scale_weights_sqrt': True,
                    'random_feature_rotation': True,
                    'pre_sample_causes': True,
                    'block_wise_dropout': False,
                    'sort_features': False,
                    'in_clique': False,
                    'prior_mlp_activations': torch.nn.ReLU
                },
                'classification': {
                    'max_num_classes': 2,
                    'num_classes': 2,          # Required for the classification adapter
                    'semantic_feature_p': 0.3,  # Enable semantic features
                    'categorical_feature_p': 0.2, # Required from the error message
                    'pad_zeros': True,
                    'balanced': False,         # Add additional parameters that may be needed
                    'multiclass_type': 'rank', # Required parameter from the error message
                    'multiclass_max_steps': 10, # Used with multiclass_type 
                    'output_multiclass_ordered_p': 0.0, # Parameter from the new error message
                    'nan_prob_no_reason': 0.0, # Additional parameters that might be needed
                    'nan_prob_a_reason': 0.0,
                    'num_features_sampler': 'uniform',
                    'feature_curriculum': False,
                    'set_value_to_nan': 0.9,   # Additional parameter that might be needed
                    'track_causal_features': False,
                }
            },
            'transformer': {
                'emsize': 128,
                'nlayers': 2,
                'nhead': 4,
                'y_encoder': 'linear',
                'classification_task': True,
            },
            'optimizer': {
                'train_mixed_precision': mixed_precision,
            },
            'dataloader': {
                'num_steps': 2,     # Small number of steps for testing
                'batch_size': 4,    # Small batch size
                'min_eval_pos': 2,  # Required parameter for PriorDataLoader
            },
            'orchestration': {
                'progress_bar': False,
            },
            'semantic_prediction': True,
        }
        
        # Create a hook to capture model outputs
        hook = ModelOutputHook()
        
        # Create data batch hook
        batch_hook = DataBatchHook()
        
        logger.info("Creating dataloader...")
        # Create a real dataloader with the configuration
        dataloader = get_dataloader(
            prior_config=config['prior'],
            dataloader_config=config['dataloader'],
            device=device
        )
        
        # Record a few batches for inspection
        logger.info("Extracting sample batches from dataloader...")
        for i, batch_data in enumerate(dataloader):
            if i >= batch_hook.max_batches:
                break
            batch_hook.batches.append(batch_data)
        
        # Examine the structure of the batches
        if batch_hook.batches:
            logger.info(f"Dataloader produced {len(batch_hook.batches)} batches")
            
            # Inspect first batch
            sample_batch = batch_hook.batches[0]
            logger.info(f"Batch structure: {type(sample_batch)}, length: {len(sample_batch)}")
            
            # Unpack batch data
            if len(sample_batch) == 3:
                data, targets, single_eval_pos = sample_batch
                logger.info("Standard batch format: (data, targets, single_eval_pos)")
                batch_info = None
            elif len(sample_batch) == 4:
                data, targets, single_eval_pos, batch_info = sample_batch
                logger.info("Extended batch format: (data, targets, single_eval_pos, info)")
            else:
                logger.error(f"Unexpected batch format with {len(sample_batch)} elements")
                return False
            
            # Check for semantic features in the data
            if isinstance(data, tuple) and len(data) == 3 and isinstance(data[0], dict):
                # This suggests it's the extended format with semantic info
                info_dict, X, y = data
                logger.info(f"Data contains semantic info dictionary with keys: {list(info_dict.keys())}")
                
                # Check for semantic targets
                semantic_info_present = False
                if 'semantic_targets' in info_dict:
                    semantic_info_present = True
                    logger.info(f"Found semantic_targets with shape: {info_dict['semantic_targets'].shape}")
                
                if 'semantic_class_targets' in info_dict:
                    semantic_info_present = True
                    logger.info(f"Found semantic_class_targets with shape: {info_dict['semantic_class_targets'].shape}")
                
                if semantic_info_present:
                    logger.info("✅ Semantic information is present in dataloader output")
                else:
                    logger.warning("❌ No semantic information found in dataloader output")
            
            # Create a basic TabPFN model
            logger.info("Building model...")
            from ticl.models.encoders import Linear
            from ticl.models.tabpfn import TabPFN
            
            # Create Y encoder (needed for TabPFN)
            y_encoder = Linear(1, emsize=128)
            
            # Create TabPFN with the total feature count
            feature_count = config['prior']['num_features']
            if config['prior']['classification']['semantic_feature_p'] > 0:
                feature_count += 50  # Add semantic features
                
            base_model = TabPFN(
                n_out=2,
                n_features=feature_count,
                y_encoder_layer=y_encoder,
                semantic_feature_p=0.3,
                emsize=128,
                nlayers=2,
                nhead=4,
                nhid_factor=4
            )
            
            # Wrap with SemanticAwareClassifier
            logger.info("Creating semantic-aware model wrapper...")
            model = create_semantic_aware_model(base_model, num_semantic_classes=3)
            model.to(device)
            model.eval()  # Set to eval mode for testing
            
            # Register a forward hook on the model to capture outputs
            handle = model.register_forward_hook(hook)
            
            # Create class text descriptions for the forward pass
            class_texts = [
                "This is a sample of class 0, representing a negative example",
                "This is a sample of class 1, representing a positive example",
                "This is a sample of an extra class for testing purposes"
            ]
            
            # Set up autocast context according to precision setting
            ctx = get_autocast_context(mixed_precision, "bfloat16" if mixed_precision else None, device)
            
            # Process a sample batch through the model
            logger.info(f"Processing a batch through the model with autocast={mixed_precision}...")
            
            with ctx:
                try:
                    # Use the data from the dataloader
                    # Note: single_eval_pos is already in the batch
                    model_input = data  # Should be a tuple as TabPFN expects
                    output = model(model_input, single_eval_pos=single_eval_pos, class_texts=class_texts)
                    logger.info("✅ Model forward pass succeeded")
                except Exception as e:
                    logger.error(f"❌ Model forward pass failed: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    return False
            
            # Remove hook
            handle.remove()
            
            # Check if hook captured outputs
            if hook.outputs is None:
                logger.error("Hook did not capture any outputs")
                return False
            
            # Log the outputs
            logger.info(f"Model output keys: {list(hook.outputs.keys())}")
            
            # Check for the specific feature tensors
            features_present = []
            for key in ['tabular_features', 'text_features', 'semantic_features', 'class_logits', 'semantic_logits']:
                if key in hook.outputs and hook.outputs[key] is not None:
                    features_present.append(key)
                    shape = hook.outputs[key].shape if hasattr(hook.outputs[key], 'shape') else 'N/A'
                    try:
                        mean = hook.outputs[key].float().mean().item() if hasattr(hook.outputs[key], 'float') else 'N/A'
                        std = hook.outputs[key].float().std().item() if hasattr(hook.outputs[key], 'float') else 'N/A'
                        logger.info(f"{key}: shape={shape}, mean={mean}, std={std}")
                    except:
                        logger.info(f"{key}: shape={shape}, stats unavailable")
                else:
                    logger.warning(f"{key}: NOT PRESENT OR NONE")
            
            # Check for essential semantic components
            essential_components = ['tabular_features', 'text_features']
            missing_components = [c for c in essential_components if c not in features_present]
            
            if missing_components:
                logger.error(f"❌ Missing essential components: {missing_components}")
                return False
            else:
                logger.info("✅ All essential semantic components are present")
                return True
        else:
            logger.error("No batches were produced by the dataloader")
            return False
            
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def run_test():
    """Run all tests."""
    # Test with real dataloader in both precision modes
    precision_true = test_model_with_real_dataloader(mixed_precision=True)
    precision_false = test_model_with_real_dataloader(mixed_precision=False)
    
    # Report results
    logger.info("\n" + "="*80)
    logger.info("DATALOADER TEST SUMMARY")
    logger.info("="*80)
    logger.info(f"Mixed precision=True:  {'PASS' if precision_true else 'FAIL'}")
    logger.info(f"Mixed precision=False: {'PASS' if precision_false else 'FAIL'}")
    logger.info("="*80)
    
    # If they differ, highlight the difference
    if precision_true != precision_false:
        logger.info("\n⚠️ IMPORTANT: Test results differ between precision modes!")
        logger.info("This confirms that mixed precision setting affects semantic feature processing")
        logger.info("The issue is in the interaction between mixed precision and the data loading pipeline")
    else:
        logger.info("\nBoth precision modes behave the same in this controlled test")
        logger.info("The issue may be more subtle and occur in the full training loop")
    
    return precision_true and precision_false

if __name__ == "__main__":
    run_test()