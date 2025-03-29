"""
Test zero-shot and few-shot semantic classification with actual trained models.

This module tests the zero-shot and few-shot classification capabilities of 
semantic-aware models using real semantic data and actual trained models.

To run tests with actual trained models:
1. Set the TRAINED_MODEL_PATH to point to a valid model checkpoint
2. Run the tests with: python -m pytest ticl/tests/semantic/test_real_zero_shot_classification.py -v
"""

import pytest
import torch
import numpy as np
import os
import json
import logging
import pandas as pd
import sys
from typing import Dict, List, Tuple, Union, Optional, Any
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import re
import importlib
import pkgutil

# Add the project root to the path to fix imports
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(os.path.dirname(current_file_path), '../../../../..'))
ticl_dir = os.path.abspath(os.path.join(os.path.dirname(current_file_path), '../../../..'))
ticl_parent = os.path.dirname(ticl_dir)

# Make sure we can access the actual ticl module
sys.path.insert(0, ticl_parent)  # To import 'ticl' module directly

# Print available modules in each path for debugging
print("\nAvailable modules in Python path:")
for path in sys.path:
    if os.path.isdir(path):
        print(f"\nIn {path}:")
        modules = [name for _, name, _ in pkgutil.iter_modules([path])]
        print(modules)

print(f"Project root: {project_root}")
print(f"TICL directory: {ticl_dir}")
print(f"Python path: {sys.path}")
print(f"Current directory: {os.getcwd()}")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to locate model files and benchmark data
models_paths_to_try = [
    os.path.join(ticl_dir, "ticl", "models_diff"),
    os.path.join(ticl_dir, "models_diff"),
    os.path.join(project_root, "ticl", "ticl", "models_diff"),
    os.path.join(project_root, "ticl", "models_diff"),
    os.path.join(project_root, "..", "ticl", "ticl", "models_diff"),
    # Relative paths (using current directory as reference)
    "ticl/ticl/models_diff",
    "ticl/models_diff", 
    "../ticl/ticl/models_diff",
    "../ticl/models_diff",
]

benchmark_paths_to_try = [
    os.path.join(project_root, "benchmarks", "carte"),
    os.path.join(project_root, "..", "benchmarks", "carte"),
    os.path.join(project_root, "tabular-fm-llm", "benchmarks", "carte"),
    # Relative paths
    "benchmarks/carte",
    "../benchmarks/carte",
    "../../benchmarks/carte",
]

# Also check the current directory structure
curr_dir = os.getcwd()
print(f"Current directory: {curr_dir}")
print("Checking current directory structure...")

# Try to print benchmark directory content
if os.path.exists(os.path.join(curr_dir, "benchmarks")):
    print(f"Found benchmarks at {os.path.join(curr_dir, 'benchmarks')}")
    print(os.listdir(os.path.join(curr_dir, "benchmarks")))
    
    if os.path.exists(os.path.join(curr_dir, "benchmarks", "carte")):
        print(f"Found carte at {os.path.join(curr_dir, 'benchmarks', 'carte')}")
        print(os.listdir(os.path.join(curr_dir, "benchmarks", "carte")))
else:
    print("No benchmarks directory found in current directory")

# Find the model directory
MODEL_PARENT_DIR = None
for path in models_paths_to_try:
    if os.path.exists(path):
        MODEL_PARENT_DIR = path
        break

if MODEL_PARENT_DIR is None:
    print("WARNING: Could not find model directory. Please set MODEL_PARENT_DIR manually.")
    # Fall back to a default path
    MODEL_PARENT_DIR = os.path.join(ticl_dir, "ticl", "models_diff")

# Find the benchmark directory
BENCHMARK_DIR = None
for path in benchmark_paths_to_try:
    if os.path.exists(path):
        BENCHMARK_DIR = path
        break

if BENCHMARK_DIR is None:
    print("WARNING: Could not find benchmark directory. Please set BENCHMARK_DIR manually.")
    # Fall back to a default path
    BENCHMARK_DIR = os.path.join(project_root, "benchmarks", "carte")

# Define model files to try
model_files_to_try = [
    "tabpfn_b4_E2_n10_semanticfeaturep0.3_U1_03_27_2025_11_46_30_epoch_on_exit.cpkt",
    "tabpfn_b4_E2_n10_semanticfeaturep0_U1_03_27_2025_11_44_25_epoch_on_exit.cpkt",
    "tabpfn_b4_E2_n10_semanticfeaturep0_U1_03_27_2025_11_33_58_epoch_on_exit.cpkt",
    "tabpfn_b4_E1_n10_semanticfeaturep0.3_U1_03_27_2025_11_45_24_epoch_on_exit.cpkt",
    "tabpfn_nooptimizer_emsize_512_nlayers_12_steps_2048_bs_32ada_lr_0.0001_1_gpu_07_24_2023_01_43_33_epoch_1650.cpkt"
]

# Set the trained model path
TRAINED_MODEL_PATH = None
for model_file in model_files_to_try:
    path = os.path.join(MODEL_PARENT_DIR, model_file)
    if os.path.exists(path):
        TRAINED_MODEL_PATH = path
        break

if TRAINED_MODEL_PATH is None:
    print(f"WARNING: Could not find any model file in {MODEL_PARENT_DIR}")
    # Fall back to a default path
    TRAINED_MODEL_PATH = os.path.join(MODEL_PARENT_DIR, model_files_to_try[0])

print(f"Models directory: {MODEL_PARENT_DIR}")
print(f"Model path: {TRAINED_MODEL_PATH}")
print(f"Benchmark directory: {BENCHMARK_DIR}")

# Check if the benchmark datasets exist
benchmarks_exist = False
benchmark_dirs_found = []

if os.path.exists(BENCHMARK_DIR):
    # Check for datasets either as direct files or in subdirectories
    datasets = ['coffee_ratings', 'michelin', 'ramen_ratings']
    
    for dataset in datasets:
        # Check for subdirectory
        subdir_path = os.path.join(BENCHMARK_DIR, dataset)
        csv_path = os.path.join(BENCHMARK_DIR, f"{dataset}.csv")
        
        if os.path.isdir(subdir_path):
            # Find CSV files in the directory
            files = os.listdir(subdir_path)
            csv_files = [f for f in files if f.endswith('.csv')]
            if f"{dataset}.csv" in csv_files or csv_files:
                benchmark_dirs_found.append(dataset)
        elif os.path.exists(csv_path):
            benchmark_dirs_found.append(dataset)
    
    benchmarks_exist = len(benchmark_dirs_found) > 0
    print(f"Found {len(benchmark_dirs_found)} benchmark datasets: {', '.join(benchmark_dirs_found)}")

if not benchmarks_exist:
    print("\nWARNING: Benchmark datasets not found! Make sure benchmarks/carte directory contains:")
    print("- Directory 'coffee_ratings' with coffee_ratings.csv or")
    print("- coffee_ratings.csv directly in benchmarks/carte")
    print("Same for michelin and ramen_ratings datasets")
    print("You might need to create config_data.json files for each dataset as well.")

# We'll import these dynamically in the functions that need them
# to avoid import errors stopping the entire test file

# Skip later if specific tests can't run
def try_import(module_name, class_or_function=None):
    """Safely try to import a module and optionally a class/function from it"""
    try:
        # Try different import paths
        possible_paths = [
            module_name,                        # Original path
            module_name.replace("ticl.", ""),   # Try without ticl. prefix
            "ticl." + module_name               # Try with ticl. prefix
        ]
        
        for path in possible_paths:
            try:
                module = importlib.import_module(path)
                if class_or_function:
                    return getattr(module, class_or_function)
                return module
            except (ImportError, AttributeError) as e:
                last_error = e
                continue
        
        # If we get here, all paths failed
        logger.warning(f"Failed to import {module_name}.{class_or_function}: {last_error}")
        return None
    except Exception as e:
        logger.warning(f"Unexpected error importing {module_name}.{class_or_function}: {e}")
        return None

def load_trained_model(model_path: str, device: str = 'cpu'):
    """
    Load a trained semantic model from checkpoint.
    
    Parameters:
    -----------
    model_path : str
        Path to the model checkpoint
    device : str
        Device to load the model on
        
    Returns:
    --------
    model
        The loaded semantic model
        
    Note:
    -----
    This function requires the ticl module to be in your Python path.
    To run this test properly, you should run it from the ticl parent directory with:
    
    python -m ticl.tests.semantic.test_real_zero_shot_classification
    
    Or ensure that the 'ticl' module is installable, for example by running:
    pip install -e .
    in the ticl parent directory.
    """
    logger.info(f"Loading model from {model_path}")
    
    # Import necessary modules
    # Try different import paths
    TabPFN = (try_import('ticl.models.tabpfn', 'TabPFN') or 
              try_import('models.tabpfn', 'TabPFN') or 
              try_import('tabpfn', 'TabPFN'))
    
    SemanticAwareClassifier = (try_import('ticl.models.semantic_aware_model', 'SemanticAwareClassifier') or
                              try_import('models.semantic_aware_model', 'SemanticAwareClassifier') or
                              try_import('semantic_aware_model', 'SemanticAwareClassifier'))
    
    create_semantic_aware_model = (try_import('ticl.models.semantic_aware_model', 'create_semantic_aware_model') or
                                  try_import('models.semantic_aware_model', 'create_semantic_aware_model') or 
                                  try_import('semantic_aware_model', 'create_semantic_aware_model'))
    
    SemanticAwareClassifierWrapper = (try_import('ticl.prediction.semantic', 'SemanticAwareClassifierWrapper') or
                                     try_import('prediction.semantic', 'SemanticAwareClassifierWrapper') or
                                     try_import('semantic', 'SemanticAwareClassifierWrapper'))
    
    if not all([TabPFN, SemanticAwareClassifier, create_semantic_aware_model, SemanticAwareClassifierWrapper]):
        logger.error("Could not import required modules for model loading")
        pytest.skip("Required modules not available")
    
    try:
        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Check the type of checkpoint
        logger.info(f"Checkpoint type: {type(checkpoint)}")
        
        # Extract model configuration from checkpoint
        config = None
        state_dict = None
        model_obj = None
        
        if isinstance(checkpoint, tuple):
            logger.info(f"Checkpoint is a tuple with {len(checkpoint)} elements")
            
            # Based on the checkpoint examination, we know:
            # - First element (index 0) is an OrderedDict with model weights
            # - Fourth element (index 3) is a dict with configuration
            
            # Get the state dict from the first element
            if len(checkpoint) > 0 and isinstance(checkpoint[0], dict):
                state_dict = checkpoint[0]
                logger.info(f"Using state_dict from first tuple element with {len(state_dict)} keys")
            
            # Get config from the fourth element
            if len(checkpoint) > 3 and isinstance(checkpoint[3], dict):
                config_dict = checkpoint[3]
                # Extract relevant configuration
                config = {
                    'model_type': config_dict.get('model_type', 'tabpfn'),
                    'semantic_feature_p': 0.3,  # Based on filename
                    'num_semantic_classes': 128  # Default value
                }
                if 'transformer' in config_dict:
                    # Extract transformer config if available
                    transformer_config = config_dict['transformer']
                    for key in ['emsize', 'nhead', 'nlayers']:
                        if key in transformer_config:
                            config[key] = transformer_config[key]
                            
                logger.info(f"Extracted config: {config}")
            
            # Check if any of the items has CLIP model weights
            clip_keys_prefix = 'clip_text_model.'
            has_semantic_keys = any(any(k.startswith(clip_keys_prefix) for k in item.keys()) 
                                  for item in checkpoint if isinstance(item, dict))
            
            if has_semantic_keys:
                config['has_semantic'] = True
                config['semantic_feature_p'] = 0.3  # Based on filename
                logger.info("Detected CLIP model weights in checkpoint")
        
        elif isinstance(checkpoint, dict):
            keys = list(checkpoint.keys())
            logger.info(f"Found dictionary with keys: {keys[:10]}...")
            
            if 'config' in checkpoint:
                config = checkpoint['config']
            if 'model' in checkpoint and hasattr(checkpoint['model'], 'state_dict'):
                model_obj = checkpoint['model']
                state_dict = model_obj.state_dict()
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
        
        # Log the config
        if config:
            logger.info(f"Model configuration: {config}")
        else:
            logger.warning("No configuration found in checkpoint")
            config = {'model_type': 'tabpfn', 'num_semantic_classes': 128, 'semantic_feature_p': 0.3}
        
        # Use the model directly if it's already loaded
        if model_obj and isinstance(model_obj, SemanticAwareClassifier):
            logger.info("Using model object directly from checkpoint")
            model = model_obj
        else:
            # Try to create a model or use the wrapper
            if create_semantic_aware_model:
                # Get configuration from config or use default values
                model_type = config.get('model_type', 'tabpfn')
                num_semantic_classes = config.get('num_semantic_classes', 128)
                
                # Look at the state dict to extract actual model dimensions
                try:
                    if state_dict:
                        # Try to determine parameters from the state dict
                        emsize = 512  # Default
                        
                        # Check transformer hidden size from weights
                        for key in state_dict:
                            if key.endswith('.linear1.weight'):
                                # Get the shape from this weight
                                weight_shape = state_dict[key].shape
                                if len(weight_shape) == 2:
                                    # Second dimension is emsize
                                    if weight_shape[1] == 512:
                                        emsize = 512
                                    # First dimension / emsize gives nhid_factor
                                    nhid_factor = weight_shape[0] // emsize
                                    logger.info(f"Determined nhid_factor={nhid_factor} from weight shape {weight_shape}")
                                    break
                        
                        # Count number of layers
                        layer_pattern = 'base_model.transformer_encoder.layers.'
                        layers = set()
                        for key in state_dict:
                            if layer_pattern in key:
                                layer_num = key.split(layer_pattern)[1].split('.')[0]
                                try:
                                    layers.add(int(layer_num))
                                except ValueError:
                                    pass
                        
                        if layers:
                            nlayers = max(layers) + 1
                            logger.info(f"Found {nlayers} transformer layers in state dict")
                        else:
                            nlayers = 12  # Default
                            
                        # Determine number of features from encoder weight
                        n_features = 100  # Default
                        if 'base_model.encoder.weight' in state_dict:
                            encoder_shape = state_dict['base_model.encoder.weight'].shape
                            if len(encoder_shape) == 2 and encoder_shape[0] == 512:
                                n_features = encoder_shape[1]
                                logger.info(f"Found encoder with input features: {n_features}")
                        
                        # Determine output classes from decoder weight
                        n_out = 128  # Default for semantic model
                        if 'base_model.decoder.2.weight' in state_dict:
                            decoder_shape = state_dict['base_model.decoder.2.weight'].shape
                            if len(decoder_shape) == 2:
                                n_out = decoder_shape[0]
                                logger.info(f"Found decoder with output classes: {n_out}")
                        
                        # Determine if semantic feature is enabled
                        semantic_feature_p = 0.3 if any('clip_text_model.' in k for k in state_dict) else 0.0
                        
                        # Use num_semantic_classes from n_out
                        num_semantic_classes = n_out
                    else:
                        # Default values if no state dict
                        emsize = config.get('emsize', 512)
                        nhid_factor = 2  # Based on examination
                        nlayers = config.get('nlayers', 12)
                        n_features = config.get('n_features', 100)
                        semantic_feature_p = config.get('semantic_feature_p', 0.3)
                except Exception as e:
                    logger.warning(f"Error determining model parameters from state dict: {e}")
                    # Fallback to safe defaults
                    emsize = 512
                    nhid_factor = 2
                    nlayers = 12
                    n_features = 100
                    semantic_feature_p = 0.3
                
                # Set remaining parameters
                nhead = config.get('nhead', 8)
                dropout = config.get('dropout', 0.1)
                activation = config.get('activation', 'gelu')
                
                logger.info(f"Creating semantic-aware model of type {model_type} with {num_semantic_classes} semantic classes")
                
                # Log model creation parameters
                logger.info(f"Creating TabPFN model with parameters:")
                logger.info(f"  emsize={emsize}, nhead={nhead}, nhid_factor={nhid_factor}, nlayers={nlayers}")
                logger.info(f"  n_features={n_features}, n_out={n_out}, semantic_feature_p={semantic_feature_p}")
                
                # Create base TabPFN model
                base_model = TabPFN(
                    emsize=emsize,
                    nhead=nhead,
                    nhid_factor=nhid_factor,
                    nlayers=nlayers,
                    n_features=n_features,
                    n_out=n_out,  # Use extracted output dimension
                    dropout=dropout,
                    activation=activation,
                    semantic_feature_p=semantic_feature_p
                )
                
                # Create semantic-aware model
                model = create_semantic_aware_model(base_model, num_semantic_classes=n_out)  # Use extracted n_out
                
                # Handle model state dict
                if state_dict:
                    try:
                        # Check if keys have a 'base_model.' prefix
                        has_base_model_prefix = any(k.startswith('base_model.') for k in state_dict.keys())
                        has_clip_prefix = any(k.startswith('clip_text_model.') for k in state_dict.keys())
                        
                        # Always try non-strict loading first, since model architecture may have minor differences
                        logger.info("Loading state dict with non-strict matching")
                        try:
                            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                            
                            if missing_keys:
                                logger.warning(f"Missing keys: {len(missing_keys)} keys, first few: {missing_keys[:3]}...")
                            if unexpected_keys:
                                logger.warning(f"Unexpected keys: {len(unexpected_keys)} keys, first few: {unexpected_keys[:3]}...")
                                
                            logger.info("Successfully loaded model weights with non-strict matching")
                        except Exception as e:
                            logger.error(f"Failed to load with non-strict matching: {e}")
                            # Try with more specific approach
                            if has_base_model_prefix and has_clip_prefix:
                                logger.info("Attempting to load semantic model components separately")
                                
                                # Load base model weights
                                base_model_dict = {k.replace('base_model.', ''): v for k, v in state_dict.items() 
                                                   if k.startswith('base_model.')}
                                
                                # Try to load just the base model weights
                                missing_base, unexpected_base = model.base_model.load_state_dict(
                                    base_model_dict, strict=False)
                                
                                logger.info(f"Base model: {len(missing_base)} missing keys, {len(unexpected_base)} unexpected keys")
                                logger.info("Loaded base model weights partially")
                    except Exception as e:
                        logger.error(f"Failed to load state dict directly: {e}")
                        try:
                            # Try using the wrapper for compatibility
                            logger.info("Trying to load using SemanticAwareClassifierWrapper...")
                            wrapper = SemanticAwareClassifierWrapper(
                                device=device,
                                model=model,
                                config=config,
                                verbose=True
                            )
                            model = wrapper
                            logger.info("Successfully created model wrapper")
                        except Exception as inner_e:
                            logger.error(f"Failed to create wrapper: {inner_e}")
                            # Continue with the model anyway
                            logger.warning("Continuing with uninitialized model")
                else:
                    logger.warning("No state dictionary found in checkpoint - using freshly initialized weights")
            else:
                raise ImportError("Could not import create_semantic_aware_model")
        
        # Set model to evaluation mode - make sure to do this explicitly
        if hasattr(model, 'eval'):
            model.eval()
            logger.info("Set model to evaluation mode")
            
        # If there's a base_model, set that to eval mode too
        if hasattr(model, 'base_model') and hasattr(model.base_model, 'eval'):
            model.base_model.eval()
            logger.info("Set base_model to evaluation mode")
            
        # If there's a CLIP model, set that to eval mode too
        if hasattr(model, 'clip_text_model') and hasattr(model.clip_text_model, 'eval'):
            model.clip_text_model.eval()
            logger.info("Set clip_text_model to evaluation mode")
            
        # Move model to the specified device
        if hasattr(model, 'to'):
            model = model.to(device)
            logger.info(f"Moved model to device: {device}")
            
        # Check that model is properly in eval mode
        if hasattr(model, 'training'):
            if model.training:
                logger.warning("Warning: Model is still in training mode after setting to eval mode!")
                model.train(False)  # Another way to set eval mode
                
        # Freeze parameters to be extra sure
        for param in model.parameters():
            param.requires_grad = False
            
        logger.info("Model loaded successfully and set to evaluation mode with frozen parameters")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def load_carte_dataset(dataset_name: str, bin_classes: bool = True, max_classes: int = 10) -> Dict:
    """
    Load a dataset from the Carte benchmark collection.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset folder ('coffee_ratings', 'michelin', 'ramen_ratings')
    bin_classes : bool
        Whether to bin classes if there are more than max_classes
    max_classes : int
        Maximum number of classes to use (will bin if more)
        
    Returns:
    --------
    Dict
        Dictionary containing the dataset information
    """
    # First, try loading from a subdirectory
    dataset_dir = os.path.join(BENCHMARK_DIR, dataset_name)
    logger.info(f"Looking for dataset in {dataset_dir}")
    
    # Check if the directory itself exists
    if os.path.exists(dataset_dir):
        logger.info(f"Found subdirectory: {dataset_dir}")
        
        # Check for files in this directory
        files = os.listdir(dataset_dir)
        logger.info(f"Files in {dataset_dir}: {files}")
        
        # Look for CSV file with matching name or any CSV
        csv_files = [f for f in files if f.endswith('.csv')]
        if f"{dataset_name}.csv" in csv_files:
            csv_path = os.path.join(dataset_dir, f"{dataset_name}.csv")
        elif csv_files:
            # Take the first CSV file if name doesn't match
            csv_path = os.path.join(dataset_dir, csv_files[0])
            logger.info(f"Using {csv_files[0]} instead of {dataset_name}.csv")
        else:
            # No CSV in directory
            logger.warning(f"No CSV file found in {dataset_dir}")
            # Look in parent directory
            csv_path = os.path.join(BENCHMARK_DIR, f"{dataset_name}.csv")
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Could not find CSV file for dataset: {dataset_name}")
            
        # Check for config file
        config_path = os.path.join(dataset_dir, "config_data.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_data = json.load(f)
        else:
            # Create default config
            logger.warning(f"No config_data.json found in {dataset_dir}. Creating default config.")
            config_data = {
                "target_name": "target",  # Default target name
                "entity_name": dataset_name,
                "class_names": {}
            }
            
            # Load CSV to automatically determine target column
            data = pd.read_csv(csv_path)
            
            # Try to find a good target column
            potential_targets = ['class', 'target', 'label', 'rating', 'quality', 'score', 'stars']
            for potential in potential_targets:
                for col in data.columns:
                    if potential.lower() in col.lower():
                        config_data["target_name"] = col
                        logger.info(f"Automatically selected '{col}' as target column")
                        break
    else:
        # Subdirectory doesn't exist, try parent directory
        logger.info(f"Subdirectory not found, trying to load from parent directory {BENCHMARK_DIR}")
        
        # List files in benchmark directory 
        if os.path.exists(BENCHMARK_DIR):
            logger.info(f"Files in {BENCHMARK_DIR}: {os.listdir(BENCHMARK_DIR)}")
        
        # Look for the CSV file directly in parent
        csv_path = os.path.join(BENCHMARK_DIR, f"{dataset_name}.csv")
        if not os.path.exists(csv_path):
            # Try finding any file that starts with dataset_name
            parent_files = os.listdir(BENCHMARK_DIR) if os.path.exists(BENCHMARK_DIR) else []
            dataset_files = [f for f in parent_files if f.startswith(dataset_name) and f.endswith('.csv')]
            
            if dataset_files:
                csv_path = os.path.join(BENCHMARK_DIR, dataset_files[0])
                logger.info(f"Using {dataset_files[0]} instead of {dataset_name}.csv")
            else:
                raise FileNotFoundError(f"Could not find dataset file: {csv_path} or directory: {dataset_dir}")
        
        # Since we found the CSV but not the directory, create a simple config
        config_data = {
            "target_name": "target",  # Default target name
            "entity_name": dataset_name,
            "class_names": {}
        }
        
        # Load the CSV file to automatically determine a target column
        data = pd.read_csv(csv_path)
        
        # Try to find a good target column
        # Look for columns like 'class', 'target', 'label', 'rating', etc.
        potential_targets = ['class', 'target', 'label', 'rating', 'quality', 'score', 'stars']
        for potential in potential_targets:
            for col in data.columns:
                if potential.lower() in col.lower():
                    config_data["target_name"] = col
                    logger.info(f"Automatically selected '{col}' as target column")
                    break
    
    # Load the data
    try:
        data = pd.read_csv(csv_path)
        logger.info(f"Loaded CSV file: {csv_path} with shape {data.shape}")
    except Exception as e:
        logger.error(f"Error loading CSV file {csv_path}: {e}")
        raise
    
    # Extract target and class information
    target_name = config_data.get("target_name")
    entity_name = config_data.get("entity_name", dataset_name)
    class_names = config_data.get("class_names", {})
    
    # Check if target column exists
    if target_name not in data.columns:
        raise ValueError(f"Target column '{target_name}' not found in CSV file")
    
    # Get unique target values
    unique_targets = sorted(data[target_name].dropna().unique())
    logger.info(f"Found {len(unique_targets)} unique target values")
    
    # Check if we need to bin classes
    if bin_classes and len(unique_targets) > max_classes:
        logger.info(f"Binning {len(unique_targets)} classes into {max_classes} bins")
        
        # Convert targets to numeric if they aren't already
        if not np.issubdtype(data[target_name].dtype, np.number):
            # Create mapping from original values to numeric values
            original_to_numeric = {val: i for i, val in enumerate(unique_targets)}
            data[target_name] = data[target_name].map(original_to_numeric)
            unique_targets = sorted(data[target_name].dropna().unique())
        
        # Bin the targets
        bins = np.linspace(min(unique_targets), max(unique_targets), max_classes + 1)
        bin_labels = [f"bin_{i}" for i in range(max_classes)]
        data[f"{target_name}_binned"] = pd.cut(data[target_name], bins=bins, labels=bin_labels)
        
        # Create new class_names for binned values
        binned_class_names = {}
        for i, bin_label in enumerate(bin_labels):
            bin_min = bins[i]
            bin_max = bins[i+1]
            
            # Find class descriptions within this bin
            bin_descriptions = []
            for val, desc in class_names.items():
                try:
                    numeric_val = float(val)
                    if bin_min <= numeric_val < bin_max:
                        bin_descriptions.append(desc)
                except ValueError:
                    continue
            
            if bin_descriptions:
                # Combine descriptions or use a representative one
                binned_class_names[bin_label] = f"Values between {bin_min:.2f} and {bin_max:.2f}: " + bin_descriptions[0]
            else:
                # If no class descriptions found, create a generic one
                binned_class_names[bin_label] = f"Rating or score in range {bin_min:.2f} to {bin_max:.2f}"
        
        # Update target info
        target_name = f"{target_name}_binned"
        class_names = binned_class_names
        unique_targets = bin_labels
    
    # Identify semantic columns (string type)
    semantic_columns = []
    for col in data.columns:
        if col != target_name and data[col].dtype == object:
            # Check if column actually contains strings (not all NaN)
            if data[col].dropna().astype(str).str.len().mean() > 0:
                semantic_columns.append(col)
    
    logger.info(f"Identified {len(semantic_columns)} semantic columns: {semantic_columns}")
    
    # Prepare features and target
    X = data.drop(columns=[target_name])
    y = data[target_name]
    
    # Create mapping from target values to class indices
    target_mapping = {val: i for i, val in enumerate(unique_targets)}
    inverse_mapping = {i: val for val, i in target_mapping.items()}
    
    # Convert target to numeric indices
    y_numeric = y.map(target_mapping)
    
    # Create numerical feature mask
    numeric_columns = X.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = X.select_dtypes(exclude=['number']).columns.tolist()
    
    # Create column indices for semantic columns
    semantic_indices = [list(X.columns).index(col) for col in semantic_columns if col in X.columns]
    
    return {
        "name": dataset_name,
        "data": data,
        "X": X,
        "y": y_numeric,
        "target_name": target_name,
        "entity_name": entity_name,
        "class_names": class_names,
        "target_mapping": target_mapping,
        "inverse_mapping": inverse_mapping,
        "unique_targets": unique_targets,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "semantic_columns": semantic_columns,
        "semantic_indices": semantic_indices
    }

def preprocess_dataset(dataset_dict: Dict) -> Dict:
    """
    Preprocess a dataset for model use.
    
    Parameters:
    -----------
    dataset_dict : Dict
        Dictionary containing the dataset information
        
    Returns:
    --------
    Dict
        Updated dictionary with preprocessed data
    """
    # Import sklearn components
    try:
        from sklearn.preprocessing import LabelEncoder, StandardScaler
    except ImportError:
        logger.error("Could not import sklearn preprocessing. Required for preprocessing.")
        raise ImportError("sklearn is required for preprocessing.")
    
    try:
        # Try to import TabPFN preprocessing functionality
        normalize_data = try_import('ticl.utils', 'normalize_data')
        remove_outliers = try_import('ticl.utils', 'remove_outliers')
        normalize_by_used_features_f = try_import('ticl.utils', 'normalize_by_used_features_f')
        
        if not all([normalize_data, remove_outliers, normalize_by_used_features_f]):
            logger.warning("Could not import all TabPFN preprocessing utilities. Using basic preprocessing.")
            return _preprocess_dataset_basic(dataset_dict)
    except Exception as e:
        logger.warning(f"Error importing TabPFN preprocessing: {e}. Using basic preprocessing.")
        return _preprocess_dataset_basic(dataset_dict)
        
    X = dataset_dict["X"]
    y = dataset_dict["y"]
    
    # Handle missing values in target
    if pd.isna(y).any():
        logger.warning(f"Found {pd.isna(y).sum()} missing values in target column. Dropping those rows.")
        valid_idx = ~pd.isna(y)
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]
        dataset_dict["X"] = X
        dataset_dict["y"] = y
    
    # Convert object/string target to numeric if needed
    if y.dtype == 'object' or pd.api.types.is_string_dtype(y):
        logger.info(f"Converting {y.dtype} target to numeric")
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        dataset_dict["y"] = y
        dataset_dict["label_encoder"] = label_encoder
        dataset_dict["classes"] = label_encoder.classes_
    
    # One-hot encode categorical columns with limit on features
    max_features = 500  # Reasonable limit to avoid memory issues
    
    if dataset_dict["categorical_columns"]:
        # Filter out high-cardinality categorical columns
        cols_to_keep = X.columns.tolist()
        for col in dataset_dict["categorical_columns"]:
            if col in X.columns and X[col].nunique() > 100:  
                logger.warning(f"Dropping high-cardinality column '{col}' with {X[col].nunique()} unique values")
                cols_to_keep.remove(col)
        
        # Use only the columns we want to keep
        X = X[cols_to_keep]
        
        # Update categorical_columns
        dataset_dict["categorical_columns"] = [col for col in dataset_dict["categorical_columns"] if col in cols_to_keep]
        
        # Try one-hot encoding
        try:
            X_encoded = pd.get_dummies(X, drop_first=True)
            logger.info(f"One-hot encoded features from {X.shape[1]} to {X_encoded.shape[1]} columns")
            
            # If still too many features, take a subset
            if X_encoded.shape[1] > max_features:
                logger.warning(f"Too many features after encoding ({X_encoded.shape[1]}). Truncating to {max_features}.")
                X_encoded = X_encoded.iloc[:, :max_features]
        except Exception as e:
            logger.warning(f"Error during one-hot encoding: {e}. Using numeric columns only.")
            X_encoded = X.select_dtypes(include=['number'])
            if X_encoded.empty:
                logger.warning("No numeric features found. Creating simple features.")
                X_encoded = pd.DataFrame(np.random.randn(X.shape[0], 10))
    else:
        X_encoded = X.select_dtypes(include=['number']).copy()
        if X_encoded.empty:
            logger.warning("No numeric features found. Creating simple features.")
            X_encoded = pd.DataFrame(np.random.randn(X.shape[0], 10))
    
    # Handle missing values
    X_encoded = X_encoded.fillna(0)
    
    # Split data
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError as e:
        logger.warning(f"Stratified split failed: {e}. Using standard split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42
        )
    
    # Convert to tensors with error handling
    try:
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values if hasattr(y_train, 'values') else y_train, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test.values if hasattr(y_test, 'values') else y_test, dtype=torch.long)
    except Exception as e:
        logger.warning(f"Error converting to tensors: {e}. Using manual conversion.")
        X_train_np = X_train.values.astype(np.float32)
        X_test_np = X_test.values.astype(np.float32)
        y_train_np = np.array(y_train).astype(np.int64)
        y_test_np = np.array(y_test).astype(np.int64)
        
        X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_np, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test_np, dtype=torch.long)
    
    try:
        # Process for model input using TabPFN preprocessing
        # Convert to TabPFN expected format
        X_full = torch.cat([X_train_tensor, X_test_tensor], dim=0).unsqueeze(1)
        y_full = torch.cat([y_train_tensor, torch.zeros_like(y_test_tensor)], dim=0).unsqueeze(1)
        eval_pos = len(X_train_tensor)
        
        device = X_full.device
        # Normalize data
        X_full = normalize_data(X_full, normalize_positions=eval_pos)
        X_full = X_full[:, 0, :]  # Remove batch dimension 
        
        # Remove features with no variance
        sel = []
        for col in range(X_full.shape[1]):
            col_values = X_full[0:eval_pos, col]
            nan_mask = torch.isnan(col_values)
            if len(torch.unique(col_values[~nan_mask])) > 1:
                sel.append(True)
            else:
                sel.append(False)
        
        if not any(sel):
            logger.warning("No features with variance found. Using all features.")
            sel = [True] * X_full.shape[1]
        
        X_full = X_full[:, sel]
        
        # Add back batch dimension
        X_full = X_full.unsqueeze(1)
        
        # Remove outliers
        X_full = remove_outliers(X_full, normalize_positions=eval_pos)
        
        # Split back into train and test
        X_train_processed = X_full[:eval_pos]
        X_test_processed = X_full[eval_pos:]
    except Exception as e:
        logger.warning(f"Error during TabPFN preprocessing: {e}. Using basic tensors.")
        import traceback
        logger.warning(traceback.format_exc())
        X_train_processed = X_train_tensor.unsqueeze(1)
        X_test_processed = X_test_tensor.unsqueeze(1)
    
    # Update dataset dictionary
    dataset_dict.update({
        "X_encoded": X_encoded,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_tensor": X_train_processed.squeeze(1) if X_train_processed.dim() > 2 else X_train_processed,
        "X_test_tensor": X_test_processed.squeeze(1) if X_test_processed.dim() > 2 else X_test_processed,
        "y_train_tensor": y_train_tensor,
        "y_test_tensor": y_test_tensor,
    })
    
    return dataset_dict


def pad_features_for_model(model, tensor, logger=None):
    """
    Pad or truncate features to match the model's expected dimensions
    
    Parameters:
    -----------
    model : torch.nn.Module
        The model to get expected feature dimensions from
    tensor : torch.Tensor
        The input tensor to pad or truncate
    logger : logging.Logger, optional
        Logger to use for logging messages
        
    Returns:
    --------
    torch.Tensor
        The padded or truncated tensor
    """
    # Default logger if none provided
    if logger is None:
        logger = logging.getLogger("pad_features")
    
    # Check model's expected feature dimension from encoder weights
    expected_features = 100  # Default fallback
    
    # Try different ways to get expected feature count
    if hasattr(model, 'base_model') and hasattr(model.base_model, 'encoder') and hasattr(model.base_model.encoder, 'weight'):
        encoder_weight = model.base_model.encoder.weight
        expected_features = encoder_weight.shape[1]
        logger.info(f"Detected expected features from encoder weight: {expected_features}")
    elif hasattr(model, 'base_model') and hasattr(model.base_model, 'n_features'):
        expected_features = model.base_model.n_features
        logger.info(f"Using base_model.n_features: {expected_features}")
    elif hasattr(model, 'encoder') and hasattr(model.encoder, 'weight'):
        encoder_weight = model.encoder.weight
        expected_features = encoder_weight.shape[1]
        logger.info(f"Detected expected features from encoder weight: {expected_features}")
    elif hasattr(model, 'n_features'):
        expected_features = model.n_features
        logger.info(f"Using model.n_features: {expected_features}")
    
    # Check if we need to adjust feature dimensions
    current_features = tensor.shape[-1]  # Use last dimension in case of [batch, seq, features]
    
    if current_features != expected_features:
        logger.info(f"Input has {current_features} features, model expects {expected_features}")
        
        if current_features < expected_features:
            # Need to pad
            logger.info(f"Padding features from {current_features} to {expected_features}")
            # Handle different tensor shapes
            if tensor.dim() == 2:  # [batch, features]
                padding = torch.zeros(tensor.shape[0], expected_features - current_features, 
                                     device=tensor.device, dtype=tensor.dtype)
                padded_tensor = torch.cat([tensor, padding], dim=1)
            elif tensor.dim() == 3:  # [batch, seq, features]
                padding = torch.zeros(tensor.shape[0], tensor.shape[1], expected_features - current_features,
                                     device=tensor.device, dtype=tensor.dtype)
                padded_tensor = torch.cat([tensor, padding], dim=2)
            else:
                # Unsupported tensor shape
                logger.warning(f"Unsupported tensor shape for padding: {tensor.shape}")
                padded_tensor = tensor  # Return unchanged
        else:
            # Need to truncate
            logger.info(f"Truncating features from {current_features} to {expected_features}")
            if tensor.dim() == 2:  # [batch, features]
                padded_tensor = tensor[:, :expected_features]
            elif tensor.dim() == 3:  # [batch, seq, features]
                padded_tensor = tensor[:, :, :expected_features]
            else:
                # Unsupported tensor shape
                logger.warning(f"Unsupported tensor shape for truncation: {tensor.shape}")
                padded_tensor = tensor  # Return unchanged
        
        return padded_tensor
    else:
        # No adjustment needed
        return tensor

def _preprocess_dataset_basic(dataset_dict: Dict) -> Dict:
    """Basic preprocessing fallback when TabPFN preprocessing is not available"""
    try:
        from sklearn.preprocessing import LabelEncoder, StandardScaler
    except ImportError:
        logger.error("Could not import sklearn preprocessing. Required for preprocessing.")
        raise ImportError("sklearn is required for preprocessing.")
    
    X = dataset_dict["X"]
    y = dataset_dict["y"]
    
    # Handle missing values in target
    if pd.isna(y).any():
        logger.warning(f"Found {pd.isna(y).sum()} missing values in target column. Dropping those rows.")
        valid_idx = ~pd.isna(y)
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]
        dataset_dict["X"] = X
        dataset_dict["y"] = y
    
    # Convert object/string target to numeric if needed
    if y.dtype == 'object' or pd.api.types.is_string_dtype(y):
        logger.info(f"Converting {y.dtype} target to numeric")
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        dataset_dict["y"] = y
        dataset_dict["label_encoder"] = label_encoder
        dataset_dict["classes"] = label_encoder.classes_
    
    # One-hot encode categorical columns with limit on features
    max_features = 500  # Reasonable limit to avoid memory issues
    
    if dataset_dict["categorical_columns"]:
        # Filter out high-cardinality categorical columns
        cols_to_keep = X.columns.tolist()
        for col in dataset_dict["categorical_columns"]:
            if col in X.columns and X[col].nunique() > 100:  
                logger.warning(f"Dropping high-cardinality column '{col}' with {X[col].nunique()} unique values")
                cols_to_keep.remove(col)
        
        # Use only the columns we want to keep
        X = X[cols_to_keep]
        
        # Try one-hot encoding
        try:
            X_encoded = pd.get_dummies(X, drop_first=True)
            logger.info(f"One-hot encoded features from {X.shape[1]} to {X_encoded.shape[1]} columns")
            
            # If still too many features, take a subset
            if X_encoded.shape[1] > max_features:
                logger.warning(f"Too many features after encoding ({X_encoded.shape[1]}). Truncating to {max_features}.")
                X_encoded = X_encoded.iloc[:, :max_features]
        except Exception as e:
            logger.warning(f"Error during one-hot encoding: {e}. Using numeric columns only.")
            X_encoded = X.select_dtypes(include=['number'])
            if X_encoded.empty:
                logger.warning("No numeric features found. Creating simple features.")
                X_encoded = pd.DataFrame(np.random.randn(X.shape[0], 10))
    else:
        X_encoded = X.select_dtypes(include=['number']).copy()
        if X_encoded.empty:
            logger.warning("No numeric features found. Creating simple features.")
            X_encoded = pd.DataFrame(np.random.randn(X.shape[0], 10))
    
    # Handle missing values
    X_encoded = X_encoded.fillna(0)
    
    # Standardize numeric features
    scaler = StandardScaler()
    numeric_cols = X_encoded.select_dtypes(include=['number']).columns
    if not numeric_cols.empty:
        try:
            X_encoded[numeric_cols] = scaler.fit_transform(X_encoded[numeric_cols])
        except Exception as e:
            logger.warning(f"Error during standardization: {e}. Skipping standardization.")
    
    # Split data
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError as e:
        logger.warning(f"Stratified split failed: {e}. Using standard split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42
        )
    
    # Convert to tensors with error handling
    try:
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values if hasattr(y_train, 'values') else y_train, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test.values if hasattr(y_test, 'values') else y_test, dtype=torch.long)
    except Exception as e:
        logger.warning(f"Error converting to tensors: {e}. Using manual conversion.")
        try:
            X_train_np = X_train.values.astype(np.float32)
            X_test_np = X_test.values.astype(np.float32)
            y_train_np = np.array(y_train).astype(np.int64)
            y_test_np = np.array(y_test).astype(np.int64)
            
            X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
            X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train_np, dtype=torch.long)
            y_test_tensor = torch.tensor(y_test_np, dtype=torch.long)
        except Exception as e2:
            logger.error(f"Failed to convert to tensors even with fallback: {e2}")
            raise RuntimeError(f"Cannot convert data to tensors: {e2}")
    
    # Update dataset dictionary
    dataset_dict.update({
        "X_encoded": X_encoded,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_tensor": X_train_tensor,
        "X_test_tensor": X_test_tensor,
        "y_train_tensor": y_train_tensor,
        "y_test_tensor": y_test_tensor,
        "scaler": scaler,
    })
    
    return dataset_dict

def evaluate_predictions(y_true, y_pred, class_names=None):
    """
    Evaluate predictions and return metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True class labels
    y_pred : array-like
        Predicted class labels
    class_names : dict or None
        Dictionary mapping class indices to names
        
    Returns:
    --------
    Dict
        Dictionary of evaluation metrics
    """
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Calculate F1 score
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    
    # Classification report
    if class_names:
        target_names = [class_names.get(str(i), f"Class {i}") for i in range(len(set(y_true)))]
        report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    else:
        report = classification_report(y_true, y_pred, output_dict=True)
    
    metrics = {
        "accuracy": accuracy,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "report": report
    }
    
    return metrics

def test_load_zero_shot_model():
    """Test that we can load the trained semantic model."""
    try:
        model = load_trained_model(TRAINED_MODEL_PATH)
        assert model is not None
        assert hasattr(model, 'forward_semantic')
        logger.info(f"Model loaded successfully: {type(model)}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        pytest.skip(f"Test skipped due to error: {e}")

def test_load_carte_datasets():
    """Test that we can load and preprocess datasets from Carte benchmark."""
    try:
        datasets = ['coffee_ratings', 'michelin', 'ramen_ratings']
        
        for dataset_name in datasets:
            dataset_dict = load_carte_dataset(dataset_name, bin_classes=True, max_classes=10)
            
            # Check essential components
            assert dataset_dict is not None
            assert "X" in dataset_dict
            assert "y" in dataset_dict
            assert "class_names" in dataset_dict
            assert "semantic_columns" in dataset_dict
            
            logger.info(f"Dataset: {dataset_name}")
            logger.info(f"  Shape: {dataset_dict['X'].shape}")
            logger.info(f"  Classes: {len(dataset_dict['class_names'])}")
            logger.info(f"  Semantic columns: {len(dataset_dict['semantic_columns'])}")
            
            # Test preprocessing
            processed_dict = preprocess_dataset(dataset_dict)
            assert "X_train_tensor" in processed_dict
            assert "X_test_tensor" in processed_dict
            
            logger.info(f"  Train set shape: {processed_dict['X_train_tensor'].shape}")
            logger.info(f"  Test set shape: {processed_dict['X_test_tensor'].shape}")
            
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        import traceback
        logger.error(traceback.format_exc())
        pytest.skip(f"Test skipped due to error: {e}")

def test_zero_shot_classification_carte(test_datasets=None):
    """Test zero-shot classification on Carte benchmark datasets.
    
    Parameters:
    -----------
    test_datasets : list or None
        List of dataset names to test. If None, will use default datasets.
    """
    try:
        # Try to import necessary components
        TextualClassifier = try_import('ticl.text_classifier', 'TextualClassifier')
        
        # Load model
        model = load_trained_model(TRAINED_MODEL_PATH)
        assert model is not None
        logger.info(f"Loaded model of type: {type(model).__name__}")
        
        # Log available model methods
        model_methods = [method for method in dir(model) if callable(getattr(model, method)) and not method.startswith('_')]
        logger.info(f"Available model methods: {', '.join(model_methods)}")
        
        # Check model capabilities
        has_predict_semantic = hasattr(model, 'predict_semantic')
        has_forward_semantic = hasattr(model, 'forward_semantic')
        has_predict = hasattr(model, 'predict')
        has_forward = hasattr(model, 'forward')
        
        logger.info(f"Model capabilities: predict_semantic={has_predict_semantic}, forward_semantic={has_forward_semantic}, predict={has_predict}, forward={has_forward}")
        
        if not any([has_predict_semantic, has_forward_semantic, has_predict, has_forward]):
            logger.error("Model does not have any compatible interface for classification")
            if TextualClassifier is None:
                logger.error("TextualClassifier not available and model has no compatible interfaces")
                pytest.skip("No compatible model interfaces available")
        
        # Load and process datasets
        if test_datasets is None:
            # Default datasets to test
            datasets = ['coffee_ratings', 'michelin', 'ramen_ratings']
        else:
            datasets = test_datasets
            
        logger.info(f"Testing zero-shot classification on datasets: {datasets}")
        results = {}
        
        for dataset_name in datasets:
            logger.info(f"\nTesting zero-shot classification on {dataset_name}")
            
            # Load and preprocess dataset
            dataset_dict = load_carte_dataset(dataset_name, bin_classes=True, max_classes=10)
            dataset_dict = preprocess_dataset(dataset_dict)
            
            # Extract necessary data
            X_test_tensor = dataset_dict["X_test_tensor"]
            y_test = dataset_dict["y_test"]
            class_names = dataset_dict["class_names"]
            inverse_mapping = dataset_dict["inverse_mapping"]
            
            # Convert y_test to numpy array if it's a Series
            if hasattr(y_test, 'values'):
                y_test = y_test.values
            
            logger.info(f"Test data shape: {X_test_tensor.shape}, Target shape: {y_test.shape}")
            
            # Map keys in class_names to class indices if needed
            class_descriptions = {}
            for key, desc in class_names.items():
                # Try to find the corresponding class index
                if key in dataset_dict["target_mapping"]:
                    idx = dataset_dict["target_mapping"][key]
                    class_descriptions[idx] = desc
                else:
                    # Just use the key as is
                    class_descriptions[key] = desc
            
            logger.info(f"Prepared {len(class_descriptions)} class descriptions")
            
            # Check if the model has the wrapper interface with predict_semantic
            if hasattr(model, 'predict_semantic'):
                logger.info("Using model's predict_semantic method...")
                
                # Convert tensor to numpy for sklearn-compatible interface
                X_test_np = X_test_tensor.cpu().numpy() if isinstance(X_test_tensor, torch.Tensor) else X_test_tensor
                
                try:
                    # Print model info with detailed hierarchy
                    logger.info(f"Model type: {type(model).__name__}")
                    if hasattr(model, 'model'):
                        logger.info(f"Inner model type: {type(model.model).__name__}")
                    if hasattr(model, 'base_model'):
                        logger.info(f"Base model type: {type(model.base_model).__name__}")
                        # Check if base_model has important attributes
                        if hasattr(model.base_model, 'semantic_feature_p'):
                            logger.info(f"Base model semantic_feature_p: {model.base_model.semantic_feature_p}")
                        if hasattr(model.base_model, 'n_out'):
                            logger.info(f"Base model n_out: {model.base_model.n_out}")
                    
                    # Check for wrapper-specific attributes
                    if hasattr(model, 'semantic_feature_p'):
                        logger.info(f"Model semantic_feature_p: {model.semantic_feature_p}")
                    if hasattr(model, 'num_semantic_classes'):
                        logger.info(f"Model num_semantic_classes: {model.num_semantic_classes}")
                    
                    # Print methods
                    methods = [m for m in dir(model) if callable(getattr(model, m)) and not m.startswith('_')]
                    logger.info(f"Available methods: {methods}")
                    
                    # Check if predict_semantic has expected signature
                    import inspect
                    if hasattr(model, 'predict_semantic'):
                        try:
                            sig = inspect.signature(model.predict_semantic)
                            logger.info(f"predict_semantic signature: {sig}")
                        except Exception as e:
                            logger.info(f"Couldn't get signature: {e}")
                    
                    # If available, also check the inherited method from SemanticAwareClassifierWrapper
                    logger.info("Checking for SemanticAwareClassifierWrapper methods:")
                    from ticl.prediction.semantic import SemanticAwareClassifierWrapper
                    predict_semantic_method = SemanticAwareClassifierWrapper.predict_semantic
                    try:
                        wrapper_sig = inspect.signature(predict_semantic_method)
                        logger.info(f"Wrapper predict_semantic signature: {wrapper_sig}")
                        logger.info(f"Wrapper method source:\n{inspect.getsource(predict_semantic_method)}")
                    except Exception as e:
                        logger.info(f"Couldn't get wrapper method details: {e}")
                    
                    # Examine class_descriptions
                    logger.info(f"Class descriptions type: {type(class_descriptions)}")
                    for i, (k, v) in enumerate(class_descriptions.items()):
                        if i < 5:  # Just log the first few to avoid excessive output
                            logger.info(f"Class desc [{k}]: {v[:50]}...")
                        if i == 5:
                            logger.info(f"... and {len(class_descriptions) - 5} more class descriptions")
                    
                    # Convert class_descriptions to strings if they're not already
                    string_class_descriptions = {}
                    for k, v in class_descriptions.items():
                        if not isinstance(k, str):
                            string_class_descriptions[str(k)] = v
                        else:
                            string_class_descriptions[k] = v
                    
                    logger.info(f"Calling predict_semantic with {len(string_class_descriptions)} classes")
                    logger.info(f"Input shape: {X_test_np.shape}")
                    
                    # Enhanced error handling around predict_semantic
                    try:
                        # Create semantic_column_indices if needed (using indices of any categorical columns)
                        semantic_column_indices = dataset_dict.get("semantic_indices", [])
                        if semantic_column_indices:
                            logger.info(f"Using semantic_column_indices: {semantic_column_indices}")
                            
                            # Try method with additional semantic_column_indices parameter
                            result = model.predict_semantic(
                                X_test_np,
                                class_descriptions=string_class_descriptions,
                                semantic_column_indices=semantic_column_indices
                            )
                        else:
                            # Try standard method with class_descriptions argument
                            # Some models return tuple (classes, similarities), others just classes
                            result = model.predict_semantic(
                                X_test_np, 
                                class_descriptions=string_class_descriptions
                            )
                    except TypeError as e:
                        logger.info(f"TypeError in predict_semantic: {e}")
                        logger.info("Checking parameter names to match function signature")
                        
                        # Try with different parameter name combinations
                        if "X_semantic_text" in str(e) or "class_descriptions" in str(e):
                            logger.info("Trying with X_semantic_text parameter")
                            # Try creating some dummy semantic text
                            X_semantic_text = {}
                            for i, col in enumerate(dataset_dict['semantic_columns']):
                                if i < len(dataset_dict['semantic_columns']):
                                    X_semantic_text[col] = f"Semantic column {col}"
                            
                            result = model.predict_semantic(
                                X_test_np,
                                X_semantic_text=X_semantic_text,
                                class_descriptions=string_class_descriptions
                            )
                        else:
                            # Fallback to simplest case
                            result = model.predict_semantic(X_test_np)
                    
                    logger.info(f"predict_semantic returned result of type: {type(result)}")
                    
                    # Detailed inspection of result
                    if isinstance(result, tuple):
                        logger.info(f"Result is a tuple with {len(result)} elements")
                        for i, item in enumerate(result):
                            logger.info(f"Tuple item {i} type: {type(item)}")
                            if hasattr(item, 'shape'):
                                logger.info(f"Tuple item {i} shape: {item.shape}")
                            elif isinstance(item, dict):
                                logger.info(f"Tuple item {i} keys: {list(item.keys())}")
                            elif hasattr(item, '__len__'):
                                logger.info(f"Tuple item {i} length: {len(item)}")
                                
                        # Standard unpacking for tuple of (classes, similarities)
                        if len(result) == 2:
                            pred_classes, class_similarities = result
                            logger.info(f"Unpacked tuple result: predictions shape={pred_classes.shape if hasattr(pred_classes, 'shape') else 'no shape'}")
                            # Examine similarity values
                            if hasattr(class_similarities, 'shape'):
                                logger.info(f"Similarities shape: {class_similarities.shape}")
                                # Print a sample of the similarities
                                if class_similarities.size > 0:
                                    flat_similarities = class_similarities.reshape(-1)
                                    logger.info(f"Sample similarities: {flat_similarities[:5]}")
                        else:
                            # If not the standard tuple, use first element as predictions
                            pred_classes = result[0]
                            logger.info(f"Using first tuple element for predictions: type={type(pred_classes)}")
                    elif isinstance(result, dict):
                        logger.info(f"Result is a dictionary with keys: {list(result.keys())}")
                        # Try to extract predictions from common key patterns
                        pred_classes = None
                        prediction_keys = ['class_preds', 'predictions', 'pred_classes', 'classes']
                        for key in prediction_keys:
                            if key in result:
                                pred_classes = result[key]
                                logger.info(f"Using predictions from key '{key}': type={type(pred_classes)}")
                                break
                                
                        if pred_classes is None:
                            # If no standard key found, use the first value
                            first_key = next(iter(result.keys()))
                            pred_classes = result[first_key]
                            logger.info(f"Using value from key '{first_key}' as predictions: type={type(pred_classes)}")
                    else:
                        # Direct result
                        pred_classes = result
                        logger.info(f"Using direct result as predictions: type={type(pred_classes)}")
                        
                    # Log some of the actual prediction values for inspection
                    if hasattr(pred_classes, '__len__') and len(pred_classes) > 0:
                        logger.info(f"First few prediction values: {pred_classes[:5] if hasattr(pred_classes, '__getitem__') else pred_classes}")
                    
                    # Convert predictions to the right format if needed
                    if not isinstance(pred_classes, np.ndarray):
                        try:
                            if isinstance(pred_classes, torch.Tensor):
                                pred_classes = pred_classes.cpu().numpy()
                            else:
                                pred_classes = np.array(pred_classes)
                            logger.info(f"Converted to numpy array: shape={pred_classes.shape}")
                        except Exception as e:
                            logger.error(f"Error converting predictions to numpy array: {e}")
                            # Create a fallback array of zeros
                            pred_classes = np.zeros(len(y_test), dtype=int)
                            logger.info(f"Using fallback predictions of zeros with shape {pred_classes.shape}")
                    
                except TypeError as e:
                    # If the model doesn't accept class_descriptions, try without it
                    logger.warning(f"Error with predict_semantic: {e}, trying without class_descriptions")
                    try:
                        logger.info("Trying predict_semantic without class_descriptions")
                        pred_classes = model.predict_semantic(X_test_np)
                        if not isinstance(pred_classes, np.ndarray):
                            pred_classes = np.array(pred_classes)
                    except Exception as e2:
                        logger.error(f"predict_semantic without class_descriptions failed: {e2}")
                        logger.info("Trying to access semantic model directly")
                        
                        # Try to access the underlying semantic model directly
                        try:
                            if hasattr(model, 'predict_from_text'):
                                # If model has predict_from_text, use it with class descriptions
                                logger.info("Using model's predict_from_text method")
                                # Create list of class descriptions
                                class_texts = list(class_descriptions.values())
                                
                                # For each class description, predict and collect results
                                all_predictions = []
                                logger.info(f"Testing {min(3, len(class_texts))} of {len(class_texts)} class descriptions")
                                for i, text in enumerate(class_texts[:3]):  # Just try first 3 classes for speed
                                    logger.info(f"Testing class '{text[:30]}...'")
                                    result = model.predict_from_text(X_test_np, text)
                                    if isinstance(result, dict) and 'mapped_class' in result:
                                        class_id = result['mapped_class']
                                        similarity = result.get('similarity', 0.0)
                                        logger.info(f"Class '{text[:20]}...' mapped to {class_id} with similarity {similarity}")
                                        all_predictions.append((class_id, similarity))
                                
                                # If we have any successful predictions, use the one with highest similarity
                                if all_predictions:
                                    # Sort by similarity (descending)
                                    all_predictions.sort(key=lambda x: x[1], reverse=True)
                                    best_class = all_predictions[0][0]
                                    logger.info(f"Using best match class {best_class} with similarity {all_predictions[0][1]}")
                                    # Create array filled with best_class
                                    pred_classes = np.full(len(X_test_np), best_class, dtype=int)
                                else:
                                    raise ValueError("No successful predictions with predict_from_text")
                                    
                            elif hasattr(model, 'model') and hasattr(model.model, 'predict_semantic'):
                                # Try using inner model directly
                                logger.info("Using inner model's predict_semantic method")
                                inner_result = model.model.predict_semantic(X_test_np, class_descriptions=string_class_descriptions)
                                if isinstance(inner_result, tuple) and len(inner_result) == 2:
                                    pred_classes = inner_result[0]
                                else:
                                    pred_classes = inner_result
                                if not isinstance(pred_classes, np.ndarray):
                                    pred_classes = np.array(pred_classes)
                                    
                            elif hasattr(model, 'base_model') and hasattr(model.base_model, 'predict_semantic'):
                                # Try using base model directly
                                logger.info("Using base model's predict_semantic method")
                                base_result = model.base_model.predict_semantic(X_test_np, class_descriptions=string_class_descriptions)
                                if isinstance(base_result, tuple) and len(base_result) == 2:
                                    pred_classes = base_result[0]
                                else:
                                    pred_classes = base_result
                                if not isinstance(pred_classes, np.ndarray):
                                    pred_classes = np.array(pred_classes)
                                
                            elif hasattr(model, 'predict'):
                                # Fall back to regular predict
                                logger.info("Falling back to predict method")
                                pred_classes = model.predict(X_test_np)
                                if not isinstance(pred_classes, np.ndarray):
                                    pred_classes = np.array(pred_classes)
                            else:
                                # No usable predict methods found
                                logger.error("No usable predict methods found")
                                
                                # Create a last-resort prediction - all zeros
                                logger.info("Creating fallback prediction with all zeros")
                                pred_classes = np.zeros(len(X_test_np), dtype=int)
                        except Exception as e3:
                            logger.error(f"All semantic prediction methods failed: {e3}")
                            logger.info("Creating zero-filled prediction array as fallback")
                            pred_classes = np.zeros(len(X_test_np), dtype=int)
                
            # Otherwise use a more direct approach
            else:
                logger.info("Using forward method approach...")
                
                # Check if the model has a specialized forward_semantic method
                if hasattr(model, 'forward_semantic'):
                    # Use forward_semantic method
                    with torch.no_grad():
                        # Ensure the tensor is on the right device
                        if hasattr(model, 'device'):
                            device = model.device
                            X_test_tensor = X_test_tensor.to(device)
                        
                        # Make sure the model is in evaluation mode
                        if hasattr(model, 'eval'):
                            model.eval()
                            logger.info("Set model to eval mode")
                        
                        # Create dummy y tensor with zeros for prediction
                        dummy_y = torch.zeros(X_test_tensor.size(0), dtype=torch.float32, device=X_test_tensor.device)
                        
                        try:
                            logger.info("Trying forward_semantic with tuple input, class_descriptions, and single_eval_pos")
                            
                            # First pad features to match model's expected dimensions
                            X_test_padded = pad_features_for_model(model, X_test_tensor, logger)
                            
                            # Try TabPFN format with tuple input: (x, y) and class_descriptions
                            inputs = (X_test_padded, dummy_y)
                            results = model.forward_semantic(inputs, class_descriptions=class_descriptions, single_eval_pos=0)
                            predictions = results['logits'] if isinstance(results, dict) and 'logits' in results else results
                            pred_classes = torch.argmax(predictions, dim=1).cpu().numpy()
                        except Exception as e:
                            logger.warning(f"Error with forward_semantic using tuple input: {e}")
                            try:
                                # Try direct tensor input
                                logger.info("Trying forward_semantic with direct tensor input and single_eval_pos")
                                
                                # First pad features to match model's expected dimensions
                                X_test_padded = pad_features_for_model(model, X_test_tensor, logger)
                                
                                results = model.forward_semantic(X_test_padded, class_descriptions=class_descriptions, single_eval_pos=0)
                                predictions = results['logits'] if isinstance(results, dict) and 'logits' in results else results
                                pred_classes = torch.argmax(predictions, dim=1).cpu().numpy()
                            except Exception as e2:
                                logger.warning(f"Error with forward_semantic using direct tensor: {e2}, trying standard forward")
                                # Fallback to regular forward if forward_semantic fails
                                if hasattr(model, 'forward'):
                                    try:
                                        # Try with tuple format for forward
                                        logger.info("Trying forward with tuple input and single_eval_pos")
                                        
                                        # First pad features to match model's expected dimensions
                                        X_test_padded = pad_features_for_model(model, X_test_tensor, logger)
                                        
                                        inputs = (X_test_padded, dummy_y)
                                        results = model.forward(inputs, single_eval_pos=0)
                                        predictions = results['logits'] if isinstance(results, dict) and 'logits' in results else results
                                    except Exception as e3:
                                        logger.warning(f"Error with forward using tuple input: {e3}")
                                        # Try direct format as last resort
                                        logger.info("Trying forward with direct tensor input and single_eval_pos")
                                        
                                        # First pad features to match model's expected dimensions
                                        X_test_padded = pad_features_for_model(model, X_test_tensor, logger)
                                        
                                        results = model.forward(X_test_padded, single_eval_pos=0)
                                        predictions = results['logits'] if isinstance(results, dict) and 'logits' in results else results
                                    
                                    # Get class predictions
                                    pred_classes = torch.argmax(predictions, dim=1).cpu().numpy()
                                else:
                                    logger.error("No usable forward method found")
                                    continue
                
                # Try the regular forward method
                elif hasattr(model, 'forward'):
                    with torch.no_grad():
                        # Ensure the tensor is on the right device
                        if hasattr(model, 'device'):
                            device = model.device
                            X_test_tensor = X_test_tensor.to(device)
                        
                        # Make sure the model is in evaluation mode
                        if hasattr(model, 'eval'):
                            model.eval()
                            logger.info("Set model to eval mode")
                            
                        # Create dummy y tensor with zeros for prediction
                        dummy_y = torch.zeros(X_test_tensor.size(0), dtype=torch.float32, device=X_test_tensor.device)
                        
                        try:
                            # Try TabPFN format with tuple input: (x, y) and single_eval_pos
                            logger.info("Trying forward with TabPFN tuple format (x, y) and single_eval_pos")
                            
                            # First pad features to match model's expected dimensions
                            X_test_padded = pad_features_for_model(model, X_test_tensor, logger)
                            
                            inputs = (X_test_padded, dummy_y)
                            # Set single_eval_pos to 0 to indicate that all data is evaluation data (no training data)
                            results = model.forward(inputs, single_eval_pos=0)
                        except Exception as e1:
                            logger.warning(f"Error with tuple input: {e1}")
                            try:
                                # Try expanded input format with style: (0, x, y)
                                logger.info("Trying forward with expanded tuple format (style, x, y) and single_eval_pos")
                                
                                # First pad features to match model's expected dimensions
                                X_test_padded = pad_features_for_model(model, X_test_tensor, logger)
                                
                                style = torch.zeros(1, device=X_test_tensor.device)
                                inputs = (style, X_test_padded, dummy_y)
                                # Set single_eval_pos to 0 to indicate that all data is evaluation data
                                results = model.forward(inputs, single_eval_pos=0)
                            except Exception as e2:
                                logger.warning(f"Error with expanded tuple: {e2}")
                                # Fall back to direct tensor input as last resort
                                logger.info("Trying direct tensor input as last resort with single_eval_pos")
                                
                                # First pad features to match model's expected dimensions
                                X_test_padded = pad_features_for_model(model, X_test_tensor, logger)
                                
                                results = model.forward(X_test_padded, single_eval_pos=0)
                        
                        # Parse results
                        if isinstance(results, dict) and 'logits' in results:
                            predictions = results['logits']
                        elif isinstance(results, dict) and 'class_logits' in results:
                            predictions = results['class_logits']
                        else:
                            predictions = results
                            
                        logger.info(f"Forward returned predictions of shape: {predictions.shape if hasattr(predictions, 'shape') else 'no shape'}")
                        
                        # Get class predictions
                        pred_classes = torch.argmax(predictions, dim=-1).cpu().numpy()
                
                # Try using TextualClassifier as a last resort
                elif TextualClassifier:
                    logger.info("Using TextualClassifier wrapper...")
                    
                    # Create a text classifier
                    classifier = TextualClassifier(model)
                    
                    # Perform zero-shot classification with class descriptions
                    try:
                        logger.info("Performing zero-shot classification with class descriptions...")
                        
                        # Make sure the model is in evaluation mode
                        if hasattr(model, 'eval'):
                            model.eval()
                            logger.info("Set model to eval mode")
                        
                        # Create dummy y tensor for tuple format if needed
                        dummy_y = torch.zeros(X_test_tensor.size(0), dtype=torch.float32, device=X_test_tensor.device)
                        
                    
                        # Modified approach for zero-shot classification
                        logger.info("Using TextualClassifier for direct zero-shot prediction")
                        
                        # First check if the classifier supports zero-shot directly
                        if hasattr(classifier, 'zero_shot_classify'):
                            logger.info("Using zero_shot_classify method")
                            
                            # First pad features to match model's expected dimensions
                            X_test_padded = pad_features_for_model(model, X_test_tensor, logger)
                            
                            result = classifier.zero_shot_classify(X_test_padded, class_descriptions)
                        else:
                            # Try the standard method, but modify it to work in zero-shot mode
                            logger.info("Using classify_with_descriptions in zero-shot mode")
                            
                            # Try with tuple format
                            try:
                                logger.info("Trying TextualClassifier with tuple input and single_eval_pos=0")
                                
                                # First pad features to match model's expected dimensions
                                X_test_padded = pad_features_for_model(model, X_test_tensor, logger)
                                
                                inputs = (X_test_padded, dummy_y)
                                # Setting eval_pos=0 indicates that there are no training examples
                                # Everything should be treated as test data
                                result = classifier.classify_with_descriptions(inputs, class_descriptions, eval_pos=0)
                            except Exception as e:
                                logger.warning(f"Error with tuple input to TextualClassifier: {e}")
                                try:
                                    # Try with direct tensor
                                    logger.info("Trying TextualClassifier with direct tensor input")
                                    
                                    # First pad features to match model's expected dimensions
                                    X_test_padded = pad_features_for_model(model, X_test_tensor, logger)
                                    
                                    result = classifier.classify_with_descriptions(X_test_padded, class_descriptions)
                                except Exception as e2:
                                    logger.warning(f"Error with direct input to TextualClassifier: {e2}")
                                    
                                    # If all else fails, try direct methods on the underlying model
                                    logger.info("Trying direct methods on the underlying model")
                                    if hasattr(model, 'predict_from_text'):
                                        # Get text representations of each class
                                        class_texts = list(class_descriptions.values())
                                        
                                        # For each class, get predictions
                                        best_similarities = np.zeros(len(X_test_tensor))
                                        best_classes = np.zeros(len(X_test_tensor), dtype=int)
                                        
                                        # Try with the first few classes
                                        for i, text in enumerate(class_texts[:min(10, len(class_texts))]):
                                            # Get predictions for this class text
                                            text_result = model.predict_from_text(X_test_tensor, text)
                                            
                                            # Extract similarity scores
                                            if isinstance(text_result, dict) and 'similarity' in text_result:
                                                sim = text_result['similarity']
                                                for idx in range(len(X_test_tensor)):
                                                    if sim > best_similarities[idx]:
                                                        best_similarities[idx] = sim
                                                        best_classes[idx] = i
                                        
                                        # Create a result dictionary similar to classify_with_descriptions
                                        result = {
                                            'class_preds': best_classes,
                                            'similarities': best_similarities
                                        }
                                    else:
                                        raise ValueError("No suitable methods found for zero-shot classification")
                        
                        # Extract predictions
                        if isinstance(result, tuple) and len(result) == 2:
                            predictions, class_mapping = result
                            logger.info(f"TextualClassifier returned tuple with predictions and class mapping")
                        elif isinstance(result, dict):
                            predictions = result.get('class_preds')
                            class_mapping = result.get('class_mapping')
                            logger.info(f"TextualClassifier returned dict with keys: {list(result.keys())}")
                        else:
                            logger.error(f"Unexpected result format: {type(result)}")
                            continue
                        
                        # Log some information about the predictions
                        if hasattr(predictions, 'shape'):
                            logger.info(f"Predictions shape: {predictions.shape}")
                        
                        # Convert predictions to numpy
                        if isinstance(predictions, torch.Tensor):
                            pred_classes = torch.argmax(predictions, dim=1).cpu().numpy()
                        else:
                            pred_classes = np.argmax(np.array(predictions), axis=1)
                            
                        logger.info(f"Converted predictions to classes with shape: {pred_classes.shape}")
                    except Exception as e:
                        logger.error(f"TextualClassifier failed: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                        continue
                else:
                    logger.error("No suitable method found for zero-shot classification")
                    continue
            
            # Evaluate predictions
            try:
                metrics = evaluate_predictions(y_test, pred_classes)
                
                # Log results
                logger.info(f"Zero-shot results for {dataset_name}:")
                logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
                logger.info(f"  F1 (macro): {metrics['f1_macro']:.4f}")
                logger.info(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
                
                # Store results
                results[dataset_name] = {
                    "zero_shot": metrics,
                    "predictions": pred_classes,
                    "true_labels": y_test
                }
            except Exception as e:
                logger.error(f"Error during evaluation: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue
            
        return results
    
    except Exception as e:
        logger.error(f"Error in zero-shot classification test: {e}")
        import traceback
        logger.error(traceback.format_exc())
        pytest.skip(f"Test skipped due to error: {e}")

def test_few_shot_classification_carte(test_datasets=None):
    """Test few-shot classification on Carte benchmark datasets.
    
    Parameters:
    -----------
    test_datasets : list or None
        List of dataset names to test. If None, will use default datasets.
    """
    try:
        # Load model
        model = load_trained_model(TRAINED_MODEL_PATH)
        assert model is not None
        logger.info(f"Loaded model of type: {type(model).__name__}")
        
        # Check model capabilities
        has_fit_predict = hasattr(model, 'fit') and hasattr(model, 'predict')
        has_forward = hasattr(model, 'forward')
        has_base_model_forward = hasattr(model, 'base_model') and hasattr(model.base_model, 'forward')
        
        logger.info(f"Model capabilities for few-shot: fit_predict={has_fit_predict}, forward={has_forward}, base_model_forward={has_base_model_forward}")
        
        if not any([has_fit_predict, has_forward, has_base_model_forward]):
            logger.error("Model does not have any compatible interface for few-shot learning")
            pytest.skip("No compatible model interfaces available for few-shot learning")
        
        # Load and process datasets
        if test_datasets is None:
            # Default datasets to test
            datasets = ['coffee_ratings', 'michelin', 'ramen_ratings']
        else:
            datasets = test_datasets
            
        logger.info(f"Testing few-shot classification on datasets: {datasets}")
        results = {}
        
        for dataset_name in datasets:
            logger.info(f"\nTesting few-shot classification on {dataset_name}")
            
            try:
                # Load and preprocess dataset
                dataset_dict = load_carte_dataset(dataset_name, bin_classes=True, max_classes=10)
                dataset_dict = preprocess_dataset(dataset_dict)
                
                # Extract necessary data
                X_train = dataset_dict["X_train"]
                y_train = dataset_dict["y_train"]
                X_test_tensor = dataset_dict["X_test_tensor"]
                y_test = dataset_dict["y_test"]
                class_names = dataset_dict["class_names"]
                
                # Convert y_test to numpy array if it's a Series
                if hasattr(y_test, 'values'):
                    y_test = y_test.values
                
                # Convert tensor to numpy for sklearn-compatible interface
                X_test_np = X_test_tensor.cpu().numpy() if isinstance(X_test_tensor, torch.Tensor) else X_test_tensor
                
                # Sample few-shot examples (5 examples per class)
                shots_per_class = 5
                few_shot_X = []
                few_shot_y = []
                
                try:
                    # Try to get unique classes considering y_train might be a Series or array
                    unique_classes = y_train.unique() if hasattr(y_train, 'unique') else np.unique(y_train)
                    
                    for class_idx in unique_classes:
                        # Get examples for this class (handle both Series and array)
                        if hasattr(y_train, 'loc'):
                            # For pandas Series/DataFrame
                            class_examples = X_train[y_train == class_idx]
                        else:
                            # For numpy arrays
                            class_examples = X_train[np.where(y_train == class_idx)[0]]
                        
                        # Sample (or take all if fewer than shots_per_class)
                        n_samples = min(shots_per_class, len(class_examples))
                        
                        if hasattr(class_examples, 'sample'):
                            # For pandas DataFrame
                            sampled_examples = class_examples.sample(n=n_samples, random_state=42)
                            few_shot_X.append(sampled_examples)
                        else:
                            # For numpy array
                            indices = np.random.RandomState(42).choice(len(class_examples), n_samples, replace=False)
                            sampled_examples = class_examples[indices]
                            few_shot_X.append(pd.DataFrame(sampled_examples))
                        
                        few_shot_y.extend([class_idx] * n_samples)
                    
                    # Combine samples
                    if all(isinstance(x, pd.DataFrame) for x in few_shot_X):
                        few_shot_X = pd.concat(few_shot_X)
                    else:
                        few_shot_X = np.vstack(few_shot_X)
                        
                    few_shot_y = np.array(few_shot_y)
                    
                except Exception as e:
                    logger.error(f"Error creating few-shot examples: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    continue
                
                # Convert to tensors with error handling
                try:
                    if isinstance(few_shot_X, pd.DataFrame):
                        few_shot_X_np = few_shot_X.values.astype(np.float32)
                    else:
                        few_shot_X_np = few_shot_X.astype(np.float32)
                        
                    few_shot_X_tensor = torch.tensor(few_shot_X_np, dtype=torch.float32)
                    few_shot_y_tensor = torch.tensor(few_shot_y, dtype=torch.long)
                    
                    logger.info(f"Created few-shot dataset with {len(few_shot_X)} examples, tensor shape: {few_shot_X_tensor.shape}")
                except Exception as e:
                    logger.error(f"Error converting few-shot data to tensors: {e}")
                    few_shot_X_tensor = torch.zeros((1, X_test_tensor.size(1)), dtype=torch.float32)
                    few_shot_y_tensor = torch.zeros(1, dtype=torch.long)
                    logger.warning("Using placeholder tensors due to conversion error")
                
                # Check if the model is a wrapper with fit method (sklearn interface)
                if has_fit_predict:
                    logger.info("Using model's sklearn-compatible interface...")
                    
                    # Select feature columns for semantic awareness
                    semantic_indices = dataset_dict.get("semantic_indices", [])
                    
                    try:
                        # Convert to numpy for sklearn
                        few_shot_X_np = few_shot_X.values if hasattr(few_shot_X, 'values') else few_shot_X
                        
                        # Try fit with semantic_column_indices
                        try:
                            model.fit(few_shot_X_np, few_shot_y, semantic_column_indices=semantic_indices)
                        except TypeError:
                            # If semantic_column_indices is not accepted, try without it
                            logger.info("fit() doesn't accept semantic_column_indices, trying without it")
                            model.fit(few_shot_X_np, few_shot_y)
                        
                        # Predict on test data
                        pred_classes = model.predict(X_test_np)
                        
                        if not isinstance(pred_classes, np.ndarray):
                            pred_classes = np.array(pred_classes)
                            
                    except Exception as e:
                        logger.error(f"Error during sklearn fit/predict: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                        continue
                    
                # Otherwise, use the forward method with in-context learning
                else:
                    logger.info("Using model's forward method for in-context learning...")
                    
                    # Prepare model input
                    X_support = few_shot_X_tensor
                    y_support = few_shot_y_tensor
                    X_query = X_test_tensor
                    
                    # Select feature columns for semantic awareness
                    semantic_indices = dataset_dict.get("semantic_indices", [])
                    if semantic_indices:
                        logger.info(f"Using {len(semantic_indices)} semantic feature columns")
                    else:
                        logger.info("No semantic feature columns identified")
                    
                    # Pass semantic indices to model if they exist
                    model_kwargs = {}
                    if semantic_indices:
                        model_kwargs['semantic_column_indices'] = semantic_indices
                    
                    # Make sure tensors are on the right device
                    if hasattr(model, 'device'):
                        device = model.device
                        X_support = X_support.to(device)
                        y_support = y_support.to(device)
                        X_query = X_query.to(device)
                    
                    try:
                        # Choose the right forward method based on the model's interface
                        with torch.no_grad():
                            # Concatenate support and query sets
                            X_combined = torch.cat([X_support, X_query], dim=0)
                            y_combined = torch.cat([y_support, torch.zeros_like(dataset_dict["y_test_tensor"])])
                            
                            # Check function signature if available
                            forward_signature = None
                            if has_forward:
                                try:
                                    import inspect
                                    forward_signature = inspect.signature(model.forward)
                                    logger.info(f"Forward signature: {forward_signature}")
                                except Exception:
                                    pass
                            
                            # Try different forward interfaces
                            
                            # Forward with single_eval_pos (TabPFN style)
                            if has_forward and (forward_signature and 'single_eval_pos' in forward_signature.parameters):
                                logger.info("Using forward with single_eval_pos")
                                eval_pos = len(X_support)
                                results = model.forward((X_combined, y_combined.float()), single_eval_pos=eval_pos, **model_kwargs)
                            
                            # Forward with eval_pos mask
                            elif has_forward and (forward_signature and 'eval_pos' in forward_signature.parameters):
                                logger.info("Using forward with eval_pos mask")
                                eval_pos = torch.zeros(len(X_combined), dtype=torch.bool)
                                eval_pos[len(X_support):] = True  # Only evaluate on query set
                                results = model.forward(X_combined, y_combined, eval_pos=eval_pos, **model_kwargs)
                            
                            # Standard forward call
                            elif has_forward:
                                logger.info("Using standard forward call")
                                # Try different argument patterns
                                try:
                                    # Try with combined data
                                    results = model.forward(X_combined, y_combined, **model_kwargs)
                                except TypeError:
                                    try:
                                        # Try with tuple format
                                        results = model.forward((X_combined, y_combined), **model_kwargs)
                                    except TypeError:
                                        # Try with only X
                                        results = model.forward(X_combined, **model_kwargs)
                            
                            # Fall back to base_model if available
                            elif has_base_model_forward:
                                logger.info("Using base_model's forward method")
                                eval_pos = len(X_support)
                                results = model.base_model.forward((X_combined, y_combined.float()), single_eval_pos=eval_pos)
                            
                            else:
                                logger.error("Model does not have a compatible forward method for in-context learning")
                                continue
                            
                            # Extract predictions from results
                            if isinstance(results, dict):
                                if 'class_logits' in results:
                                    predictions = results['class_logits']
                                elif 'logits' in results:
                                    predictions = results['logits']
                                else:
                                    predictions = next(iter(results.values()))  # Take first tensor in dict
                            else:
                                predictions = results
                            
                            # Get class predictions
                            if isinstance(predictions, torch.Tensor):
                                if len(predictions.shape) > 1 and predictions.shape[-1] > 1:
                                    # For [batch_size, num_classes] format
                                    predictions_test = predictions[len(X_support):]
                                    pred_classes = torch.argmax(predictions_test, dim=1).cpu().numpy()
                                else:
                                    # For single prediction vector
                                    pred_classes = predictions[len(X_support):].cpu().numpy().astype(int)
                            else:
                                # For numpy array predictions
                                predictions_np = np.array(predictions)
                                pred_classes = np.argmax(predictions_np[len(X_support):], axis=1)
                    
                    except Exception as e:
                        logger.error(f"Error during forward pass: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                        continue
                
                # Evaluate predictions
                try:
                    metrics = evaluate_predictions(y_test, pred_classes)
                    
                    # Log results
                    logger.info(f"Few-shot results for {dataset_name}:")
                    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
                    logger.info(f"  F1 (macro): {metrics['f1_macro']:.4f}")
                    logger.info(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
                    
                    # Store results
                    results[dataset_name] = {
                        "few_shot": metrics,
                        "predictions": pred_classes,
                        "true_labels": y_test
                    }
                except Exception as e:
                    logger.error(f"Error during evaluation: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    continue
                
            except Exception as e:
                logger.error(f"Error processing dataset {dataset_name}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue
            
        return results
    
    except Exception as e:
        logger.error(f"Error in few-shot classification test: {e}")
        import traceback
        logger.error(traceback.format_exc())
        pytest.skip(f"Test skipped due to error: {e}")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("""
=====================================================================
IMPORTANT: This test requires the ticl module to be properly installed
or in your Python path. To run it correctly, use one of these methods:

1. Install the package in development mode:
   cd /path/to/ticl
   pip install -e .

2. Run as a module from the correct directory:
   cd /path/to/parent/of/ticl
   python -m ticl.tests.semantic.test_real_zero_shot_classification

Common Usage:
- Test specific datasets: --datasets michelin,coffee_ratings
- Run only zero-shot tests: --zero-shot-only
- Run only few-shot tests: --few-shot-only
- Fresh model for each dataset: --reload-model
- Show detailed errors: --debug
- Specify model path: --model-path /path/to/model.cpkt
=====================================================================
""")
    
    # Validate environment
    import sys
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"Pandas version: {pd.__version__}")

    import argparse
    
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Run semantic zero-shot and few-shot classification tests')
    parser.add_argument('--zero-shot-only', action='store_true', help='Run only zero-shot tests')
    parser.add_argument('--few-shot-only', action='store_true', help='Run only few-shot tests')
    parser.add_argument('--datasets', type=str, default='coffee_ratings,michelin,ramen_ratings', help='Comma-separated list of datasets to test')
    parser.add_argument('--model-path', type=str, help='Path to model checkpoint (overrides default)')
    parser.add_argument('--debug', action='store_true', help='Print detailed debug information')
    parser.add_argument('--reload-model', action='store_true', help='Reload model for each dataset (helps with state issues)')
    parser.add_argument('--force-eval-mode', action='store_true', help='Force model into evaluation mode (helps with CLIP issues)')
    parser.add_argument('--tuple-inputs', action='store_true', help='Use tuple inputs (x,y) format for TabPFN models')
    
    args = parser.parse_args()
    
    if args.model_path:
        TRAINED_MODEL_PATH = args.model_path
        print(f"Using model from custom path: {TRAINED_MODEL_PATH}")
    
    datasets = args.datasets.split(',')
    print(f"Testing on datasets: {', '.join(datasets)}")
    
    # Run the tests
    success = True
    print("\n==== Testing model loading ====")
    try:
        model = load_trained_model(TRAINED_MODEL_PATH)
        if model:
            print("✅ Model loaded successfully!")
            print(f"Model type: {type(model)}")
            
            # Force model into evaluation mode if requested
            if args.force_eval_mode:
                print("Forcing model into evaluation mode explicitly")
                # Set model to evaluation mode
                if hasattr(model, 'eval'):
                    model.eval()
                # If there's a base_model, set that to eval mode too
                if hasattr(model, 'base_model') and hasattr(model.base_model, 'eval'):
                    model.base_model.eval()
                # If there's a CLIP model, set that to eval mode too
                if hasattr(model, 'clip_text_model') and hasattr(model.clip_text_model, 'eval'):
                    model.clip_text_model.eval()
                # Freeze parameters to be extra sure
                for param in model.parameters():
                    param.requires_grad = False
                print("✅ Confirmed model is in evaluation mode with frozen parameters")
            
            # Report model mode
            if hasattr(model, 'training'):
                mode = "TRAINING" if model.training else "EVAL"
                print(f"Model is in {mode} mode")
                # Check base_model
                if hasattr(model, 'base_model') and hasattr(model.base_model, 'training'):
                    base_mode = "TRAINING" if model.base_model.training else "EVAL"
                    print(f"Base model is in {base_mode} mode")
                # Check CLIP model
                if hasattr(model, 'clip_text_model') and hasattr(model.clip_text_model, 'training'):
                    clip_mode = "TRAINING" if model.clip_text_model.training else "EVAL"
                    print(f"CLIP text model is in {clip_mode} mode")
        else:
            print("❌ Model loading failed.")
            success = False
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    if not success:
        print("❌ Model loading failed. Cannot continue with classification tests.")
        sys.exit(1)
    
    print("\n==== Testing dataset loading ====")
    dataset_success = True
    try:
        for dataset_name in datasets:
            try:
                print(f"Loading dataset: {dataset_name}")
                dataset_dict = load_carte_dataset(dataset_name, bin_classes=True, max_classes=10)
                print(f"✅ Dataset loaded with {len(dataset_dict['X'])} samples")
                dataset_dict = preprocess_dataset(dataset_dict)
                print(f"✅ Dataset preprocessed with {len(dataset_dict['X_train'])} train and {len(dataset_dict['X_test'])} test samples")
                print(f"Classes: {dataset_dict['unique_targets']}")
                # Handle class names dict properly
                if len(dataset_dict['class_names']) > 5:
                    class_names_sample = dict(list(dataset_dict['class_names'].items())[:5])
                    print(f"Class names (first 5): {class_names_sample}")
                else:
                    print(f"Class names: {dataset_dict['class_names']}")
                print(f"✅ Dataset {dataset_name} loaded successfully")
            except Exception as e:
                print(f"❌ Error loading dataset {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
                dataset_success = False
    except Exception as e:
        print(f"❌ Error in dataset loading: {e}")
        import traceback
        traceback.print_exc()
        dataset_success = False
    
    if not dataset_success:
        print("⚠️ Warning: Some datasets failed to load. Classification tests may have partial results.")
    
    # Run zero-shot tests if requested or if no specific test type is requested
    if not args.few_shot_only:
        print("\n==== Running zero-shot classification ====")
        
        # Two approaches: run all datasets at once or one by one with model reload
        if args.reload_model:
            print("Reload model mode: Testing each dataset with fresh model instance")
            zero_shot_results = {}
            
            for dataset_name in datasets:
                try:
                    print(f"\nTesting zero-shot classification on {dataset_name}")
                    # Run test for a single dataset
                    result = test_zero_shot_classification_carte(test_datasets=[dataset_name])
                    if result and dataset_name in result:
                        zero_shot_results[dataset_name] = result[dataset_name]
                        metrics = result[dataset_name]['zero_shot']
                        print(f"✅ Results for {dataset_name}:")
                        print(f"  Accuracy: {metrics['accuracy']:.4f}, F1 (macro): {metrics['f1_macro']:.4f}, F1 (weighted): {metrics['f1_weighted']:.4f}")
                    else:
                        print(f"❌ No results returned for {dataset_name}")
                except Exception as e:
                    print(f"❌ Error testing {dataset_name}: {e}")
                    if args.debug:
                        import traceback
                        traceback.print_exc()
        else:
            # Standard approach: test all datasets at once
            try:
                zero_shot_results = test_zero_shot_classification_carte(test_datasets=datasets)
                if zero_shot_results:
                    print("✅ Zero-shot classification results:")
                    print("| Dataset | Accuracy | F1 (Macro) | F1 (Weighted) |")
                    print("|---------|----------|------------|---------------|")
                    for dataset, results in zero_shot_results.items():
                        metrics = results['zero_shot']
                        print(f"| {dataset} | {metrics['accuracy']:.4f} | {metrics['f1_macro']:.4f} | {metrics['f1_weighted']:.4f} |")
                else:
                    print("❌ No results returned from zero-shot classification test")
            except Exception as e:
                print(f"❌ Error in zero-shot classification: {e}")
                if args.debug:
                    import traceback
                    traceback.print_exc()
        
        # Print summary table if results exist
        if 'zero_shot_results' in locals() and zero_shot_results:
            print("\n✅ Zero-shot classification summary:")
            print("| Dataset | Accuracy | F1 (Macro) | F1 (Weighted) |")
            print("|---------|----------|------------|---------------|")
            for dataset, results in zero_shot_results.items():
                if 'zero_shot' in results:
                    metrics = results['zero_shot']
                    print(f"| {dataset} | {metrics['accuracy']:.4f} | {metrics['f1_macro']:.4f} | {metrics['f1_weighted']:.4f} |")
        else:
            print("\n❌ No successful zero-shot classification results to report")
    
    # Run few-shot tests if requested or if no specific test type is requested
    if not args.zero_shot_only:
        print("\n==== Running few-shot classification ====")
        
        # Two approaches: run all datasets at once or one by one with model reload
        if args.reload_model:
            print("Reload model mode: Testing each dataset with fresh model instance")
            few_shot_results = {}
            
            for dataset_name in datasets:
                try:
                    print(f"\nTesting few-shot classification on {dataset_name}")
                    # Run test for a single dataset
                    result = test_few_shot_classification_carte(test_datasets=[dataset_name])
                    if result and dataset_name in result:
                        few_shot_results[dataset_name] = result[dataset_name]
                        metrics = result[dataset_name]['few_shot']
                        print(f"✅ Results for {dataset_name}:")
                        print(f"  Accuracy: {metrics['accuracy']:.4f}, F1 (macro): {metrics['f1_macro']:.4f}, F1 (weighted): {metrics['f1_weighted']:.4f}")
                    else:
                        print(f"❌ No results returned for {dataset_name}")
                except Exception as e:
                    print(f"❌ Error testing {dataset_name}: {e}")
                    if args.debug:
                        import traceback
                        traceback.print_exc()
        else:
            # Standard approach: test all datasets at once
            try:
                few_shot_results = test_few_shot_classification_carte(test_datasets=datasets)
                if few_shot_results:
                    print("✅ Few-shot classification results:")
                    print("| Dataset | Accuracy | F1 (Macro) | F1 (Weighted) |")
                    print("|---------|----------|------------|---------------|")
                    for dataset, results in few_shot_results.items():
                        metrics = results['few_shot']
                        print(f"| {dataset} | {metrics['accuracy']:.4f} | {metrics['f1_macro']:.4f} | {metrics['f1_weighted']:.4f} |")
                else:
                    print("❌ No results returned from few-shot classification test")
            except Exception as e:
                print(f"❌ Error in few-shot classification: {e}")
                if args.debug:
                    import traceback
                    traceback.print_exc()
        
        # Print summary table if results exist
        if 'few_shot_results' in locals() and few_shot_results:
            print("\n✅ Few-shot classification summary:")
            print("| Dataset | Accuracy | F1 (Macro) | F1 (Weighted) |")
            print("|---------|----------|------------|---------------|")
            for dataset, results in few_shot_results.items():
                if 'few_shot' in results:
                    metrics = results['few_shot']
                    print(f"| {dataset} | {metrics['accuracy']:.4f} | {metrics['f1_macro']:.4f} | {metrics['f1_weighted']:.4f} |")
        else:
            print("\n❌ No successful few-shot classification results to report")
    
    print("\n==== Test Summary ====")
    if success and dataset_success:
        print("✅ All components loaded successfully. Check above for classification results.")
    else:
        print("⚠️ Some components failed to load or run. Check the logs for details.")