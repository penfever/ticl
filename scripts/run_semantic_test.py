#!/usr/bin/env python
"""
Script to run semantic classification tests with enhanced debugging.

This script provides a command-line interface to run the semantic classification
tests with various debugging options and output formats.
"""

import os
import sys
import argparse
import logging
import importlib.util
import torch

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("run_semantic_test")

def import_module_from_path(module_name, file_path):
    """Import a module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def find_test_module():
    """Find the test_real_zero_shot_classification.py module in the directory structure."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Possible paths to the test module
    possible_paths = [
        os.path.join(current_dir, "ticl", "ticl", "tests", "semantic", "test_real_zero_shot_classification.py"),
        os.path.join(current_dir, "tests", "semantic", "test_real_zero_shot_classification.py"),
        os.path.join(current_dir, "test_real_zero_shot_classification.py"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            logger.info(f"Found test module at: {path}")
            return path
    
    # If no path found, look in the current directory structure
    for root, dirs, files in os.walk(current_dir):
        if "test_real_zero_shot_classification.py" in files:
            path = os.path.join(root, "test_real_zero_shot_classification.py")
            logger.info(f"Found test module at: {path}")
            return path
    
    raise FileNotFoundError("Could not find test_real_zero_shot_classification.py in the directory structure")

def find_model_files(model_dir=None):
    """Find available model files in the directory structure."""
    if model_dir is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Try to find models_diff directory
        possible_dirs = [
            os.path.join(current_dir, "ticl", "ticl", "models_diff"),
            os.path.join(current_dir, "ticl", "models_diff"),
            os.path.join(current_dir, "models_diff"),
        ]
        
        for dir_path in possible_dirs:
            if os.path.exists(dir_path) and os.path.isdir(dir_path):
                model_dir = dir_path
                break
        
        if model_dir is None:
            # Search for models_diff directory
            for root, dirs, files in os.walk(current_dir):
                if "models_diff" in dirs:
                    model_dir = os.path.join(root, "models_diff")
                    break
    
    if model_dir is None or not os.path.exists(model_dir):
        logger.warning(f"Could not find models directory")
        return []
    
    logger.info(f"Looking for model files in {model_dir}")
    model_files = []
    for file in os.listdir(model_dir):
        if file.endswith(".cpkt") or file.endswith(".pickle"):
            model_files.append(os.path.join(model_dir, file))
    
    return model_files

def main():
    parser = argparse.ArgumentParser(description='Run semantic classification tests with enhanced debugging')
    parser.add_argument('--model-path', type=str, help='Path to model checkpoint')
    parser.add_argument('--benchmark-dir', type=str, help='Path to benchmark directory')
    parser.add_argument('--dataset', type=str, default='coffee_ratings', 
                        help='Dataset to test (default: coffee_ratings)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--zero-shot-only', action='store_true', help='Run only zero-shot tests')
    parser.add_argument('--few-shot-only', action='store_true', help='Run only few-shot tests')
    parser.add_argument('--list-models', action='store_true', help='List available model files')
    parser.add_argument('--list-datasets', action='store_true', help='List available benchmark datasets')
    parser.add_argument('--examine-model', action='store_true', 
                        help='Examine the model structure without running tests')
    parser.add_argument('--force-eval-mode', action='store_true', 
                        help='Force model into evaluation mode (helps with CLIP issues)')
    parser.add_argument('--tuple-inputs', action='store_true', 
                        help='Use tuple inputs (x,y) format for TabPFN models')
    parser.add_argument('--reload-model', action='store_true',
                        help='Reload model for each dataset (helps with state issues)')
    
    args = parser.parse_args()
    
    # Set debug level if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Find test module
    try:
        test_module_path = find_test_module()
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        return 1
    
    # List available models if requested
    if args.list_models:
        model_files = find_model_files()
        if model_files:
            print("\nAvailable model files:")
            for i, file_path in enumerate(model_files):
                print(f"{i+1}. {os.path.basename(file_path)}")
                # Try to get file size
                try:
                    size_mb = os.path.getsize(file_path) / (1024 * 1024)
                    print(f"   Size: {size_mb:.2f} MB")
                except:
                    pass
            print("\nUse --model-path to specify a model file")
        else:
            print("No model files found")
        return 0
    
    # Import the test module
    test_module = import_module_from_path("test_real_zero_shot_classification", test_module_path)
    
    # Set model path if provided
    if args.model_path:
        test_module.TRAINED_MODEL_PATH = args.model_path
        logger.info(f"Using model from path: {args.model_path}")
    
    # Set benchmark directory if provided
    if args.benchmark_dir:
        test_module.BENCHMARK_DIR = args.benchmark_dir
        logger.info(f"Using benchmark directory: {args.benchmark_dir}")
        
    # Set up command-line arguments for the test module
    # Create a new args namespace for the test module to use
    test_args = argparse.Namespace()
    test_args.force_eval_mode = args.force_eval_mode
    test_args.tuple_inputs = args.tuple_inputs
    test_args.reload_model = args.reload_model
    test_args.debug = args.debug
    test_args.zero_shot_only = args.zero_shot_only
    test_args.few_shot_only = args.few_shot_only
    test_args.datasets = args.dataset
    test_args.model_path = args.model_path
    
    # Pass these arguments to the test module
    test_module.args = test_args
    
    if args.force_eval_mode:
        logger.info("Force evaluation mode enabled - will explicitly put models in eval mode")
    
    if args.tuple_inputs:
        logger.info("Using tuple inputs format for TabPFN models")
    
    # List available datasets if requested
    if args.list_datasets:
        # Try to find benchmark directory if not provided
        if not hasattr(test_module, "BENCHMARK_DIR") or not os.path.exists(test_module.BENCHMARK_DIR):
            logger.warning("Benchmark directory not found or not specified")
            return 1
        
        print(f"\nLooking for datasets in: {test_module.BENCHMARK_DIR}")
        
        # Check if directory exists
        if os.path.exists(test_module.BENCHMARK_DIR):
            # List contents
            contents = os.listdir(test_module.BENCHMARK_DIR)
            
            # Find datasets (directories or CSV files)
            datasets = []
            for item in contents:
                item_path = os.path.join(test_module.BENCHMARK_DIR, item)
                if os.path.isdir(item_path):
                    # Check if the directory contains a CSV file
                    if any(f.endswith(".csv") for f in os.listdir(item_path)):
                        datasets.append(item)
                elif item.endswith(".csv"):
                    datasets.append(item.replace(".csv", ""))
            
            if datasets:
                print("\nAvailable datasets:")
                for i, dataset in enumerate(datasets):
                    print(f"{i+1}. {dataset}")
                print("\nUse --dataset to specify a dataset")
            else:
                print("No datasets found")
        else:
            print(f"Benchmark directory {test_module.BENCHMARK_DIR} not found")
        
        return 0
    
    # Check if model path is set
    if not hasattr(test_module, "TRAINED_MODEL_PATH") or not os.path.exists(test_module.TRAINED_MODEL_PATH):
        logger.error("Model path not set or file not found")
        model_files = find_model_files()
        if model_files:
            logger.info("Available model files:")
            for i, file_path in enumerate(model_files):
                logger.info(f"{i+1}. {os.path.basename(file_path)}")
            logger.info("Use --model-path to specify one of these files")
        return 1
    
    # Examine model if requested
    if args.examine_model:
        logger.info(f"Examining model: {test_module.TRAINED_MODEL_PATH}")
        
        try:
            # Load the model
            model = test_module.load_trained_model(test_module.TRAINED_MODEL_PATH)
            
            # Print model info
            print(f"\nModel type: {type(model).__name__}")
            
            # Check if model is in training or eval mode
            if hasattr(model, 'training'):
                mode = "TRAINING" if model.training else "EVALUATION"
                print(f"Model is in {mode} mode")
            
            # Check for various attributes
            if hasattr(model, 'model'):
                print(f"Inner model type: {type(model.model).__name__}")
                # Check inner model mode
                if hasattr(model.model, 'training'):
                    mode = "TRAINING" if model.model.training else "EVALUATION"
                    print(f"Inner model is in {mode} mode")
            
            if hasattr(model, 'base_model'):
                print(f"Base model type: {type(model.base_model).__name__}")
                # Check base model mode
                if hasattr(model.base_model, 'training'):
                    mode = "TRAINING" if model.base_model.training else "EVALUATION"
                    print(f"Base model is in {mode} mode")
                
                # Print base model dimensions
                if hasattr(model.base_model, 'emsize'):
                    print(f"Base model emsize: {model.base_model.emsize}")
                if hasattr(model.base_model, 'n_features'):
                    print(f"Base model n_features: {model.base_model.n_features}")
                if hasattr(model.base_model, 'n_out'):
                    print(f"Base model n_out: {model.base_model.n_out}")
                if hasattr(model.base_model, 'nhid_factor'):
                    print(f"Base model nhid_factor: {model.base_model.nhid_factor}")
                if hasattr(model.base_model, 'nlayers'):
                    print(f"Base model nlayers: {model.base_model.nlayers}")
                if hasattr(model.base_model, 'semantic_feature_p'):
                    print(f"Base model semantic_feature_p: {model.base_model.semantic_feature_p}")
                    
            # Check for CLIP model
            if hasattr(model, 'clip_text_model'):
                print(f"CLIP text model type: {type(model.clip_text_model).__name__}")
                # Check CLIP model mode
                if hasattr(model.clip_text_model, 'training'):
                    mode = "TRAINING" if model.clip_text_model.training else "EVALUATION"
                    print(f"CLIP text model is in {mode} mode")
                    if model.clip_text_model.training:
                        print("⚠️ WARNING: CLIP model is in TRAINING mode - this may cause issues!")
                        
            # Check parameters requires_grad
            requires_grad_count = 0
            total_params = 0
            for name, param in model.named_parameters():
                total_params += 1
                if param.requires_grad:
                    requires_grad_count += 1
                    
            if requires_grad_count > 0:
                print(f"⚠️ WARNING: {requires_grad_count}/{total_params} parameters have requires_grad=True")
                
            # Option to force model into eval mode
            if args.force_eval_mode:
                print("\n=== Forcing model into evaluation mode ===")
                # Set model to evaluation mode
                if hasattr(model, 'eval'):
                    model.eval()
                # If there's a base_model, set that to eval mode too
                if hasattr(model, 'base_model') and hasattr(model.base_model, 'eval'):
                    model.base_model.eval()
                # If there's a CLIP model, set that to eval mode too
                if hasattr(model, 'clip_text_model') and hasattr(model.clip_text_model, 'eval'):
                    model.clip_text_model.eval()
                # Freeze parameters
                for param in model.parameters():
                    param.requires_grad = False
                print("✅ Model has been set to evaluation mode with frozen parameters")
            
            # Print semantic model dimensions
            if hasattr(model, 'emsize'):
                print(f"Model emsize: {model.emsize}")
            if hasattr(model, 'num_semantic_classes'):
                print(f"Model num_semantic_classes: {model.num_semantic_classes}")
            if hasattr(model, 'semantic_feature_p'):
                print(f"Model semantic_feature_p: {model.semantic_feature_p}")
            
            # Print available methods
            methods = [m for m in dir(model) if callable(getattr(model, m)) and not m.startswith('_')]
            print(f"\nAvailable methods: {', '.join(methods)}")
            
            # Check if the model has the right prediction methods
            prediction_methods = ['predict', 'predict_semantic', 'predict_from_text', 'forward_semantic']
            available_prediction_methods = [m for m in prediction_methods if m in methods]
            print(f"\nPrediction methods: {', '.join(available_prediction_methods)}")
            
            # Check for state_dict size
            if hasattr(model, 'state_dict'):
                state_dict = model.state_dict()
                # Calculate total parameters
                total_params = sum(p.numel() for p in model.parameters())
                print(f"\nTotal parameters: {total_params:,}")
                print(f"State dict has {len(state_dict)} keys")
                
                # Print a few state dict keys
                print("\nSample state dict keys:")
                for i, key in enumerate(list(state_dict.keys())[:5]):
                    shape = state_dict[key].shape
                    print(f"  {key}: shape={shape}")
                
        except Exception as e:
            logger.error(f"Error examining model: {e}")
            import traceback
            traceback.print_exc()
            return 1
        
        return 0
    
    # If no specific tests requested, run both
    run_zero_shot = not args.few_shot_only
    run_few_shot = not args.zero_shot_only
    
    # Run the tests
    if run_zero_shot:
        logger.info(f"Running zero-shot classification test on dataset: {args.dataset}")
        try:
            # Force the model into evaluation mode before running tests if requested
            if args.force_eval_mode and hasattr(test_module, 'model'):
                model = test_module.model
                logger.info("Forcing model into evaluation mode before zero-shot test")
                if hasattr(model, 'eval'):
                    model.eval()
                if hasattr(model, 'base_model') and hasattr(model.base_model, 'eval'):
                    model.base_model.eval()
                if hasattr(model, 'clip_text_model') and hasattr(model.clip_text_model, 'eval'):
                    model.clip_text_model.eval()
                # Freeze parameters
                for param in model.parameters():
                    param.requires_grad = False
            
            # Call the test function
            result = test_module.test_zero_shot_classification_carte(test_datasets=[args.dataset])
            
            # Print results
            if result and args.dataset in result:
                metrics = result[args.dataset]['zero_shot']
                print(f"\nZero-shot results for {args.dataset}:")
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
                print(f"  F1 (macro): {metrics['f1_macro']:.4f}")
                print(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
                
                # If a classification report is available, show it
                if 'report' in metrics:
                    print("\nDetailed classification report:")
                    from pprint import pprint
                    pprint(metrics['report'])
            else:
                print(f"No results returned for {args.dataset}")
        except Exception as e:
            logger.error(f"Error in zero-shot test: {e}")
            import traceback
            traceback.print_exc()
    
    if run_few_shot:
        logger.info(f"Running few-shot classification test on dataset: {args.dataset}")
        try:
            # Force the model into evaluation mode before running tests if requested
            if args.force_eval_mode and hasattr(test_module, 'model'):
                model = test_module.model
                logger.info("Forcing model into evaluation mode before few-shot test")
                if hasattr(model, 'eval'):
                    model.eval()
                if hasattr(model, 'base_model') and hasattr(model.base_model, 'eval'):
                    model.base_model.eval()
                if hasattr(model, 'clip_text_model') and hasattr(model.clip_text_model, 'eval'):
                    model.clip_text_model.eval()
                # Freeze parameters
                for param in model.parameters():
                    param.requires_grad = False
            
            # Call the test function
            result = test_module.test_few_shot_classification_carte(test_datasets=[args.dataset])
            
            # Print results
            if result and args.dataset in result:
                metrics = result[args.dataset]['few_shot']
                print(f"\nFew-shot results for {args.dataset}:")
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
                print(f"  F1 (macro): {metrics['f1_macro']:.4f}")
                print(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
                
                # If a classification report is available, show it
                if 'report' in metrics:
                    print("\nDetailed classification report:")
                    from pprint import pprint
                    pprint(metrics['report'])
            else:
                print(f"No results returned for {args.dataset}")
        except Exception as e:
            logger.error(f"Error in few-shot test: {e}")
            import traceback
            traceback.print_exc()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())