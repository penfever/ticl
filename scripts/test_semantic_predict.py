#!/usr/bin/env python
"""
Script to directly test the predict_semantic method of the model.

This script loads a trained semantic model and runs the predict_semantic method
on a simple test input to verify that it works correctly.
"""

import os
import sys
import argparse
import logging
import importlib
import numpy as np
import torch
import random

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_semantic_predict")

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

def import_module_from_path(module_name, file_path):
    """Import a module from a file path."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def create_test_data(n_samples=10, n_features=100):
    """Create some random test data."""
    # Create random data
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    return X

def create_class_descriptions(num_classes=3):
    """Create sample class descriptions."""
    class_descriptions = {}
    
    # Some example class types
    class_types = [
        "positive", "negative", "neutral",
        "high", "medium", "low",
        "good", "bad", "average",
        "excellent", "poor", "fair"
    ]
    
    # Create descriptions for each class
    for i in range(num_classes):
        if i < len(class_types):
            class_type = class_types[i]
        else:
            class_type = f"class_{i}"
            
        # Create a description
        class_descriptions[i] = f"This is {class_type} class. Items in this class represent {class_type} outcomes or examples."
    
    return class_descriptions

def main():
    parser = argparse.ArgumentParser(description='Test the predict_semantic method directly')
    parser.add_argument('--model-path', type=str, help='Path to model checkpoint')
    parser.add_argument('--num-features', type=int, default=100, help='Number of features for test data')
    parser.add_argument('--num-samples', type=int, default=5, help='Number of samples for test data')
    parser.add_argument('--num-classes', type=int, default=3, help='Number of classes to test')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--list-models', action='store_true', help='List available model files')
    parser.add_argument('--test-wrapper', action='store_true', 
                        help='Test the SemanticAwareClassifierWrapper directly')
    parser.add_argument('--test-model', action='store_true',
                        help='Test the model from other entry points (predict_from_text, etc.)')
    
    args = parser.parse_args()
    
    # Set debug level if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
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
    
    # Find test module to load models
    try:
        test_module_path = find_test_module()
        test_module = import_module_from_path("test_real_zero_shot_classification", test_module_path)
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        return 1
    
    # Set model path if provided
    if args.model_path:
        test_module.TRAINED_MODEL_PATH = args.model_path
        logger.info(f"Using model from path: {args.model_path}")
    
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
    
    # Load the model
    logger.info(f"Loading model from {test_module.TRAINED_MODEL_PATH}")
    try:
        model = test_module.load_trained_model(test_module.TRAINED_MODEL_PATH)
        logger.info(f"Successfully loaded model: {type(model).__name__}")
        
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
            
        # Freeze parameters to be extra sure
        for param in model.parameters():
            param.requires_grad = False
            
        logger.info("Frozen all model parameters")
        
        # Print model capabilities
        if hasattr(model, 'predict_semantic'):
            logger.info("Model has predict_semantic method")
        else:
            logger.warning("Model does not have predict_semantic method")
            
        if hasattr(model, 'predict_from_text'):
            logger.info("Model has predict_from_text method")
        else:
            logger.warning("Model does not have predict_from_text method")
            
        if hasattr(model, 'forward_semantic'):
            logger.info("Model has forward_semantic method")
        else:
            logger.warning("Model does not have forward_semantic method")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Create test data
    X = create_test_data(n_samples=args.num_samples, n_features=args.num_features)
    logger.info(f"Created test data with shape {X.shape}")
    
    # Create class descriptions
    class_descriptions = create_class_descriptions(num_classes=args.num_classes)
    logger.info(f"Created {len(class_descriptions)} class descriptions")
    
    # Test the wrapper explicitly if requested
    if args.test_wrapper:
        logger.info("Testing the SemanticAwareClassifierWrapper directly")
        try:
            # Try to import the wrapper
            from ticl.prediction.semantic import SemanticAwareClassifierWrapper
            
            # Get a configuration dict
            config = {}
            if hasattr(model, 'base_model'):
                # Extract configuration from base model if available
                if hasattr(model.base_model, 'n_features'):
                    config['n_features'] = model.base_model.n_features
                if hasattr(model.base_model, 'n_out'):
                    config['n_out'] = model.base_model.n_out
                if hasattr(model.base_model, 'semantic_feature_p'):
                    config['semantic_feature_p'] = model.base_model.semantic_feature_p
            
            # Create a wrapper around the model
            device = 'cpu'
            if hasattr(model, 'device'):
                device = model.device
                
            wrapper = SemanticAwareClassifierWrapper(
                device=device,
                model=model,
                config=config,
                verbose=True
            )
            
            # Create some training data for fit
            y = np.random.randint(0, args.num_classes, size=args.num_samples)
            
            # Convert X to tensor for padding
            X_tensor = torch.from_numpy(X).float()
            
            # Pad features to match model's expected dimensions
            logger.info("Padding features for wrapper")
            X_tensor_padded = pad_features_for_model(model, X_tensor, logger)
            logger.info(f"Input tensor shape after padding: {X_tensor_padded.shape}")
            
            # Convert back to numpy for the wrapper
            X_padded = X_tensor_padded.numpy()
            
            # Fit the wrapper
            wrapper.fit(X_padded, y)
            
            # Test predict_semantic
            try:
                logger.info("Testing predict_semantic with class_descriptions")
                result = wrapper.predict_semantic(X_padded, class_descriptions=class_descriptions)
                
                # Check the result
                if isinstance(result, tuple) and len(result) == 2:
                    pred_classes, class_similarities = result
                    logger.info(f"predict_semantic returned classes {pred_classes} and similarities with shape {class_similarities.shape}")
                else:
                    logger.info(f"predict_semantic returned {result}")
                
                # Test prediction
                predictions = wrapper.predict(X_padded)
                logger.info(f"predict returned {predictions}")
            except Exception as e:
                logger.error(f"Error testing wrapper methods: {e}")
                import traceback
                traceback.print_exc()
        except Exception as e:
            logger.error(f"Error creating wrapper: {e}")
            import traceback
            traceback.print_exc()
    
    # Test the model's semantic methods directly
    if args.test_model:
        logger.info("Testing model semantic methods directly")
        try:
            # First test predict_from_text if available
            if hasattr(model, 'predict_from_text'):
                logger.info("Testing predict_from_text method")
                # Get a sample class description
                sample_text = next(iter(class_descriptions.values()))
                try:
                    # Convert X to tensor for padding
                    X_tensor = torch.from_numpy(X).float()
                    
                    # Pad features to match model's expected dimensions
                    logger.info("Padding features for predict_from_text")
                    X_tensor_padded = pad_features_for_model(model, X_tensor, logger)
                    logger.info(f"Input tensor shape after padding: {X_tensor_padded.shape}")
                    
                    # Create dummy y tensor for TabPFN format
                    dummy_y = torch.zeros(X_tensor_padded.size(0), dtype=torch.float32)
                    
                    # For SemanticAwareClassifier, inspect and try predict_from_text
                    if type(model).__name__ == 'SemanticAwareClassifier':
                        logger.info("Inspecting SemanticAwareClassifier for predict_from_text")
                        
                        # Inspect methods
                        methods = dir(model)
                        predict_from_text_method = getattr(model, 'predict_from_text', None)
                        logger.info(f"Is predict_from_text callable? {callable(predict_from_text_method)}")
                        
                        # Print details about the method
                        if predict_from_text_method is not None:
                            logger.info(f"predict_from_text method type: {type(predict_from_text_method)}")
                        else:
                            logger.info("predict_from_text method is None")
                        
                        # Check method implementation if available
                        if hasattr(model.__class__, 'predict_from_text'):
                            logger.info("predict_from_text is defined in the class")
                            logger.info(f"Method definition: {model.__class__.predict_from_text}")
                        
                        try:
                            # Create tuple format for TabPFN
                            inputs = (X_tensor_padded, dummy_y)
                            
                            # Try to get the code first
                            if predict_from_text_method is not None:
                                import inspect
                                logger.info(f"Method code: {inspect.getsource(model.__class__.predict_from_text)}")
                            
                            # Call predict_from_text
                            result = model.predict_from_text(inputs, sample_text)
                            logger.info(f"predict_from_text returned {result}")
                            return
                        except Exception as e:
                            logger.warning(f"Direct predict_from_text call failed: {e}")
                            logger.info("Continuing with other approaches...")
                    
                    # Try with tuple format first
                    try:
                        inputs = (X_tensor_padded, dummy_y)
                        logger.info("Trying predict_from_text with tuple input and single_eval_pos")
                        result = model.predict_from_text(inputs, sample_text, single_eval_pos=0)
                    except Exception as e:
                        logger.warning(f"Error with tuple input in predict_from_text: {e}")
                        
                        try:
                            # Try without single_eval_pos
                            logger.info("Trying predict_from_text without single_eval_pos")
                            result = model.predict_from_text(inputs, sample_text)
                        except Exception as e2:
                            logger.warning(f"Error without single_eval_pos: {e2}")
                            
                            # Convert back to numpy for the method as fallback
                            X_padded = X_tensor_padded.cpu().numpy() if X_tensor_padded.device.type != 'cpu' else X_tensor_padded.numpy()
                            
                            # Try with the padded numpy input
                            logger.info("Falling back to numpy input in predict_from_text")
                            result = model.predict_from_text(X_padded, sample_text, single_eval_pos=0)
                    
                    logger.info(f"predict_from_text returned {result}")
                except Exception as e:
                    logger.error(f"Error in predict_from_text: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Test forward_semantic if available
            if hasattr(model, 'forward_semantic') or hasattr(model, 'forward'):
                logger.info("Testing forward_semantic or forward method")
                
                # Convert numpy to torch tensor
                X_tensor = torch.from_numpy(X).float()
                
                # Pad features to match the model's expected dimensions
                logger.info("Padding features to match model dimensions")
                X_tensor_padded = pad_features_for_model(model, X_tensor, logger)
                logger.info(f"Input tensor shape after padding: {X_tensor_padded.shape}")
                
                # Create some dummy semantic batch info
                batch_info = {
                    'semantic_targets': [],
                    'class_token_patterns': {}
                }
                
                # Add class token patterns
                for class_idx, description in class_descriptions.items():
                    batch_info['class_token_patterns'][class_idx] = {
                        'tokens': description,
                        'column_name': f"class_{class_idx}",
                        'class_name': str(class_idx),
                        'semantic_class': class_idx
                    }
                
                try:
                    # Create dummy y tensor for TabPFN format
                    dummy_y = torch.zeros(X_tensor_padded.size(0), dtype=torch.float32)
                    inputs = (X_tensor_padded, dummy_y)
                    
                    # Try different forward methods based on what's available
                    if hasattr(model, 'forward_semantic'):
                        # First try with tuple input and single_eval_pos
                        logger.info("Trying forward_semantic with tuple input and single_eval_pos")
                        try:
                            result = model.forward_semantic(
                                inputs, 
                                class_texts=list(class_descriptions.values()), 
                                batch_info=batch_info,
                                single_eval_pos=0  # For zero-shot, use 0
                            )
                        except Exception as e:
                            logger.warning(f"Error with forward_semantic tuple input: {e}")
                            # Try with direct tensor input
                            logger.info("Falling back to direct tensor input with single_eval_pos")
                            result = model.forward_semantic(
                                X_tensor_padded, 
                                class_texts=list(class_descriptions.values()), 
                                batch_info=batch_info,
                                single_eval_pos=0  # For zero-shot, use 0
                            )
                    else:
                        # Try regular forward if forward_semantic is not available
                        logger.info("Trying standard forward with tuple input and single_eval_pos")
                        try:
                            result = model.forward(inputs, single_eval_pos=0)
                        except Exception as e:
                            logger.warning(f"Error with forward tuple input: {e}")
                            try:
                                # Try without single_eval_pos
                                logger.info("Trying standard forward without single_eval_pos")
                                result = model.forward(inputs)
                            except Exception as e2:
                                logger.warning(f"Error with standard forward: {e2}")
                                logger.warning("All forward attempts failed")
                    
                    # Print result
                    if isinstance(result, dict):
                        logger.info(f"forward_semantic returned dict with keys: {list(result.keys())}")
                    else:
                        logger.info(f"forward_semantic returned {result}")
                except Exception as e:
                    logger.error(f"Error in forward_semantic: {e}")
                    import traceback
                    traceback.print_exc()
        except Exception as e:
            logger.error(f"Error testing model semantic methods: {e}")
            import traceback
            traceback.print_exc()
    
    # Test the predict_semantic method
    logger.info("Testing the predict_semantic method")
    try:
        # Convert class descriptions keys to strings if they aren't already
        string_class_descriptions = {}
        for k, v in class_descriptions.items():
            if not isinstance(k, str):
                string_class_descriptions[str(k)] = v
            else:
                string_class_descriptions[k] = v
        
        # Try method with class_descriptions argument
        if hasattr(model, 'predict_semantic'):
            logger.info(f"Calling predict_semantic with {len(string_class_descriptions)} classes")
            
            # Convert to torch tensor for creating dummy y
            X_tensor = torch.from_numpy(X).float()
            
            # Pad features to match the model's expected dimensions
            logger.info("Padding features to match model dimensions")
            X_tensor_padded = pad_features_for_model(model, X_tensor, logger)
            logger.info(f"Input tensor shape after padding: {X_tensor_padded.shape}")
            
            # Create dummy y tensor
            dummy_y = torch.zeros(X_tensor_padded.size(0), dtype=torch.float32)
            
            # Try various input formats
            try:
                logger.info("Using tuple format with single_eval_pos=0")
                inputs = (X_tensor_padded, dummy_y)
                result = model.predict_semantic(
                    inputs, 
                    class_descriptions=string_class_descriptions,
                    single_eval_pos=0  # For zero-shot testing
                )
            except Exception as e:
                logger.warning(f"Error with tuple input: {e}")
                try:
                    # Try direct numpy input with single_eval_pos
                    # First pad the numpy array
                    logger.info("Falling back to numpy input with single_eval_pos")
                    
                    # Convert the padded tensor back to numpy for this approach
                    # Make sure it's on CPU first to avoid CUDA or MPS tensor issues
                    X_padded_numpy = X_tensor_padded.cpu().numpy() if X_tensor_padded.device.type != 'cpu' else X_tensor_padded.numpy()
                    result = model.predict_semantic(
                        X_padded_numpy, 
                        class_descriptions=string_class_descriptions,
                        single_eval_pos=0  # For zero-shot testing
                    )
                except Exception as e2:
                    logger.warning(f"Error with numpy input: {e2}")
                    try:
                        # Try direct tensor input without single_eval_pos (last resort)
                        logger.info("Falling back to direct tensor input without single_eval_pos")
                        result = model.predict_semantic(X_tensor_padded, class_descriptions=string_class_descriptions)
                    except Exception as e3:
                        logger.error(f"All predict_semantic attempts failed: {e3}")
                        raise
            
            # Check result type
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
                    logger.info(f"predict_semantic returned classes {pred_classes} and similarities with shape {class_similarities.shape}")
                    
                    # Calculate metrics (e.g. is there a clear winner in each prediction?)
                    if hasattr(class_similarities, 'shape') and len(class_similarities.shape) >= 2:
                        # For each sample, calculate the difference between highest and second highest similarities
                        # This gives us a confidence measure
                        sorted_similarities = np.sort(class_similarities, axis=1)
                        if sorted_similarities.shape[1] >= 2:
                            confidence = sorted_similarities[:, -1] - sorted_similarities[:, -2]
                            logger.info(f"Average confidence: {confidence.mean():.4f}")
                            logger.info(f"Confidence range: {confidence.min():.4f} - {confidence.max():.4f}")
            else:
                logger.info(f"predict_semantic returned {result}")
        else:
            logger.warning("Model does not have predict_semantic method")
            
            # Try to find alternative methods
            if hasattr(model, 'predict'):
                logger.info("Testing predict method as fallback")
                
                # Convert X to tensor for padding
                X_tensor = torch.from_numpy(X).float()
                
                # Pad features to match model's expected dimensions
                logger.info("Padding features for predict fallback")
                X_tensor_padded = pad_features_for_model(model, X_tensor, logger)
                logger.info(f"Input tensor shape after padding: {X_tensor_padded.shape}")
                
                # Convert back to numpy for the predict method
                X_padded = X_tensor_padded.cpu().numpy() if X_tensor_padded.device.type != 'cpu' else X_tensor_padded.numpy()
                
                # Create dummy y tensor for TabPFN format
                dummy_y = torch.zeros(X_tensor_padded.size(0), dtype=torch.float32)
                
                # Create tuple input format for TabPFN
                inputs = (X_tensor_padded, dummy_y)
                
                # For SemanticAwareClassifier, inspect the model and try direct forward call
                if type(model).__name__ == 'SemanticAwareClassifier':
                    logger.info("Detected SemanticAwareClassifier - inspecting available methods")
                    
                    # Inspect methods
                    methods = dir(model)
                    forward_method = getattr(model, 'forward', None)
                    logger.info(f"Is forward callable? {callable(forward_method)}")
                    
                    # Print details about the forward method
                    if forward_method is not None:
                        logger.info(f"forward method type: {type(forward_method)}")
                    else:
                        logger.info("forward method is None")
                        
                    # Get all methods that start with forward
                    forward_methods = [m for m in methods if m.startswith('forward')]
                    logger.info(f"Methods starting with 'forward': {forward_methods}")
                    
                    # Try to use the base model directly
                    if hasattr(model, 'base_model'):
                        logger.info("Trying to use base_model directly")
                        base_model = model.base_model
                        
                        try:
                            # Call base_model forward
                            logger.info("Calling base_model forward")
                            forward_result = base_model(inputs, single_eval_pos=0)
                            
                            # Get predictions
                            _, predictions = torch.max(forward_result, 1)
                            result = predictions.detach().numpy() if predictions.device.type == 'cpu' else predictions.detach().cpu().numpy()
                            
                            logger.info(f"base_model forward call successful, result shape: {forward_result.shape}")
                            logger.info(f"predictions: {result}")
                        except Exception as e:
                            logger.warning(f"base_model forward call failed: {e}")
                            # Save the error for debugging
                            result = f"base_model forward call failed: {e}"
                    else:
                        logger.warning("Model does not have base_model attribute")
                        result = "Could not find base_model"
                    
                    # Return early since we've already handled this case
                    logger.info(f"predict returned {result}")
                    return 0
                
                # Try with single_eval_pos parameter first
                try:
                    result = model.predict(inputs, single_eval_pos=0)
                except Exception as e:
                    logger.warning(f"Error in predict with single_eval_pos: {e}")
                    try:
                        # Try directly calling forward
                        logger.info("Trying direct forward call with single_eval_pos")
                        forward_result = model.forward(inputs, single_eval_pos=0)
                        # Convert result to predictions
                        _, result = torch.max(forward_result, 1)
                        result = result.numpy() if result.device.type == 'cpu' else result.cpu().numpy()
                    except Exception as e2:
                        logger.warning(f"Error with forward call: {e2}")
                        # Last resort - try standard predict without params
                        try:
                            result = model.predict(inputs)
                        except Exception as e3:
                            logger.warning(f"Error with predict: {e3}")
                            logger.warning("All prediction attempts failed")
                            result = "Prediction failed"
                    
                logger.info(f"predict returned {result}")
    except Exception as e:
        logger.error(f"Error testing predict_semantic: {e}")
        import traceback
        traceback.print_exc()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())