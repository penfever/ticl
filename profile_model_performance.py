import torch
import time
import logging
import argparse
import numpy as np
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def profile_model_performance(model_type='tabflex', 
                             n_samples=1000, 
                             batch_size=16, 
                             semantic_feature_p=0.3, 
                             epochs=1, 
                             device='cuda'):
    """Profile model performance with and without semantic features"""
    
    from ticl.model_builder import build_model
    from ticl.model_configs import get_experiment_config
    
    # Get default config for the specified model
    config = get_experiment_config(model_type)
    
    # Update config with our parameters
    config['semantic_feature_p'] = semantic_feature_p
    config['train']['batch_size'] = batch_size
    config['num_epochs'] = epochs
    
    # Device configuration
    use_cuda = torch.cuda.is_available() and 'cuda' in device
    use_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and 'mps' in device
    
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    logger.info(f"Using device: {device}")
    
    # Build the model
    logger.info(f"Building {model_type} model...")
    model = build_model(config, device)
    logger.info(f"Model type: {type(model).__name__}")
    
    # Generate dummy data
    logger.info("Generating dummy data...")
    X_train = torch.randn(n_samples, config['train']['features'])
    y_train = torch.randint(0, config['train'].get('num_classes', 2), (n_samples,))
    X_valid = torch.randn(n_samples // 5, config['train']['features'])
    y_valid = torch.randint(0, config['train'].get('num_classes', 2), (n_samples // 5,))
    
    # Move data to the device
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_valid = X_valid.to(device)
    y_valid = y_valid.to(device)
    
    # Profile training process
    logger.info("Profiling training process...")
    start_time = time.time()
    
    # Run through the training process
    model.fit(X_train, y_train, 
             X_val=X_valid, y_val=y_valid,
             max_epochs=epochs, 
             log_every=5,
             return_best='val_loss',
             batch_size=batch_size)
    
    train_time = time.time() - start_time
    logger.info(f"Training time for {epochs} epochs: {train_time:.2f} seconds")
    
    # Profile inference process
    logger.info("Profiling inference process...")
    start_time = time.time()
    
    # Inference on validation set
    num_batches = 10
    for _ in tqdm(range(num_batches)):
        with torch.no_grad():
            _ = model.predict(X_valid)
    
    inference_time = (time.time() - start_time) / num_batches
    logger.info(f"Average inference time per batch: {inference_time:.4f} seconds")
    
    return {
        'model_type': model_type,
        'semantic_feature_p': semantic_feature_p,
        'device': str(device),
        'training_time': train_time,
        'inference_time': inference_time,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Profile model performance')
    parser.add_argument('--model', type=str, default='tabflex', 
                       choices=['tabflex', 'tabpfn', 'mothernet'],
                       help='Model type to profile')
    parser.add_argument('--semantic', type=float, default=0.3,
                       help='Semantic feature probability')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--n-samples', type=int, default=1000,
                       help='Number of samples to use')
    parser.add_argument('--epochs', type=int, default=1,
                       help='Number of epochs')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu', 'mps'],
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Run the profiling
    results = profile_model_performance(
        model_type=args.model,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        semantic_feature_p=args.semantic,
        epochs=args.epochs,
        device=args.device
    )
    
    # Display results
    print("\n===== Performance Results =====")
    print(f"Model: {results['model_type']}")
    print(f"Semantic feature probability: {results['semantic_feature_p']}")
    print(f"Device: {results['device']}")
    print(f"Training time: {results['training_time']:.2f} seconds")
    print(f"Inference time per batch: {results['inference_time']:.4f} seconds")