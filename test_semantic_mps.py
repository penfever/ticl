import torch
import os
import argparse
import numpy as np
from tqdm import tqdm

def test_stability(device_type='cuda'):
    """
    Test numerical stability of semantic similarity calculations on different devices.
    Compares CPU, MPS (if available), and CUDA (if available) for handling the same operations.
    
    Parameters:
    -----------
    device_type : str
        Default device to test ('cuda', 'mps', or 'cpu')
    """
    print(f"Testing numerical stability on {device_type}")
    
    # Create random features
    batch_size = 4
    embed_dim = 512
    num_classes = 10
    
    # Devices to test
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        devices.append('mps')
        
    print(f"Available devices: {devices}")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create a common set of data on CPU first
    tabular_features_cpu = torch.randn(batch_size, embed_dim)
    text_features_cpu = torch.randn(num_classes, embed_dim)
    
    # Normalize features (essential for proper cosine similarity)
    tabular_features_cpu = torch.nn.functional.normalize(tabular_features_cpu, dim=1)
    text_features_cpu = torch.nn.functional.normalize(text_features_cpu, dim=1)
    
    # Test each device
    results = {}
    for device in devices:
        try:
            print(f"\nTesting on {device.upper()}")
            
            # Move tensors to device
            tabular_features = tabular_features_cpu.to(device)
            text_features = text_features_cpu.to(device)
            
            # Calculate similarity and apply standard normalization
            similarity = torch.matmul(tabular_features, text_features.t())
            
            # Report stats
            print(f"Raw similarity - min: {similarity.min().item():.6f}, max: {similarity.max().item():.6f}")
            print(f"Raw similarity - mean: {similarity.mean().item():.6f}, std: {similarity.std().item():.6f}")
            
            # Apply normalization by max value + epsilon
            for eps in [1e-8, 1e-6, 1e-4, 1e-3, 1e-2]:
                print(f"\nTesting epsilon: {eps}")
                
                # Normalize by max (CLIP-style)
                norm_similarity = similarity / (similarity.abs().max() + eps)
                
                # Check for NaN values
                has_nan = torch.isnan(norm_similarity).any().item()
                has_inf = torch.isinf(norm_similarity).any().item()
                
                # Apply scaling and clipping (varying values)
                for scale in [0.5, 1.0, 2.0, 5.0]:
                    for clip_val in [5.0, 10.0, 20.0]:
                        # Scale and clip
                        logits = norm_similarity * scale
                        logits = torch.clamp(logits, min=-clip_val, max=clip_val)
                        
                        # Check and report
                        logits_has_nan = torch.isnan(logits).any().item()
                        logits_has_inf = torch.isinf(logits).any().item()
                        
                        # Get stats
                        logits_min = logits.min().item()
                        logits_max = logits.max().item()
                        logits_mean = logits.mean().item()
                        logits_std = logits.std().item()
                        
                        # Store results
                        key = f"{device}_eps{eps}_scale{scale}_clip{clip_val}"
                        results[key] = {
                            'device': device,
                            'epsilon': eps,
                            'scale': scale,
                            'clip_val': clip_val,
                            'has_nan': has_nan or logits_has_nan,
                            'has_inf': has_inf or logits_has_inf,
                            'min': logits_min,
                            'max': logits_max,
                            'mean': logits_mean,
                            'std': logits_std
                        }
                        
                        # Report
                        status = "✓" if not (has_nan or logits_has_nan or has_inf or logits_has_inf) else "✗"
                        print(f"Scale {scale}, Clip {clip_val}: {status} - Min/Max: {logits_min:.4f}/{logits_max:.4f}, Mean±Std: {logits_mean:.4f}±{logits_std:.4f}")
            
        except Exception as e:
            print(f"Error testing on {device}: {e}")
    
    # Summary
    print("\n===== STABILITY SUMMARY =====")
    device_success = {device: [] for device in devices}
    
    for key, data in results.items():
        device = data['device']
        stable = not (data['has_nan'] or data['has_inf'])
        device_success[device].append(stable)
    
    for device in devices:
        success_rate = sum(device_success[device]) / len(device_success[device]) * 100
        print(f"{device.upper()}: {success_rate:.1f}% configurations stable")
    
    # Test gradient stability
    print("\n===== GRADIENT STABILITY =====")
    for device in devices:
        try:
            print(f"\nTesting gradient stability on {device.upper()}")
            
            # Create new tensors with gradients
            tabular_features = torch.randn(batch_size, embed_dim, device=device, requires_grad=True)
            text_features = torch.randn(num_classes, embed_dim, device=device)
            
            # Normalize
            tabular_features = torch.nn.functional.normalize(tabular_features, dim=1)
            text_features = torch.nn.functional.normalize(text_features, dim=1)
            
            # Select best parameters from earlier test
            best_eps = 1e-3 if device != 'cuda' else 1e-2
            best_scale = 2.0 if device != 'cuda' else 1.0
            best_clip = 10.0 if device != 'cuda' else 5.0
            
            # Test forward pass
            similarity = torch.matmul(tabular_features, text_features.t())
            norm_similarity = similarity / (similarity.abs().max() + best_eps)
            logits = norm_similarity * best_scale
            logits = torch.clamp(logits, min=-best_clip, max=best_clip)
            
            # Create "labels" (random for testing)
            labels = torch.randint(0, num_classes, (batch_size,), device=device)
            
            # Calculate loss
            loss = torch.nn.functional.cross_entropy(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Check gradients
            grad = tabular_features.grad
            grad_has_nan = torch.isnan(grad).any().item()
            grad_has_inf = torch.isinf(grad).any().item()
            
            # Report
            status = "✓" if not (grad_has_nan or grad_has_inf) else "✗"
            print(f"Gradient stability: {status}")
            print(f"Gradient stats - min: {grad.min().item():.6f}, max: {grad.max().item():.6f}")
            print(f"Gradient stats - mean: {grad.mean().item():.6f}, std: {grad.std().item():.6f}")
            
            # Test our fix
            print("\nTesting our CUDA-specific fix:")
            
            # Different parameters for CUDA
            if device == 'cuda':
                # Apply CUDA-specific adjustments
                tabular_features = torch.randn(batch_size, embed_dim, device=device, requires_grad=True)
                tabular_features = torch.nn.functional.normalize(tabular_features, dim=1)
                
                similarity = torch.matmul(tabular_features, text_features.t())
                norm_similarity = similarity / (similarity.abs().max() + 1e-2)  # Larger epsilon
                logits = norm_similarity * 1.0  # Smaller scale
                logits = torch.clamp(logits, min=-5.0, max=5.0)  # Tighter clamp
                
                # Normalize via softmax to handle extreme cases
                normalized_logits = torch.nn.functional.softmax(logits, dim=-1)
                logits = torch.log(normalized_logits + 1e-6) * 2.0
                
                # Loss and backward
                loss = torch.nn.functional.cross_entropy(logits, labels)
                loss.backward()
                
                # Check gradients
                grad = tabular_features.grad
                grad_has_nan = torch.isnan(grad).any().item()
                grad_has_inf = torch.isinf(grad).any().item()
                
                # Report
                status = "✓" if not (grad_has_nan or grad_has_inf) else "✗"
                print(f"CUDA fix gradient stability: {status}")
                print(f"CUDA fix gradient stats - min: {grad.min().item():.6f}, max: {grad.max().item():.6f}")
                print(f"CUDA fix gradient stats - mean: {grad.mean().item():.6f}, std: {grad.std().item():.6f}")
            
        except Exception as e:
            print(f"Error testing gradients on {device}: {e}")
    
    print("\nStability test complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test semantic model stability')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'mps', 'cpu'],
                        help='Device to test (default: cuda)')
    
    args = parser.parse_args()
    test_stability(args.device)