import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate
import pandas as pd

from ticl.priors.classification_adapter import ClassificationAdapter
from ticl.datasets.semantic_prior_data_loader import get_random_semantic_data

class DummyPrior:
    """Simple dummy prior for feature generation"""
    def __init__(self):
        pass
        
    def get_batch(self, batch_size, n_samples, num_features, device, **kwargs):
        x = torch.randn(n_samples, batch_size, num_features, device=device)
        y = torch.randint(0, 5, (n_samples, batch_size, 1), device=device).float()
        return x, y, y

def extract_semantic_features(
    num_features=20,
    batch_size=8,
    n_samples=10,
    num_classes=5,
    device='cpu',
    random_seed=42
):
    """Extract semantic features and their column name mappings"""
    
    # Create a base prior
    base_prior = DummyPrior()
    
    # Configuration for semantic features
    config = {
        'num_features_sampler': 'uniform',
        'feature_curriculum': False,
        'pad_zeros': False,
        'balanced': False,
        'multiclass_type': 'rank',
        'output_multiclass_ordered_p': 0.5,
        'multiclass_max_steps': 5,
        'nan_prob_a_reason': 0.0,
        'nan_prob_no_reason': 0.0,
        'categorical_feature_p': 0.0,
        'max_num_classes': 10,
        'num_classes': num_classes,
        'set_value_to_nan': 'nan',
        'semantic_feature_p': 1.0,  # Always enable semantic features
        'random_seed': random_seed
    }
    
    # Create the adapter
    adapter = ClassificationAdapter(base_prior, config)
    
    # Generate features
    single_eval_pos = n_samples // 2
    x, y, _, info = adapter(batch_size, n_samples, num_features, device, single_eval_pos=single_eval_pos)
    
    # Extract semantic feature information
    class_token_patterns = info.get('class_token_patterns', {})
    semantic_targets = info.get('semantic_targets', None)
    semantic_features = info.get('semantic_features', [])
    
    # Extract semantic feature values
    semantic_values = x[:, :, semantic_features] if semantic_features else None
    
    # Extract column names if available
    column_names = []
    semantic_classes = []
    for class_idx, pattern in class_token_patterns.items():
        column_name = pattern.get('column_name', f"Column_{pattern.get('semantic_class', 'Unknown')}")
        column_names.append(column_name)
        semantic_classes.append(pattern.get('semantic_class'))
    
    return {
        'x': x,
        'y': y,
        'semantic_targets': semantic_targets,
        'semantic_features': semantic_features,
        'semantic_values': semantic_values,
        'class_token_patterns': class_token_patterns,
        'column_names': column_names,
        'semantic_classes': semantic_classes,
        'class_to_semantic': {class_idx: pattern.get('semantic_class') for class_idx, pattern in class_token_patterns.items()}
    }

def calculate_similarity_matrix(features, num_samples=10, average_by_class=True):
    """Calculate similarity matrix between semantic features"""
    # Take a few samples for analysis
    if isinstance(features, torch.Tensor):
        if features.dim() == 3:
            # First n samples, first batch only, all semantic features
            sample_features = features[:num_samples, 0, :].detach().cpu().numpy()
        else:
            sample_features = features[:num_samples].detach().cpu().numpy()
    else:
        sample_features = features[:num_samples]
    
    # Calculate pairwise cosine similarity
    similarity = cosine_similarity(sample_features)
    
    return similarity

def visualize_semantic_features(data, output_dir=None, show_plots=True):
    """Visualize semantic features and their relationship to column names"""
    
    # Extract data
    x = data['x']
    y = data['y']
    semantic_targets = data['semantic_targets']
    semantic_features = data['semantic_features']
    semantic_values = data['semantic_values']
    class_token_patterns = data['class_token_patterns']
    column_names = data['column_names']
    class_to_semantic = data['class_to_semantic']
    
    # 1. Plaintext visualization of feature-column relationships
    print("\n=== Semantic Feature to Column Mapping ===")
    mapping_table = []
    
    # For each class
    for class_idx, pattern in class_token_patterns.items():
        semantic_class = pattern.get('semantic_class')
        column_name = pattern.get('column_name', f"Column_{semantic_class}")
        tokens = pattern.get('tokens', torch.tensor([])).detach().cpu().numpy()
        
        # Truncate tokens if too many
        if len(tokens) > 5:
            token_str = f"{tokens[:5]} ... (truncated, {len(tokens)} total)"
        else:
            token_str = tokens
            
        mapping_table.append([
            class_idx, 
            semantic_class,
            column_name if column_name else "N/A",
            token_str
        ])
    
    print(tabulate(mapping_table, 
                  headers=["Class", "Semantic Class", "Column Name", "Example Tokens"],
                  tablefmt="pretty"))
    
    # 2. Create a feature matrix visualization
    # Select a subset of samples for visualization
    num_viz_samples = min(5, semantic_values.shape[0])
    num_viz_batches = min(3, semantic_values.shape[1])
    
    # Calculate average semantic feature value per class
    class_avg_features = {}
    for class_idx in class_to_semantic.keys():
        # Get mask for this class
        class_mask = (y.squeeze(-1) == class_idx)
        
        # Skip if no samples
        if not class_mask.any():
            continue
            
        # Get features for this class
        class_features = semantic_values[class_mask]
        
        # Calculate average
        if len(class_features) > 0:
            class_avg_features[class_idx] = class_features.mean(dim=0).detach().cpu().numpy()
    
    # 3. Create a similarity matrix between classes and semantic features
    print("\n=== Class-Semantic Feature Similarity Matrix ===")
    
    # Calculate cosine similarity between feature vectors
    if len(class_avg_features) > 1:
        # Convert to matrix
        class_indices = sorted(class_avg_features.keys())
        feature_matrix = np.array([class_avg_features[idx] for idx in class_indices])
        
        # Calculate similarity
        similarity = cosine_similarity(feature_matrix)
        
        # Create a DataFrame with class names
        class_names = [f"Class {idx}" for idx in class_indices]
        sim_df = pd.DataFrame(similarity, index=class_names, columns=class_names)
        
        # Print as table
        print(tabulate(sim_df, headers=sim_df.columns, tablefmt="pretty"))
        
        # Save or show heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(sim_df, annot=True, cmap="YlGnBu")
        plt.title("Cosine Similarity Between Class Feature Vectors")
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, "class_similarity.png"))
        if show_plots:
            plt.show()
        plt.close()
    
    # 4. Create a token-class matrix (similar to CLIP)
    print("\n=== Token-Class Matrix (CLIP-style) ===")
    
    # Get all unique token values from all classes
    all_tokens = []
    for pattern in class_token_patterns.values():
        tokens = pattern.get('tokens', torch.tensor([]))
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.detach().cpu().numpy()
        all_tokens.extend(tokens)
    
    # Get unique tokens
    unique_tokens = sorted(set([float(t) for t in all_tokens]))
    
    # Create token-class matrix
    token_class_matrix = np.zeros((len(unique_tokens), len(class_to_semantic)))
    
    # Fill in the matrix
    for i, token in enumerate(unique_tokens):
        for j, class_idx in enumerate(sorted(class_to_semantic.keys())):
            pattern = class_token_patterns[class_idx]
            tokens = pattern.get('tokens', torch.tensor([]))
            if isinstance(tokens, torch.Tensor):
                tokens = tokens.detach().cpu().numpy()
            
            # Calculate frequency of this token in this class
            token_count = np.sum(tokens == token)
            token_class_matrix[i, j] = token_count
    
    # Normalize by column
    col_sums = token_class_matrix.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1  # Avoid division by zero
    token_class_matrix = token_class_matrix / col_sums
    
    # Create DataFrame
    token_df = pd.DataFrame(
        token_class_matrix, 
        index=[f"Token {i:.2f}" for i in unique_tokens],
        columns=[f"Class {i}" for i in sorted(class_to_semantic.keys())]
    )
    
    # Print as table
    print(tabulate(token_df, headers=token_df.columns, tablefmt="pretty"))
    
    # Save or show heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(token_df, cmap="viridis", cbar_kws={'label': 'Normalized Frequency'})
    plt.title("Token Distribution Across Classes (CLIP-style)")
    plt.xlabel("Classes")
    plt.ylabel("Tokens")
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, "token_class_matrix.png"))
    if show_plots:
        plt.show()
    plt.close()
    
    # 5. Visualize sample values for each semantic feature
    print("\n=== Sample Semantic Feature Values ===")
    
    # Take first few samples, few batches, all semantic features
    samples = semantic_values[:num_viz_samples, :num_viz_batches, :].detach().cpu().numpy()
    
    # Create a table for each batch
    for b in range(samples.shape[1]):
        print(f"\nBatch {b+1}:")
        
        # Create a DataFrame
        batch_samples = samples[:, b, :]
        sample_df = pd.DataFrame(
            batch_samples,
            index=[f"Sample {i+1}" for i in range(batch_samples.shape[0])],
            columns=[f"Feat {i}" for i in range(batch_samples.shape[1])]
        )
        
        # Print as table (only show first 10 features if more)
        if batch_samples.shape[1] > 10:
            print(tabulate(sample_df.iloc[:, :10], headers=sample_df.columns[:10], tablefmt="pretty"))
            print("... (truncated, showing first 10 features only)")
        else:
            print(tabulate(sample_df, headers=sample_df.columns, tablefmt="pretty"))
    
    # 6. Visualize class vs semantic class assignments
    print("\n=== Class to Semantic Class Mapping ===")
    class_mapping = []
    for class_idx, semantic_class in class_to_semantic.items():
        class_mapping.append([class_idx, semantic_class])
    
    print(tabulate(class_mapping, headers=["Class", "Semantic Class"], tablefmt="pretty"))
    
    return token_df, sim_df if len(class_avg_features) > 1 else None

def main():
    parser = argparse.ArgumentParser(description='Visualize semantic features')
    parser.add_argument('--num-features', type=int, default=20, help='Number of base features')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--n-samples', type=int, default=20, help='Number of samples')
    parser.add_argument('--num-classes', type=int, default=5, help='Number of classes')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], 
                       help='Device to use')
    parser.add_argument('--output-dir', type=str, default='logs/visualizations', help='Directory to save outputs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no-plots', action='store_true', help='Disable showing matplotlib plots')
    
    args = parser.parse_args()
    
    # Use 'cuda' if available and requested
    if args.device == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
    elif args.device == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    # Create output directory if specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Extract and visualize semantic features
    data = extract_semantic_features(
        num_features=args.num_features,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
        num_classes=args.num_classes,
        device=device,
        random_seed=args.seed
    )
    
    # Visualize the features
    token_df, sim_df = visualize_semantic_features(
        data,
        output_dir=args.output_dir,
        show_plots=not args.no_plots
    )
    
    # If output directory is specified, save the tables as CSV
    if args.output_dir:
        token_df.to_csv(os.path.join(args.output_dir, "token_class_matrix.csv"))
        if sim_df is not None:
            sim_df.to_csv(os.path.join(args.output_dir, "class_similarity.csv"))
    
    print(f"\nAnalysis complete{' and saved to ' + args.output_dir if args.output_dir else ''}.")

if __name__ == "__main__":
    main()