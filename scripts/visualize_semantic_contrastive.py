import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from transformers import CLIPTokenizerFast

from ticl.priors.classification_adapter import ClassificationAdapter
from ticl.datasets.semantic_prior_data_loader import get_random_semantic_data

# Load CLIP tokenizer for decoding tokens
CLIP_TOKENIZER = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")

class DummyPrior:
    """Simple dummy prior for feature generation"""
    def __init__(self):
        pass
        
    def get_batch(self, batch_size, n_samples, num_features, device, **kwargs):
        x = torch.randn(n_samples, batch_size, num_features, device=device)
        y = torch.randint(0, 5, (n_samples, batch_size, 1), device=device).float()
        return x, y, y

def generate_semantic_data(num_classes=10, seed=42, device='cpu'):
    """Generate semantic data directly, bypassing the ClassificationAdapter"""
    semantic_data, semantic_data_column_names = get_random_semantic_data(
        num_classes=num_classes,
        seed=seed,
        use_cache=True
    )
    
    return semantic_data.to(device), semantic_data_column_names

def decode_clip_tokens(tokens):
    """
    Decode CLIP token IDs to strings
    
    Parameters:
    -----------
    tokens : torch.Tensor or int
        Token IDs to decode
        
    Returns:
    --------
    str
        Decoded token string
    """
    if isinstance(tokens, int) or (isinstance(tokens, torch.Tensor) and tokens.numel() == 1):
        # Single token
        token_id = tokens if isinstance(tokens, int) else tokens.item()
        return CLIP_TOKENIZER.decode([token_id])
    else:
        # Batch of tokens
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.detach().cpu().tolist()
        return CLIP_TOKENIZER.decode(tokens)

def create_contrastive_matrix(semantic_data, column_names=None, k=5):
    """
    Create a contrastive matrix between semantic classes
    
    This cosine similarity matrix compares semantic classes (column names) with each other
    to understand relationships between different column distributions
    """
    
    # Ensure semantic data is on CPU and convert to numpy
    if isinstance(semantic_data, torch.Tensor):
        semantic_data = semantic_data.detach().cpu().numpy()
    
    # Number of semantic classes
    num_classes = semantic_data.shape[0]
    
    # Calculate pairwise cosine similarity
    similarity = cosine_similarity(semantic_data)
    
    # Create a DataFrame with class names if provided
    if column_names is None:
        column_names = [f"Class {i}" for i in range(num_classes)]
    else:
        # Truncate to match number of classes
        column_names = column_names[:num_classes]
        # Add index for any missing names
        if len(column_names) < num_classes:
            column_names = column_names + [f"Class {i}" for i in range(len(column_names), num_classes)]
    
    # Create DataFrame
    sim_df = pd.DataFrame(similarity, index=column_names, columns=column_names)
    
    return sim_df

def plot_contrastive_matrix(sim_df, output_path=None, show_plot=True, highlight_diagonal=True,
                           figsize=(12, 10), title="Semantic Class Similarity Matrix"):
    """Plot the contrastive matrix in a CLIP-like format"""
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Create a custom colormap if highlighting diagonal
    if highlight_diagonal:
        # Calculate max non-diagonal value for better color scaling
        max_non_diag = sim_df.values[~np.eye(sim_df.shape[0], dtype=bool)].max()
        # Create custom colormap
        cmap = LinearSegmentedColormap.from_list(
            'custom_cmap', 
            [(0, 'white'), (max_non_diag/2, 'lightblue'), (max_non_diag, 'blue'), (1, 'darkred')]
        )
    else:
        cmap = "YlGnBu"
    
    # Plot heatmap
    sns.heatmap(sim_df, annot=True, cmap=cmap, 
                fmt=".2f", linewidths=0.5, 
                cbar_kws={'label': 'Cosine Similarity'})
    
    # Add title and tweak appearance
    plt.title(title, fontsize=16)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    
    # Save if path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    # Show if requested
    if show_plot:
        plt.show()
    
    plt.close()

def create_token_class_matrix(semantic_data, column_names=None, normalize=True, max_tokens=50):
    """
    Create a token-class matrix similar to CLIP's visualization
    
    This creates a matrix where:
    - Columns are semantic classes (column names)
    - Rows are actual CLIP tokens (decoded from token IDs)
    """
    
    # Ensure semantic data is on CPU and convert to numpy
    if isinstance(semantic_data, torch.Tensor):
        semantic_data = semantic_data.detach().cpu().numpy()
    
    # Number of semantic classes and tokens
    num_classes, num_tokens = semantic_data.shape
    
    # Get unique token values across all classes
    token_values = np.unique(semantic_data)
    
    # Limit the number of tokens to display if there are too many
    if len(token_values) > max_tokens:
        # Choose the most frequent tokens
        token_counts = {token: np.sum(semantic_data == token) for token in token_values}
        sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
        token_values = np.array([t[0] for t in sorted_tokens[:max_tokens]])
    
    # Create token-class matrix
    token_class_matrix = np.zeros((len(token_values), num_classes))
    
    # Fill the matrix with token occurrences
    for i, token in enumerate(token_values):
        for j in range(num_classes):
            token_class_matrix[i, j] = np.sum(semantic_data[j] == token)
    
    # Normalize if requested
    if normalize:
        # Normalize by column (class)
        col_sums = token_class_matrix.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1  # Avoid division by zero
        token_class_matrix = token_class_matrix / col_sums
    
    # Create column names if not provided
    if column_names is None:
        column_names = [f"Class {i}" for i in range(num_classes)]
    else:
        # Truncate to match number of classes
        column_names = column_names[:num_classes]
        # Add index for any missing names
        if len(column_names) < num_classes:
            column_names = column_names + [f"Class {i}" for i in range(len(column_names), num_classes)]
    
    # Decode token values to strings
    token_strings = [f"{decode_clip_tokens(int(token))} ({int(token)})" for token in token_values]
    
    # Create DataFrame
    token_df = pd.DataFrame(
        token_class_matrix, 
        index=token_strings,
        columns=column_names
    )
    
    return token_df

def plot_token_class_matrix(token_df, output_path=None, show_plot=True, 
                           figsize=(14, 12), title="Token Distribution Across Semantic Classes"):
    """Plot the token-class matrix in a CLIP-like format"""
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(token_df, cmap="viridis", 
                cbar_kws={'label': 'Normalized Frequency'})
    
    # Add title and labels
    plt.title(title, fontsize=16)
    plt.xlabel("Semantic Classes", fontsize=14)
    plt.ylabel("Token Values", fontsize=14)
    
    # Adjust for readability
    plt.xticks(rotation=45, ha="right", fontsize=10)
    if token_df.shape[0] > 20:
        # If many tokens, only show a subset of labels
        plt.yticks(np.arange(0, token_df.shape[0], 5), fontsize=10)
    else:
        plt.yticks(fontsize=10)
    
    plt.tight_layout()
    
    # Save if path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    # Show if requested
    if show_plot:
        plt.show()
    
    plt.close()

def create_column_token_similarity(semantic_data, column_names=None, top_k=20):
    """
    Create a similarity matrix between columns and their most distinctive tokens
    
    Parameters:
    -----------
    semantic_data : torch.Tensor
        Tensor of semantic tokens with shape [num_semantic_classes, num_tokens]
    column_names : list
        List of column names
    top_k : int
        Number of top tokens to display
        
    Returns:
    --------
    pd.DataFrame
        Similarity matrix between columns and tokens
    """
    # Ensure semantic data is on CPU
    if isinstance(semantic_data, torch.Tensor):
        semantic_data = semantic_data.detach().cpu().numpy()
    
    # Number of semantic classes
    num_classes = semantic_data.shape[0]
    
    # Create column names if not provided
    if column_names is None:
        column_names = [f"Class {i}" for i in range(num_classes)]
    else:
        # Truncate to match number of classes
        column_names = column_names[:num_classes]
        # Add index for any missing names
        if len(column_names) < num_classes:
            column_names = column_names + [f"Class {i}" for i in range(len(column_names), num_classes)]
    
    # Get unique tokens across all columns
    all_tokens = np.unique(semantic_data)
    
    # If there are too many tokens, select the most informative ones
    if len(all_tokens) > top_k:
        # Calculate token-class frequencies
        token_class_freqs = np.zeros((len(all_tokens), num_classes))
        
        for i, token in enumerate(all_tokens):
            for j in range(num_classes):
                token_class_freqs[i, j] = np.sum(semantic_data[j] == token)
        
        # Normalize by column (class)
        col_sums = token_class_freqs.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1  # Avoid division by zero
        token_class_freqs = token_class_freqs / col_sums
        
        # Calculate entropy of each token's distribution (lower means more distinctive)
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        entropy = -np.sum(token_class_freqs * np.log(token_class_freqs + epsilon), axis=1)
        
        # Select tokens with lowest entropy (most informative/distinctive)
        top_indices = np.argsort(entropy)[:top_k]
        selected_tokens = all_tokens[top_indices]
    else:
        selected_tokens = all_tokens
    
    # Create one-hot encoded representation for each token across classes
    # This will show which classes each token appears in
    token_class_matrix = np.zeros((len(selected_tokens), num_classes))
    
    for i, token in enumerate(selected_tokens):
        for j in range(num_classes):
            # Count occurrences and normalize by column length
            token_class_matrix[i, j] = np.sum(semantic_data[j] == token) / semantic_data.shape[1]
    
    # Decode token values to strings
    token_strings = [f"{decode_clip_tokens(int(token))} ({int(token)})" for token in selected_tokens]
    
    # Create DataFrame
    column_token_df = pd.DataFrame(
        token_class_matrix, 
        index=token_strings,
        columns=column_names
    )
    
    return column_token_df

def plot_column_token_similarity(column_token_df, output_path=None, show_plot=True, 
                               figsize=(14, 12), title="Column-Token Similarity Matrix"):
    """Plot the column-token similarity matrix"""
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(column_token_df, cmap="viridis", 
                cbar_kws={'label': 'Normalized Frequency'})
    
    # Add title and labels
    plt.title(title, fontsize=16)
    plt.xlabel("Semantic Classes", fontsize=14)
    plt.ylabel("Token Values", fontsize=14)
    
    # Adjust for readability
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    
    plt.tight_layout()
    
    # Save if path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    # Show if requested
    if show_plot:
        plt.show()
    
    plt.close()

def create_plaintext_visualization(sim_df, token_df, column_token_df=None, output_path=None):
    """Create plaintext visualization of the matrices"""
    
    # Create a formatted string
    output = "=== Semantic Class Similarity Matrix ===\n\n"
    output += tabulate(sim_df, headers=sim_df.columns, tablefmt="grid")
    
    output += "\n\n=== Token-Class Distribution Matrix ===\n\n"
    
    # If token matrix is large, show a sample
    if token_df.shape[0] > 20:
        output += tabulate(token_df.head(20), headers=token_df.columns, tablefmt="grid")
        output += f"\n[Showing 20/{token_df.shape[0]} tokens rows]\n"
    else:
        output += tabulate(token_df, headers=token_df.columns, tablefmt="grid")
    
    # Add column-token similarity if provided
    if column_token_df is not None:
        output += "\n\n=== Column-Token Similarity Matrix (Most Distinctive Tokens) ===\n\n"
        
        # If matrix is large, show a sample
        if column_token_df.shape[0] > 20:
            output += tabulate(column_token_df.head(20), headers=column_token_df.columns, tablefmt="grid")
            output += f"\n[Showing 20/{column_token_df.shape[0]} tokens rows]\n"
        else:
            output += tabulate(column_token_df, headers=column_token_df.columns, tablefmt="grid")
    
    # Print to console
    print(output)
    
    # Save to file if path provided
    if output_path:
        with open(output_path, 'w') as f:
            f.write(output)
        print(f"Plaintext visualization saved to {output_path}")

def analyze_column_tokens(semantic_data, column_names=None, top_k=10):
    """
    Analyze and display most common tokens for each semantic column
    
    Parameters:
    -----------
    semantic_data : torch.Tensor
        Tensor of semantic tokens with shape [num_semantic_classes, num_tokens]
    column_names : list
        List of column names
    top_k : int
        Number of top tokens to display per column
        
    Returns:
    --------
    dict
        Mapping from column names to their most common tokens
    """
    # Ensure semantic data is on CPU
    if isinstance(semantic_data, torch.Tensor):
        semantic_data = semantic_data.detach().cpu().numpy()
    
    # Create mapping
    column_token_map = {}
    
    # Process each column
    for i in range(semantic_data.shape[0]):
        # Get column name or default
        col_name = column_names[i] if column_names and i < len(column_names) else f"Class_{i}"
        
        # Get tokens for this column
        column_tokens = semantic_data[i]
        
        # Count token frequencies
        unique_tokens, counts = np.unique(column_tokens, return_counts=True)
        
        # Sort by frequency (descending)
        sorted_indices = np.argsort(-counts)
        top_tokens = unique_tokens[sorted_indices][:top_k]
        top_counts = counts[sorted_indices][:top_k]
        
        # Decode tokens
        decoded_tokens = []
        for token, count in zip(top_tokens, top_counts):
            token_str = decode_clip_tokens(int(token))
            decoded_tokens.append({
                'token_id': int(token),
                'token_text': token_str,
                'count': int(count),
                'display': f"{token_str} ({int(token)}): {count}"
            })
        
        # Add to map
        column_token_map[col_name] = decoded_tokens
    
    return column_token_map

def create_column_token_report(column_token_map, output_path=None):
    """
    Create a report showing the most common tokens for each column
    
    Parameters:
    -----------
    column_token_map : dict
        Mapping from column names to their most common tokens
    output_path : str, optional
        Path to save the report
        
    Returns:
    --------
    str
        The report text
    """
    report = "=== Column-Token Association Report ===\n\n"
    
    for col_name, tokens in column_token_map.items():
        report += f"Column: {col_name}\n"
        report += "-" * (len(col_name) + 8) + "\n"
        
        for i, token_info in enumerate(tokens):
            report += f"{i+1}. {token_info['display']}\n"
        
        report += "\n"
    
    # Print to console
    print(report)
    
    # Save if path provided
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"Column-token report saved to {output_path}")
    
    return report

def main():
    parser = argparse.ArgumentParser(description='Visualize semantic feature contrastive relationships')
    parser.add_argument('--num-classes', type=int, default=10, help='Number of semantic classes')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], 
                       help='Device to use')
    parser.add_argument('--output-dir', type=str, default='logs/visualizations', help='Directory to save outputs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no-plots', action='store_true', help='Disable showing matplotlib plots')
    parser.add_argument('--max-tokens', type=int, default=50, help='Maximum number of tokens to display')
    parser.add_argument('--top-k', type=int, default=10, help='Top-K tokens to show per column')
    
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
    
    # Generate semantic data
    print(f"Generating semantic data with {args.num_classes} classes...")
    semantic_data, column_names = generate_semantic_data(
        num_classes=args.num_classes,
        seed=args.seed,
        device=device
    )
    
    # Print basic information about the data
    print(f"Generated semantic data shape: {semantic_data.shape}")
    print(f"Number of column names: {len(column_names) if column_names else 0}")
    
    # Create matrices
    sim_df = create_contrastive_matrix(semantic_data, column_names)
    token_df = create_token_class_matrix(semantic_data, column_names, max_tokens=args.max_tokens)
    
    # Create column-token similarity matrix (showing most distinctive tokens per column)
    column_token_df = create_column_token_similarity(semantic_data, column_names, top_k=args.max_tokens)
    
    # Analyze column-token relationships
    column_token_map = analyze_column_tokens(semantic_data, column_names, top_k=args.top_k)
    
    # Create plaintext visualization
    plaintext_path = os.path.join(args.output_dir, "semantic_visualization.txt") if args.output_dir else None
    create_plaintext_visualization(sim_df, token_df, column_token_df, plaintext_path)
    
    # Create column-token report
    report_path = os.path.join(args.output_dir, "column_token_report.txt") if args.output_dir else None
    create_column_token_report(column_token_map, report_path)
    
    # Plot visualizations
    if not args.no_plots:
        # Plot similarity matrix
        sim_path = os.path.join(args.output_dir, "similarity_matrix.png") if args.output_dir else None
        plot_contrastive_matrix(sim_df, sim_path, show_plot=not args.no_plots,
                               title=f"Semantic Class Similarity Matrix (n={args.num_classes})")
        
        # Plot token-class matrix
        token_path = os.path.join(args.output_dir, "token_class_matrix.png") if args.output_dir else None
        plot_token_class_matrix(token_df, token_path, show_plot=not args.no_plots,
                               title=f"Token Distribution Across {args.num_classes} Semantic Classes")
        
        # Plot column-token similarity matrix
        column_token_path = os.path.join(args.output_dir, "column_token_similarity.png") if args.output_dir else None
        plot_column_token_similarity(column_token_df, column_token_path, show_plot=not args.no_plots,
                                   title=f"Most Distinctive Tokens per Column (n={args.num_classes})")
    
    # Save matrices as CSV
    if args.output_dir:
        sim_df.to_csv(os.path.join(args.output_dir, "similarity_matrix.csv"))
        token_df.to_csv(os.path.join(args.output_dir, "token_class_matrix.csv"))
        column_token_df.to_csv(os.path.join(args.output_dir, "column_token_similarity.csv"))
    
    print(f"Analysis complete{' and saved to ' + args.output_dir if args.output_dir else ''}.")

if __name__ == "__main__":
    main()