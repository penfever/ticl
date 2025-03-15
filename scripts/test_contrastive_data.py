"""
Test script to verify the data getting passed into contrastive loss.

This script inspects the relationship between column names and tokens to verify
that the contrastive loss is working correctly in the semantic-aware model.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from transformers import CLIPTokenizerFast, CLIPTextModel
from tabulate import tabulate
import pandas as pd

from ticl.priors.classification_adapter import ClassificationAdapter
from ticl.datasets.semantic_prior_data_loader import get_random_semantic_data, load_semantic_prior_data
from ticl.models.semantic_aware_model import SemanticConsistencyLoss

# Define our own get_clip_text_embeddings function since the one from semantic_aware_model 
# has an undefined logging reference
def get_clip_text_embeddings(texts, clip_model, tokenizer, batch_size=5, device=None):
    """
    Process texts using CLIP text encoder to get embeddings in batches.
    """
    # Setup logging
    logger = logging.getLogger('test_contrastive')
    
    # Auto-detect device if not specified
    if device is None:
        if torch.cuda.is_available():
            processing_device = "cuda"
        elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
            processing_device = "mps"
        else:
            processing_device = "cpu"
    else:
        processing_device = device
    
    # Get model's current device
    model_device = next(clip_model.parameters()).device
    
    # Move model if needed
    if model_device != processing_device:
        logger.info(f"Moving CLIP model from {model_device} to {processing_device}")
        clip_model = clip_model.to(processing_device)
    
    # Process texts in batches
    all_embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
            try:
                # Tokenize and move to device
                tokens = tokenizer(
                    batch_texts, 
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=77  # CLIP's context length
                )
                tokens = {k: v.to(processing_device) for k, v in tokens.items()}
                
                # Get embeddings
                outputs = clip_model(**tokens)
                embeddings = outputs.pooler_output
                all_embeddings.append(embeddings)
                
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                # Try on CPU as fallback
                if processing_device != 'cpu':
                    logger.info(f"Trying CPU fallback")
                    try:
                        cpu_tokens = tokenizer(
                            batch_texts, 
                            return_tensors="pt",
                            padding="max_length",
                            truncation=True,
                            max_length=77
                        )
                        
                        cpu_model = clip_model.to('cpu')
                        outputs = cpu_model(**cpu_tokens)
                        embeddings = outputs.pooler_output
                        embeddings = embeddings.to(processing_device)
                        all_embeddings.append(embeddings)
                        
                        clip_model = clip_model.to(processing_device)
                        logger.info(f"CPU fallback succeeded")
                    except Exception as e2:
                        logger.error(f"CPU fallback also failed: {e2}")
    
    # Move model back if needed
    if model_device != processing_device:
        logger.info(f"Moving CLIP model back to {model_device}")
        clip_model = clip_model.to(model_device)
    
    # Concatenate results
    if all_embeddings:
        logger.info(f"Concatenating {len(all_embeddings)} batches")
        text_embeddings = torch.cat(all_embeddings, dim=0)
        return text_embeddings
    else:
        # Return empty tensor if all failed
        logger.warning(f"No embeddings generated - returning empty tensor")
        return torch.zeros((0, clip_model.config.hidden_size), device=processing_device)

# Set up logger for detailed output
import logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('test_contrastive')

def decode_clip_tokens(tokenizer, tokens):
    """
    Decode CLIP token IDs to strings
    
    Parameters:
    -----------
    tokenizer : CLIPTokenizerFast
        CLIP tokenizer to use for decoding
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
        return tokenizer.decode([token_id])
    else:
        # Batch of tokens
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.detach().cpu().tolist()
        return tokenizer.decode(tokens)

def analyze_column_tokens(tokenizer, semantic_data, column_names=None, top_k=5):
    """
    Analyze the relationship between column names and tokens.
    
    Parameters:
    -----------
    tokenizer : CLIPTokenizerFast
        CLIP tokenizer for decoding tokens
    semantic_data : torch.Tensor
        Tensor of semantic tokens with shape [num_classes, num_tokens]
    column_names : list
        List of column names
    top_k : int
        Number of top tokens to display
        
    Returns:
    --------
    dict
        Mapping of column names to their most common tokens
    """
    if isinstance(semantic_data, torch.Tensor):
        semantic_data = semantic_data.detach().cpu().numpy()
    
    num_classes = semantic_data.shape[0]
    
    # Create default column names if not provided
    if column_names is None:
        column_names = [f"Class_{i}" for i in range(num_classes)]
    else:
        # Ensure we have enough names
        if len(column_names) < num_classes:
            column_names = column_names + [f"Class_{i}" for i in range(len(column_names), num_classes)]
    
    # Create mapping
    column_token_map = {}
    
    # Process each column
    for i in range(semantic_data.shape[0]):
        col_name = column_names[i] if i < len(column_names) else f"Class_{i}"
        
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
            token_str = decode_clip_tokens(tokenizer, int(token))
            decoded_tokens.append({
                'token_id': int(token),
                'token_text': token_str,
                'count': int(count),
                'display': f"{token_str} ({int(token)}): {count}"
            })
        
        # Add to map
        column_token_map[col_name] = decoded_tokens
    
    return column_token_map

def create_column_token_report(column_token_map):
    """
    Create a formatted report showing column-token relationships.
    
    Parameters:
    -----------
    column_token_map : dict
        Mapping from column names to their most common tokens
        
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
    
    return report

def get_column_name_embeddings(tokenizer, text_model, column_names, device='cpu'):
    """
    Generate CLIP embeddings for column names.
    
    Parameters:
    -----------
    tokenizer : CLIPTokenizerFast
        CLIP tokenizer
    text_model : CLIPTextModel
        CLIP text model
    column_names : list
        List of column names
    device : str
        Device to use for processing
        
    Returns:
    --------
    torch.Tensor
        Tensor of column name embeddings
    """
    # Clean column names (remove underscores, etc.)
    cleaned_names = [name.replace("_", " ").replace("-", " ") for name in column_names]
    
    # Process in batches to avoid OOM
    embeddings = get_clip_text_embeddings(
        cleaned_names, 
        text_model, 
        tokenizer, 
        batch_size=16, 
        device=device
    )
    
    return embeddings

def compute_contrastive_alignment(column_embeddings, semantic_data, tokenizer, text_model, column_names=None, device='cpu'):
    """
    Compute alignment scores between column name embeddings and token vectors using CLIP.
    
    Parameters:
    -----------
    column_embeddings : torch.Tensor
        CLIP embeddings for column names
    semantic_data : torch.Tensor
        Tensor of tokens for each column
    tokenizer : CLIPTokenizerFast
        CLIP tokenizer for decoding tokens
    text_model : CLIPTextModel
        CLIP text model for generating embeddings
    column_names : list
        List of column names for display
    device : str
        Device to use for processing
        
    Returns:
    --------
    torch.Tensor
        Similarity matrix between column embeddings and token vectors
    """
    logger = logging.getLogger('test_contrastive')
    logger.info("Computing contrastive alignment with CLIP embeddings")
    
    # Normalize column embeddings
    norm_col_embeddings = torch.nn.functional.normalize(column_embeddings, p=2, dim=1)
    
    # Convert semantic data to array of integers
    if isinstance(semantic_data, torch.Tensor):
        token_vectors = semantic_data.cpu().numpy().astype(np.int32)
    else:
        token_vectors = np.array(semantic_data, dtype=np.int32)
    
    # Create embeddings for each column's tokens using CLIP
    column_token_embeds = []
    
    for i in range(token_vectors.shape[0]):
        # Extract tokens for this column
        col_tokens = token_vectors[i]
        
        # Find most frequent tokens in this column (top 5)
        unique_tokens, counts = np.unique(col_tokens, return_counts=True)
        sorted_indices = np.argsort(-counts)
        top_tokens = unique_tokens[sorted_indices][:5]  # Take top 5 tokens
        
        # Decode the tokens to text
        token_texts = []
        for token in top_tokens:
            if token > 0:  # Skip padding tokens
                decoded = decode_clip_tokens(tokenizer, int(token))
                if decoded.strip():  # Only add non-empty strings
                    token_texts.append(decoded)
        
        # If we have no valid tokens, use the column name
        if not token_texts:
            if column_names and i < len(column_names):
                token_texts = [column_names[i].replace("_", " ")]
            else:
                token_texts = ["unknown"]
        
        # Create a combined text string from these tokens
        combined_text = " ".join(token_texts)
        
        # Get CLIP embedding for this combined text
        with torch.no_grad():
            tokens = tokenizer(
                combined_text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=77
            ).to(device)
            
            token_embedding = text_model(**tokens).pooler_output
            column_token_embeds.append(token_embedding)
    
    # Concatenate all token embeddings
    token_embeddings = torch.cat(column_token_embeds, dim=0)
    
    # Normalize token embeddings
    norm_token_embeds = torch.nn.functional.normalize(token_embeddings, p=2, dim=1)
    
    # Compute alignment (cosine similarity)
    similarities = torch.matmul(norm_col_embeddings, norm_token_embeds.t())
    
    return similarities

def analyze_clip_embeddings(tokenizer, text_model, semantic_data, column_names, device='cpu'):
    """
    Analyze CLIP embeddings for column names and their alignment with token distributions.
    
    Parameters:
    -----------
    tokenizer : CLIPTokenizerFast
        CLIP tokenizer
    text_model : CLIPTextModel
        CLIP text model
    semantic_data : torch.Tensor
        Tensor of tokens for each column
    column_names : list
        List of column names
    device : str
        Device to use for processing
        
    Returns:
    --------
    dict
        Dictionary with analysis results
    """
    logger.info("Generating CLIP embeddings for column names...")
    
    # Get column name embeddings
    column_embeddings = get_column_name_embeddings(
        tokenizer, text_model, column_names, device
    )
    
    logger.info(f"Generated column name embeddings with shape: {column_embeddings.shape}")
    
    # Compute similarity matrix with improved approach
    logger.info("Computing alignment between column names and token vectors using CLIP...")
    similarities = compute_contrastive_alignment(
        column_embeddings, semantic_data, tokenizer, text_model, column_names, device
    )
    
    logger.info(f"Generated similarity matrix with shape: {similarities.shape}")
    
    # Convert to DataFrame for better visualization
    sim_df = pd.DataFrame(
        similarities.detach().cpu().numpy(),
        index=column_names,
        columns=column_names
    )
    
    # Get index of maximum similarity for each column
    max_indices = similarities.argmax(dim=1).cpu().numpy()
    max_similarities = similarities.max(dim=1)[0].cpu().numpy()
    
    # Check diagonal elements (self-similarity)
    diagonal_indices = np.arange(len(column_names))
    diagonal_matches = (max_indices == diagonal_indices)
    diagonal_match_count = diagonal_matches.sum()
    
    # Create match report
    match_data = []
    for i, col_name in enumerate(column_names):
        max_idx = max_indices[i]
        matched_col = column_names[max_idx]
        sim_value = max_similarities[i]
        is_correct = (max_idx == i)
        
        # Get the top tokens for this column
        if isinstance(semantic_data, torch.Tensor):
            col_tokens = semantic_data[i].cpu().numpy()
        else:
            col_tokens = semantic_data[i]
            
        unique_tokens, counts = np.unique(col_tokens, return_counts=True)
        sorted_indices = np.argsort(-counts)
        top_tokens = unique_tokens[sorted_indices][:5]
        
        # Decode top tokens
        token_texts = []
        for token in top_tokens:
            if token > 0:
                decoded = decode_clip_tokens(tokenizer, int(token))
                if decoded.strip():
                    token_texts.append(decoded)
                    
        # Add this information to match data
        match_data.append({
            'column': col_name,
            'best_match': matched_col,
            'similarity': sim_value,
            'is_correct': is_correct,
            'top_tokens': token_texts
        })
    
    return {
        'column_embeddings': column_embeddings,
        'similarities': similarities,
        'similarity_df': sim_df,
        'max_indices': max_indices,
        'diagonal_match_count': diagonal_match_count,
        'match_accuracy': diagonal_match_count / len(column_names),
        'match_data': match_data
    }

def create_visualizations(similarity_df, match_data, output_dir=None):
    """
    Create visualizations for the contrastive alignment analysis.
    
    Parameters:
    -----------
    similarity_df : pd.DataFrame
        DataFrame with similarity values
    match_data : list
        List of dictionaries with match information
    output_dir : str, optional
        Directory to save output files
        
    Returns:
    --------
    None
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Create similarity heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity_df, annot=True, cmap="YlGnBu", 
                fmt=".3f", linewidths=0.5)
    plt.title("Contrastive Alignment: Column Names to Token Vectors", fontsize=16)
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, "contrastive_alignment.png"), dpi=300)
    plt.show()
    
    # Create enhanced match report table with token information
    match_report = "=== Column Name to Token Vector Matching ===\n\n"
    
    table_rows = []
    for d in match_data:
        # Format token information
        tokens_str = ", ".join(d.get('top_tokens', [])[:3])  # Show first 3 tokens
        if len(d.get('top_tokens', [])) > 3:
            tokens_str += "..."
            
        row = [
            d['column'], 
            d['best_match'], 
            f"{d['similarity']:.4f}", 
            "✓" if d['is_correct'] else "✗",
            tokens_str
        ]
        table_rows.append(row)
    
    match_report += tabulate(
        table_rows,
        headers=["Column", "Best Match", "Similarity", "Correct", "Top Tokens"],
        tablefmt="github"
    )
    
    print(match_report)
    
    if output_dir:
        with open(os.path.join(output_dir, "match_report.txt"), 'w') as f:
            f.write(match_report)
    
    # Create accuracy summary
    correct_count = sum(1 for d in match_data if d['is_correct'])
    total_count = len(match_data)
    accuracy = correct_count / total_count * 100
    
    print(f"\nSummary: {correct_count}/{total_count} columns correctly matched ({accuracy:.2f}%)")
    
    # Create detailed match report with token alignment information
    detailed_report = "=== Detailed Token Alignment Analysis ===\n\n"
    
    for d in match_data:
        col_name = d['column']
        best_match = d['best_match']
        sim_value = d['similarity']
        is_correct = d['is_correct']
        tokens = d.get('top_tokens', [])
        
        status = "✓ CORRECT" if is_correct else "✗ INCORRECT"
        
        detailed_report += f"Column: {col_name}\n"
        detailed_report += f"{'=' * (len(col_name) + 8)}\n"
        detailed_report += f"Best match: {best_match} (similarity: {sim_value:.4f}) {status}\n\n"
        
        if tokens:
            detailed_report += "Top tokens in this column:\n"
            for i, token in enumerate(tokens):
                detailed_report += f"  {i+1}. {token}\n"
        else:
            detailed_report += "No meaningful tokens found in this column.\n"
            
        detailed_report += "\n"
    
    # Save detailed report
    if output_dir:
        with open(os.path.join(output_dir, "detailed_token_report.txt"), 'w') as f:
            f.write(detailed_report)
    
    # Create bar chart of similarities
    plt.figure(figsize=(12, 6))
    
    # Extract data
    columns = [d['column'] for d in match_data]
    similarities = [d['similarity'] for d in match_data]
    is_correct = [d['is_correct'] for d in match_data]
    
    # Create color mapping
    colors = ['green' if c else 'red' for c in is_correct]
    
    # Create bar chart
    bars = plt.bar(columns, similarities, color=colors)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Correct Match'),
        Patch(facecolor='red', label='Incorrect Match')
    ]
    plt.legend(handles=legend_elements)
    
    # Format plot
    plt.title("Column Name to Token Vector Similarity", fontsize=16)
    plt.xlabel("Column Name")
    plt.ylabel("Maximum Similarity")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, "similarity_bars.png"), dpi=300)
    plt.show()

def main():
    logger.info("Starting contrastive data inspection test")
    
    # Use CPU by default, but check for CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = "contrastive_test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load CLIP models
    logger.info("Loading CLIP tokenizer and model...")
    model_name = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizerFast.from_pretrained(model_name)
    text_model = CLIPTextModel.from_pretrained(model_name).to(device)
    
    # Load semantic data
    logger.info("Loading semantic data with token filtering enabled...")
    try:
        # Try loading real data first with token filtering enabled
        column_name_tokens, column_value_tokens = load_semantic_prior_data(
            log_file_path="ticl/datasets/completed_columns.json",
            use_cache=True,
            filter_tokens=True  # Enable token filtering to improve matches
        )
        logger.info(f"Loaded {len(column_name_tokens)} columns from semantic data")
        
        # Get column names from data
        column_names = list(column_name_tokens.keys())
        
        # Convert token dictionaries to tensor format
        semantic_data = torch.stack([column_value_tokens[col] for col in column_names])
        logger.info(f"Created semantic data tensor with shape: {semantic_data.shape}")
        
    except (FileNotFoundError, Exception) as e:
        logger.warning(f"Could not load real semantic data: {e}")
        logger.info("Falling back to random semantic data generation")
        
        # Generate random data instead with token filtering
        num_classes = 10
        semantic_data, column_names = get_random_semantic_data(
            num_classes=num_classes,
            seed=42,
            use_cache=True,
            filter_tokens=True  # Enable token filtering to improve matches
        )
        logger.info(f"Generated random semantic data with shape: {semantic_data.shape}")
    
    # Limit to first 10 columns for clarity if we have too many
    max_cols = 10
    if len(column_names) > max_cols:
        logger.info(f"Limiting analysis to first {max_cols} columns for clarity")
        column_names = column_names[:max_cols]
        if isinstance(semantic_data, torch.Tensor):
            semantic_data = semantic_data[:max_cols]
        else:
            semantic_data = semantic_data[:max_cols]
    
    # Move semantic data to device
    semantic_data = semantic_data.to(device)
    
    # Analyze column tokens
    logger.info("Analyzing column tokens...")
    column_token_map = analyze_column_tokens(
        tokenizer, semantic_data, column_names
    )
    
    # Create and print token report
    token_report = create_column_token_report(column_token_map)
    print(token_report)
    
    # Save token report
    with open(os.path.join(output_dir, "column_token_report.txt"), 'w') as f:
        f.write(token_report)
    
    # Analyze CLIP embeddings and token alignment
    logger.info("Analyzing CLIP embeddings and contrastive alignment...")
    analysis_results = analyze_clip_embeddings(
        tokenizer, text_model, semantic_data, column_names, device
    )
    
    # Create visualizations
    logger.info("Creating visualizations...")
    create_visualizations(
        analysis_results['similarity_df'],
        analysis_results['match_data'],
        output_dir
    )
    
    # Print summary
    match_accuracy = analysis_results['match_accuracy'] * 100
    logger.info(f"Contrastive alignment accuracy: {match_accuracy:.2f}%")
    logger.info(f"Analysis complete. Results saved to: {output_dir}")

if __name__ == "__main__":
    main()