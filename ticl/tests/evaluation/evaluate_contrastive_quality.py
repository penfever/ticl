"""
Evaluate the quality of semantic contrastive pairings with and without token filtering.
"""

import torch
import numpy as np
from transformers import CLIPTokenizerFast, CLIPTextModel
from ticl.datasets.semantic_prior_data_loader import get_random_semantic_data

def evaluate_contrastive_quality(num_test_columns=5, tensor_size=100):
    """
    Evaluate how well column names match with their token distributions
    with and without token filtering.
    """
    print("=" * 80)
    print("EVALUATING SEMANTIC CONTRASTIVE PAIRING QUALITY")
    print("=" * 80)
    
    # Use same seed for reproducibility
    seed = 42
    
    # Load models for text encoding
    tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
    text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    
    # Generate data with and without filtering
    unfiltered, column_names = get_random_semantic_data(
        num_classes=num_test_columns,
        tensor_size=tensor_size,
        seed=seed,
        filter_tokens=False
    )
    
    filtered, _ = get_random_semantic_data(
        num_classes=num_test_columns,
        tensor_size=tensor_size,
        seed=seed,
        filter_tokens=True
    )
    
    print(f"Testing with {num_test_columns} columns:")
    for i, name in enumerate(column_names):
        print(f"  {i+1}. {name}")
    
    # Encode column names
    column_texts = [name.replace("_", " ") for name in column_names]
    column_inputs = tokenizer(
        column_texts, 
        padding=True, 
        return_tensors="pt",
        max_length=77,
        truncation=True
    )
    
    with torch.no_grad():
        column_embeddings = text_model(**column_inputs).last_hidden_state[:, 0, :]
        column_embeddings = column_embeddings / column_embeddings.norm(dim=1, keepdim=True)
    
    # We'll encode tokens in batches to avoid memory issues
    def encode_token_batch(token_ids, batch_size=32):
        # Filter -100 values
        valid_mask = token_ids != -100
        valid_tokens = token_ids[valid_mask].tolist()
        
        # Handle empty case
        if not valid_tokens:
            return torch.zeros((1, column_embeddings.shape[1]))
        
        # Encode each token
        all_embeddings = []
        for i in range(0, len(valid_tokens), batch_size):
            batch = valid_tokens[i:i+batch_size]
            token_inputs = tokenizer(
                [tokenizer.decode([t]) for t in batch], 
                padding=True, 
                return_tensors="pt",
                max_length=77,
                truncation=True
            )
            
            with torch.no_grad():
                batch_embeddings = text_model(**token_inputs).last_hidden_state[:, 0, :]
                batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=1, keepdim=True)
                all_embeddings.append(batch_embeddings)
        
        # Concatenate embeddings
        if all_embeddings:
            return torch.cat(all_embeddings)
        else:
            return torch.zeros((1, column_embeddings.shape[1]))
    
    # Function to calculate similarity matrix
    def calculate_similarity_matrix(data_tensor, column_embeds):
        similarities = {}
        
        # For each column, encode its tokens and compare to all column names
        for col_idx in range(data_tensor.shape[0]):
            # Get token embeddings
            token_embeds = encode_token_batch(data_tensor[col_idx])
            
            # Average token embeddings
            avg_token_embed = token_embeds.mean(dim=0, keepdim=True)
            avg_token_embed = avg_token_embed / avg_token_embed.norm(dim=1, keepdim=True)
            
            # Calculate similarities
            similarity = torch.matmul(avg_token_embed, column_embeds.T).squeeze()
            similarities[col_idx] = similarity.tolist()
        
        return similarities
    
    print("\nCalculating semantic similarities...")
    
    # Calculate similarity matrices
    unfiltered_sim = calculate_similarity_matrix(unfiltered, column_embeddings)
    filtered_sim = calculate_similarity_matrix(filtered, column_embeddings)
    
    # Print similarity matrices
    print("\nUnfiltered token similarity matrix:")
    print_similarity_matrix(unfiltered_sim, column_names)
    
    print("\nFiltered token similarity matrix:")
    print_similarity_matrix(filtered_sim, column_names)
    
    # Calculate average diagonal (correct match) values
    unfiltered_diagonal = [unfiltered_sim[i][i] for i in range(num_test_columns)]
    filtered_diagonal = [filtered_sim[i][i] for i in range(num_test_columns)]
    
    avg_unfiltered = sum(unfiltered_diagonal) / len(unfiltered_diagonal)
    avg_filtered = sum(filtered_diagonal) / len(filtered_diagonal)
    
    # Calculate off-diagonal (incorrect match) averages
    off_diagonal_unfiltered = []
    off_diagonal_filtered = []
    
    for i in range(num_test_columns):
        for j in range(num_test_columns):
            if i != j:
                off_diagonal_unfiltered.append(unfiltered_sim[i][j])
                off_diagonal_filtered.append(filtered_sim[i][j])
    
    avg_off_unfiltered = sum(off_diagonal_unfiltered) / len(off_diagonal_unfiltered)
    avg_off_filtered = sum(off_diagonal_filtered) / len(off_diagonal_filtered)
    
    # Calculate contrast (diagonal minus off-diagonal)
    contrast_unfiltered = avg_unfiltered - avg_off_unfiltered
    contrast_filtered = avg_filtered - avg_off_filtered
    
    # Print results
    print("\nResults:")
    print(f"  Correct match similarity (unfiltered): {avg_unfiltered:.4f}")
    print(f"  Correct match similarity (filtered):   {avg_filtered:.4f}")
    print(f"  Improvement: {(avg_filtered - avg_unfiltered) * 100:.2f}%")
    
    print(f"\n  Incorrect match similarity (unfiltered): {avg_off_unfiltered:.4f}")
    print(f"  Incorrect match similarity (filtered):   {avg_off_filtered:.4f}")
    print(f"  Improvement: {(avg_off_unfiltered - avg_off_filtered) * 100:.2f}%")
    
    print(f"\n  Contrast (unfiltered): {contrast_unfiltered:.4f}")
    print(f"  Contrast (filtered):   {contrast_filtered:.4f}")
    print(f"  Improvement: {(contrast_filtered - contrast_unfiltered) * 100:.2f}%")
    
    print("\nAnalysis:")
    if contrast_filtered > contrast_unfiltered:
        print("✅ IMPROVED: Token filtering improves semantic contrast between column names and token distributions")
        
        improvement = (contrast_filtered - contrast_unfiltered) / contrast_unfiltered * 100
        if improvement > 50:
            print(f"   Significant improvement: +{improvement:.1f}% semantic contrast!")
        elif improvement > 20:
            print(f"   Good improvement: +{improvement:.1f}% semantic contrast")
        else:
            print(f"   Modest improvement: +{improvement:.1f}% semantic contrast")
    else:
        print("❌ NOT IMPROVED: Token filtering did not improve semantic contrast")
    
    print("=" * 80)

def print_similarity_matrix(similarities, column_names):
    """Print a similarity matrix in a readable format."""
    n = len(column_names)
    
    # Print header
    header = "          "
    for i, name in enumerate(column_names):
        header += f"{i+1:<10}"
    print(header)
    
    # Print column names
    name_row = "          "
    for i, name in enumerate(column_names):
        name_short = name[:8]
        name_row += f"{name_short:<10}"
    print(name_row)
    
    # Print separator
    print("=" * (10 + 10 * n))
    
    # Print matrix rows
    for i, name in enumerate(column_names):
        row = f"{i+1:<2} {name[:7]:<8}"
        for j in range(n):
            sim = similarities[i][j]
            # Highlight diagonal elements
            if i == j:
                row += f"\033[1m{sim:.4f}\033[0m    "
            else:
                row += f"{sim:.4f}    "
        print(row)

if __name__ == "__main__":
    evaluate_contrastive_quality()