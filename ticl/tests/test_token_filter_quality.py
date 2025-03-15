"""
Test script to visualize the impact of token filtering on semantic feature quality.
"""

import torch
from transformers import CLIPTokenizerFast
from ticl.datasets.semantic_prior_data_loader import get_random_semantic_data, get_unwanted_token_ids

def decode_tokens(tokens, tokenizer):
    """Decode token IDs to text for visualization."""
    results = []
    for token_id in tokens:
        if token_id == -100:
            results.append("[FILTERED]")
        else:
            word = tokenizer.decode([token_id]).strip()
            results.append(word if word else f"<{token_id}>") 
    return results

def visualize_token_filtering(num_tokens=20):
    """
    Visualize the effect of token filtering by generating random tokens
    and showing them before and after filtering.
    """
    print("=" * 80)
    print("TESTING TOKEN FILTERING FOR SEMANTIC FEATURES")
    print("=" * 80)
    
    # Load tokenizer
    tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
    
    # Get unwanted token IDs
    unwanted_token_ids = get_unwanted_token_ids(tokenizer)
    print(f"Number of unwanted tokens: {len(unwanted_token_ids)}")
    print("Examples of unwanted tokens:", end=" ")
    for i in range(min(5, len(unwanted_token_ids))):
        token_id = unwanted_token_ids[i].item()
        text = tokenizer.decode([token_id])
        print(f"'{text}' ({token_id})", end=", ")
    print("...")
    
    # Generate data with and without filtering
    seed = 42
    print("\nGenerating data with seed", seed)
    
    unfiltered, column_names = get_random_semantic_data(
        num_classes=2,
        tensor_size=100,
        seed=seed,
        filter_tokens=False
    )
    
    filtered, _ = get_random_semantic_data(
        num_classes=2,
        tensor_size=100,
        seed=seed,
        filter_tokens=True
    )
    
    # Count filtered tokens
    filtered_tokens = (filtered == -100).sum().item()
    total_tokens = filtered.numel()
    filter_percentage = (filtered_tokens / total_tokens) * 100
    
    print(f"\nFiltered tokens: {filtered_tokens}/{total_tokens} ({filter_percentage:.1f}%)")
    
    # Visualize some sample tokens before and after filtering
    print("\nSample tokens before and after filtering:")
    print("-" * 80)
    print("Column: ", column_names[0])
    
    # Select a range of tokens to display
    start_idx = 20  # Skip the first few which might be column name related
    tokens_to_show = min(num_tokens, unfiltered.shape[1] - start_idx)
    
    unfiltered_sample = unfiltered[0, start_idx:start_idx+tokens_to_show].tolist()
    filtered_sample = filtered[0, start_idx:start_idx+tokens_to_show].tolist()
    
    unfiltered_text = decode_tokens(unfiltered_sample, tokenizer)
    filtered_text = decode_tokens(filtered_sample, tokenizer)
    
    # Print tokens side by side
    print("{:<15} {:<15} {:<15}".format("Token ID", "Before", "After"))
    print("-" * 45)
    
    for i in range(tokens_to_show):
        token_id = unfiltered_sample[i]
        before = unfiltered_text[i]
        after = filtered_text[i]
        print("{:<15} {:<15} {:<15}".format(token_id, before, after))
    
    print("\n" + "=" * 80)
    print("Analysis:")
    print("=" * 80)
    print("The token filtering removes:")
    print("1. Punctuation tokens (,.!?\"'()[]{}-+=*/\\)")
    print("2. Number tokens (0-9)")
    print("3. Special characters and whitespace (newlines, tabs, etc.)")
    print("4. CLIP special tokens (start, end, padding)")
    print("\nThis focuses the semantic features on meaningful words and improves")
    print("the contrastive pairing between column names and token distributions.")
    print("=" * 80)

if __name__ == "__main__":
    visualize_token_filtering()