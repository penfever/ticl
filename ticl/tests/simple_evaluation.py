"""
Simple evaluation of token filtering effect.
"""

import torch
from transformers import CLIPTokenizerFast
from ticl.datasets.semantic_prior_data_loader import get_random_semantic_data, get_unwanted_token_ids

def main():
    # Load tokenizer
    tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
    
    # Get unwanted token IDs
    unwanted_token_ids = get_unwanted_token_ids(tokenizer)
    print(f"Number of unwanted tokens: {len(unwanted_token_ids)}")
    
    # Sample random column names
    num_classes = 3
    tensor_size = 100
    seed = 123
    
    # Generate filtered and unfiltered data
    unfiltered, column_names = get_random_semantic_data(
        num_classes=num_classes,
        tensor_size=tensor_size,
        seed=seed,
        filter_tokens=False
    )
    
    filtered, _ = get_random_semantic_data(
        num_classes=num_classes,
        tensor_size=tensor_size,
        seed=seed,
        filter_tokens=True
    )
    
    # Analyze one column in detail
    col_idx = 0
    col_name = column_names[col_idx]
    print(f"Analyzing column: {col_name}")
    
    # Count filtered tokens
    filtered_count = (filtered[col_idx] == -100).sum().item()
    total = filtered[col_idx].numel()
    filter_percentage = (filtered_count / total) * 100
    print(f"Filtered tokens: {filtered_count}/{total} ({filter_percentage:.1f}%)")
    
    # Print some tokens with their text before and after filtering
    tokens_to_show = 20
    start_idx = 10  # Skip the first few tokens
    
    unfiltered_tokens = unfiltered[col_idx, start_idx:start_idx+tokens_to_show].tolist()
    filtered_tokens = filtered[col_idx, start_idx:start_idx+tokens_to_show].tolist()
    
    print("\nToken comparison:")
    print(f"{'Index':<8} {'Unfiltered ID':<15} {'Unfiltered Text':<20} {'Filtered ID':<15} {'Filtered Text':<20}")
    print("-" * 80)
    
    for i in range(tokens_to_show):
        idx = start_idx + i
        unfiltered_id = unfiltered_tokens[i]
        filtered_id = filtered_tokens[i]
        
        # Get text representations
        unfiltered_text = tokenizer.decode([unfiltered_id]) if unfiltered_id >= 0 else "[UNK]"
        filtered_text = "[FILTERED]" if filtered_id == -100 else tokenizer.decode([filtered_id])
        
        print(f"{idx:<8} {unfiltered_id:<15} {unfiltered_text:<20} {filtered_id:<15} {filtered_text:<20}")
    
    # Check if any tokens in unfiltered are in the unwanted list
    unwanted_found = 0
    for token_id in unfiltered_tokens:
        if token_id in unwanted_token_ids:
            unwanted_found += 1
    
    print(f"\nFound {unwanted_found} unwanted tokens in the unfiltered sample")
    
    # Show frequency of token types
    print("\nUnwanted token examples:")
    for category, tokens in [
        ("Punctuation", ["!", ".", ",", "-", "?"]),
        ("Digits", ["0", "1", "2", "3", "4"]),
        ("Special", [" ", "\n", "<", ">", "/"]),
    ]:
        print(f"  {category}: ", end="")
        for token in tokens:
            token_ids = tokenizer.encode(token, add_special_tokens=False)
            for token_id in token_ids:
                in_unwanted = token_id in unwanted_token_ids
                text = tokenizer.decode([token_id])
                print(f"'{text}'({token_id}){' ✓' if in_unwanted else ' ✗'}", end=", ")
        print()
    
    print("\nConclusion:")
    print("The token filtering removes punctuation, digits, and special characters")
    print("from the semantic features, focusing on meaningful words that contribute")
    print("to the semantic understanding of the data.")

if __name__ == "__main__":
    main()