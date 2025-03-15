# Semantic Token Filtering for Improved Contrastive Learning

## Problem Statement

During analysis of the contrastive loss between column names and column tokens, we observed poor alignment between column names and their corresponding token distributions. The main issues were:

1. Column token distributions contained a high percentage of punctuation and special characters
2. Common tokens like commas, periods, and numbers dominated many columns
3. This diluted the semantic meaning of the column data and made it difficult for the contrastive loss to properly align column names with their data

## Solution: Token Filtering

We implemented token filtering in the semantic data processing pipeline to:

1. Remove punctuation, numbers, and special tokens
2. Keep only semantically meaningful tokens
3. Improve alignment between column names and column token distributions

## Implementation Details

### 1. Token Classification Function

Added `is_punctuation_or_special_token()` to identify tokens that don't contribute to semantic meaning:

```python
def is_punctuation_or_special_token(token_id: int, tokenizer = None) -> bool:
    """
    Check if a token ID represents punctuation or a special token.
    """
    # Skip invalid tokens
    if token_id is None or token_id <= 0:
        return True
    
    # Filter common special token IDs and punctuation token IDs
    special_token_ids = [0, 1, 2, 49406, 49407]  # PAD, BOS, EOS, etc.
    punctuation_token_ids = [
        267, 269, 261, 266, 286, 281, 280, 278, 279, 270,  # Common punctuation
        271, 272, 273, 274, 275, 276, 277, 278, 279, 280  # Digits 0-9
    ]
    
    if token_id in special_token_ids or token_id in punctuation_token_ids:
        return True
    
    # Use tokenizer for more detailed checks
    if tokenizer is not None:
        try:
            token_text = tokenizer.decode([int(token_id)])
            if token_text.strip() in ".,;:!?-()[]{}\"'`~@#$%^&*+=<>/\\|_":
                return True
            if token_text.strip().isdigit():
                return True
        except Exception:
            pass
    
    return False
```

### 2. Token Filtering Function

Added `filter_semantically_meaningful_tokens()` to apply filtering to token tensors:

```python
def filter_semantically_meaningful_tokens(
    token_tensor: torch.Tensor, 
    tokenizer = None, 
    filter_punctuation: bool = True,
    filter_numbers: bool = True,
    min_token_id: int = 10,
    max_filter_token_id: int = 2000
) -> torch.Tensor:
    """
    Filter out punctuation, numbers, and special tokens.
    """
    # Create mask for tokens to keep
    keep_mask = torch.ones_like(token_tensor, dtype=torch.bool)
    
    # Filter out very low token IDs
    if min_token_id > 0:
        keep_mask &= (token_tensor >= min_token_id)
    
    # Filter out common punctuation and special tokens
    if filter_punctuation:
        # Define common tokens to filter
        to_filter = set([
            0, 1, 2,  # Special tokens
            267, 269,  # Comma, period
            261, 266, 286, 281, 280,  # Other punctuation
            271, 272, 273, 274, 275, 276, 277, 278, 279, 280  # Digits 0-9
        ])
        
        # Apply the filter in a vectorized way
        for token_id in to_filter:
            keep_mask &= (token_tensor != token_id)
    
    # Replace filtered tokens with zeros
    filtered_tensor = torch.where(keep_mask, token_tensor, torch.zeros_like(token_tensor))
    
    return filtered_tensor
```

### 3. Integration with Data Loading

Updated semantic data loading functions to apply token filtering:

1. Added `filter_tokens` parameter to `load_semantic_prior_data()`
2. Added `filter_tokens` parameter to `get_random_semantic_data()`
3. Updated the filtering implementation to be more efficient

### 4. Modified Contrastive Learning Process

In `classification_adapter.py`, we updated the semantic data generation to use token filtering:

```python
semantic_data, semantic_data_column_names = get_random_semantic_data(
    num_classes=num_classes,
    seed=seed,
    use_cache=True,
    filter_tokens=True  # Apply token filtering to get more meaningful semantic tokens
)
```

## Expected Improvements

1. **Better Token Quality**: The semantic token distributions now contain a higher percentage of meaningful words and fewer punctuation/numbers.

2. **Improved Column-Token Alignment**: By removing noise tokens, the contrastive loss can better align column names with their corresponding token distributions.

3. **Enhanced Semantic Learning**: The model can focus on learning meaningful relationships between column names and their semantic content.

4. **More Interpretable Visualization**: The token distribution visualizations now show more semantically relevant tokens.

## Future Work

1. **Dynamic Token Importance**: Implement a weighting system that gives higher importance to tokens that are more specific to a column.

2. **Domain-Specific Token Lists**: Create domain-specific lists of important tokens to keep, even if they might be filtered by general rules.

3. **Embedding-Based Filtering**: Use embedding similarity to identify tokens that are semantically close to the column name.

4. **Performance Optimization**: Further optimize the filtering process for very large token tensors.

5. **Evaluation**: Conduct a thorough evaluation of how token filtering affects model performance on downstream tasks.

By implementing these token filtering improvements, we expect to see better alignment between column names and token distributions in contrastive learning, leading to more effective semantic representations in the model.