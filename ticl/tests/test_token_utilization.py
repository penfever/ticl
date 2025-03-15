"""
Test if the changes to token utilization improve information throughput.
"""

from ticl.datasets.semantic_prior_data_loader import get_random_semantic_data
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_semantic_data_utilization():
    """Test how token filtering and utilization affect semantic data."""
    print("=" * 80)
    print("TESTING SEMANTIC DATA INFORMATION UTILIZATION")
    print("=" * 80)
    
    # Generate semantic data
    num_classes = 20
    tensor_size = 200
    seed = 42
    
    print(f"Generating semantic data with {num_classes} classes and {tensor_size} tokens per class")
    semantic_data, column_names = get_random_semantic_data(
        num_classes=num_classes,
        tensor_size=tensor_size,
        seed=seed,
        filter_tokens=True
    )
    
    # Analyze token usage
    # Count non-zero tokens (tokens != 0 and tokens != -100)
    non_zero_tokens = ((semantic_data != 0) & (semantic_data != -100)).sum().item()
    
    # Count filtered tokens (tokens == -100)
    filtered_tokens = (semantic_data == -100).sum().item()
    
    # Count total tokens
    total_tokens = semantic_data.numel()
    
    # Calculate utilization percentages
    non_zero_percentage = (non_zero_tokens / total_tokens) * 100
    filtered_percentage = (filtered_tokens / total_tokens) * 100
    
    print(f"Semantic data tensor shape: {semantic_data.shape}")
    print(f"Valid tokens: {non_zero_tokens} / {total_tokens} ({non_zero_percentage:.2f}%)")
    print(f"Filtered tokens: {filtered_tokens} / {total_tokens} ({filtered_percentage:.2f}%)")
    
    # Count unique non-zero tokens
    unique_tokens = torch.unique(semantic_data)
    unique_valid = torch.unique(semantic_data[(semantic_data != 0) & (semantic_data != -100)])
    
    print(f"Unique token count (including 0 and -100): {len(unique_tokens)}")
    print(f"Unique valid token count: {len(unique_valid)}")
    
    # Examine distribution of tokens across classes
    for i, col_name in enumerate(column_names):
        class_data = semantic_data[i]
        valid_tokens_in_class = ((class_data != 0) & (class_data != -100)).sum().item()
        filtered_tokens_in_class = (class_data == -100).sum().item()
        
        valid_percentage = (valid_tokens_in_class / class_data.numel()) * 100
        filtered_percentage = (filtered_tokens_in_class / class_data.numel()) * 100
        
        unique_tokens_in_class = torch.unique(class_data[(class_data != 0) & (class_data != -100)])
        
        print(f"\nClass {i}: {col_name}")
        print(f"  Valid tokens: {valid_tokens_in_class} / {class_data.numel()} ({valid_percentage:.2f}%)")
        print(f"  Filtered tokens: {filtered_tokens_in_class} / {class_data.numel()} ({filtered_percentage:.2f}%)")
        print(f"  Unique valid tokens: {len(unique_tokens_in_class)}")
    
    print("\nConclusion:")
    if len(unique_valid) > 100:
        print("✅ GOOD: Semantic data contains a diverse set of tokens (>100 unique tokens)")
    elif len(unique_valid) > 50:
        print("⚠️ MODERATE: Semantic data contains a reasonable number of tokens (>50 unique tokens)")
    else:
        print("❌ POOR: Semantic data contains few unique tokens (<50)")
        
    if non_zero_percentage > 50:
        print("✅ GOOD: Semantic data has high utilization (>50% valid tokens)")
    elif non_zero_percentage > 30:
        print("⚠️ MODERATE: Semantic data has reasonable utilization (>30% valid tokens)")
    else:
        print("❌ POOR: Semantic data has low utilization (<30% valid tokens)")
    
    print("=" * 80)

if __name__ == "__main__":
    test_semantic_data_utilization()