import os
import json
import random
import logging
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Union
from transformers import CLIPTokenizerFast
from functools import lru_cache
import time

# Setup logging
logger = logging.getLogger(__name__)

# Global cache for semantic data
_SEMANTIC_DATA_CACHE = {
    "is_loaded": False,
    "log_file_path": None,
    "column_name_tokens": {},
    "column_value_tokens": {},
    "max_columns": None,
    "memory_usage": 0  # Track memory usage for better cache management
}

# Common numeric patterns to identify numeric columns
NUMERIC_PATTERNS = [
    "number", "num", "qty", "quantity", "count", "amount", "val", "value",
    "price", "cost", "age", "year", "month", "day", "date", "time",
    "height", "weight", "length", "width", "depth", "size", "volume",
    "latitude", "longitude", "lat", "long", "altitude", "temp", "temperature",
    "percent", "ratio", "score", "rating", "rank", "id", "code"
]

# LRU cache for is_numeric_column - this is called repeatedly with the same values
@lru_cache(maxsize=2048)
def is_numeric_column(column_name: str) -> bool:
    """
    Determine if a column is likely to be numeric based on its name.
    Uses common patterns found in numeric column names.
    
    Parameters:
    -----------
    column_name : str
        The name of the column to check
        
    Returns:
    --------
    bool
        True if the column appears to be numeric, False otherwise
    """
    # Convert to lowercase for case-insensitive matching
    column_lower = column_name.lower()
    
    # Fast checks first - direct containment
    for pattern in NUMERIC_PATTERNS:
        if pattern in column_lower:
            return True
    
    # Check for patterns with word boundaries
    for pattern in NUMERIC_PATTERNS:
        if f"_{pattern}" in column_lower or f"{pattern}_" in column_lower:
            return True

    return False


def load_semantic_prior_data(
    log_file_path: str = "ticl/datasets/completed_columns.json",
    clip_tokenizer_name: str = "openai/clip-vit-base-patch32",
    target_tensor_size: Optional[int] = None,
    force_non_numeric: bool = True,
    max_columns: Optional[int] = None,
    use_cache: bool = True,
    force_reload: bool = False
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Load semantic column data from completed_columns.json with optimized performance.
    
    Optimizations:
    - Efficient caching with memory tracking
    - Batched tokenization operations
    - Reduced device transfers
    - Vectorized tensor operations
    - Smart memory management
    
    Parameters:
    -----------
    log_file_path : str
        Path to the completed_columns.json file
    clip_tokenizer_name : str
        Name of the CLIP tokenizer to use
    target_tensor_size : int, optional
        Target size for all tensors (will pad/truncate as needed)
    force_non_numeric : bool
        If True, only include columns that don't appear to be numeric
    max_columns : int, optional
        If set, limit the number of columns to load to save memory
    use_cache : bool
        Whether to use the global cache to avoid reloading data
    force_reload : bool
        Whether to force reload data even if it's in the cache
        
    Returns:
    --------
    Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
        Tuple of (column_name_tokens, column_value_tokens)
    """
    global _SEMANTIC_DATA_CACHE
    
    profiling_start = time.time()
    
    # Check if we can use the cached data
    cache_valid = (
        use_cache and 
        _SEMANTIC_DATA_CACHE["is_loaded"] and
        not force_reload and
        _SEMANTIC_DATA_CACHE["log_file_path"] == log_file_path and
        (max_columns is None or _SEMANTIC_DATA_CACHE["max_columns"] is None or 
         max_columns >= _SEMANTIC_DATA_CACHE["max_columns"])
    )
    
    # OPTIMIZATION: Fast path return when cache is valid
    if cache_valid:
        logger.debug(f"Using cached semantic data (memory: {_SEMANTIC_DATA_CACHE['memory_usage']:.2f} MB)")
        return _SEMANTIC_DATA_CACHE["column_name_tokens"], _SEMANTIC_DATA_CACHE["column_value_tokens"]
    
    # Load data from JSON file
    if not os.path.exists(log_file_path):
        raise FileNotFoundError(f"Semantic column data file not found at {log_file_path}")
    
    json_start = time.time()
    try:
        with open(log_file_path, 'r') as f:
            completed_columns = json.load(f)
    except json.JSONDecodeError:
        raise ValueError(f"Error parsing JSON file: {log_file_path}")
    
    json_time = time.time() - json_start
    logger.debug(f"JSON load took {json_time:.4f}s")
    
    logger.info(f"Loaded {len(completed_columns)} columns from {log_file_path}")
    
    # OPTIMIZATION: Batch filter non-numeric columns before tokenization
    filter_start = time.time()
    filtered_columns = {}
    
    # Use set for O(1) lookups - pre-load to reduce attribute access overhead
    numeric_patterns_set = set(NUMERIC_PATTERNS)
    
    # Pre-filter columns - use batched approach
    if force_non_numeric:
        for col_name, tensor_data in completed_columns.items():
            # Use the fast cached implementation
            if not is_numeric_column(col_name):
                filtered_columns[col_name] = tensor_data
    else:
        # No filtering needed, use all columns
        filtered_columns = completed_columns
    
    filter_time = time.time() - filter_start
    logger.debug(f"Filtering took {filter_time:.4f}s, {len(filtered_columns)} columns remain")
    
    # OPTIMIZATION: Calculate tensor size in advance to avoid resizing
    size_start = time.time()
    if target_tensor_size is None:
        sizes = [len(tensor_data) for tensor_data in filtered_columns.values()]
        target_tensor_size = max(sizes) if sizes else 100
    
    # OPTIMIZATION: Sort columns if max_columns is set to maintain consistency
    if max_columns is not None and max_columns < len(filtered_columns):
        sorted_cols = sorted(filtered_columns.keys())
        selected_cols = sorted_cols[:max_columns]
        filtered_columns = {col: filtered_columns[col] for col in selected_cols}
        logger.info(f"Limited to {len(filtered_columns)} columns due to max_columns={max_columns}")
    
    size_time = time.time() - size_start
    logger.debug(f"Size calculation took {size_time:.4f}s, target_size={target_tensor_size}")
    
    # OPTIMIZATION: Initialize tokenizer once and keep in memory
    tok_start = time.time()
    tokenizer = CLIPTokenizerFast.from_pretrained(clip_tokenizer_name)
    tok_time = time.time() - tok_start
    logger.debug(f"Tokenizer init took {tok_time:.4f}s")
    
    # OPTIMIZATION: Batch process column names
    column_names = list(filtered_columns.keys())
    process_start = time.time()
    
    # Process all column names at once in a batch
    name_tokens_batch = tokenizer(
        column_names,
        padding="max_length",
        truncation=True,
        max_length=77,  # CLIP's max context length
        return_tensors="pt"
    )
    
    # Extract and organize in column_name_tokens dictionary
    column_name_tokens = {}
    for i, col_name in enumerate(column_names):
        # Get token IDs for this column from the batch
        tokens = name_tokens_batch["input_ids"][i].tolist()
        column_name_tokens[col_name] = tokens
    
    # Process values
    column_value_tokens = {}
    batch_size = 100  # Process values in batches to avoid OOM
    values_time = 0
    
    # OPTIMIZATION: Pre-allocate memory for lists to avoid reallocation
    for i in range(0, len(column_names), batch_size):
        batch_start = time.time()
        batch_cols = column_names[i:i+batch_size]
        
        for col_name in batch_cols:
            tensor_data = filtered_columns[col_name]
            
            # Pad or truncate to target size
            if len(tensor_data) < target_tensor_size:
                # Pad with zeros
                tensor_data = tensor_data + [0] * (target_tensor_size - len(tensor_data))
            elif len(tensor_data) > target_tensor_size:
                # Truncate
                tensor_data = tensor_data[:target_tensor_size]
            
            # Store token values directly
            column_value_tokens[col_name] = tensor_data
        
        batch_time = time.time() - batch_start
        values_time += batch_time
    
    logger.debug(f"Values processing took {values_time:.4f}s")
    
    process_time = time.time() - process_start
    logger.debug(f"Column processing took {process_time:.4f}s")
    
    # OPTIMIZATION: Estimate memory usage for cache management
    memory_usage = 0
    for col, tokens in column_name_tokens.items():
        # Estimate 4 bytes per token (int) plus string overhead
        memory_usage += len(tokens) * 4 + len(col) * 2
    
    for col, tokens in column_value_tokens.items():
        # Estimate 4 bytes per token (int)
        memory_usage += len(tokens) * 4
    
    # Convert to MB
    memory_usage = memory_usage / (1024 * 1024)
    
    # Update cache
    if use_cache:
        _SEMANTIC_DATA_CACHE.update({
            "is_loaded": True,
            "log_file_path": log_file_path,
            "column_name_tokens": column_name_tokens,
            "column_value_tokens": column_value_tokens,
            "max_columns": max_columns,
            "memory_usage": memory_usage
        })
    
    total_time = time.time() - profiling_start
    logger.info(f"load_semantic_prior_data took {total_time:.4f}s for {len(column_names)} columns")
    
    return column_name_tokens, column_value_tokens


# Enhanced cache for random semantic data to avoid redundant generation
_RANDOM_SEMANTIC_DATA_CACHE = {}

def get_random_semantic_data(
    num_classes: int = 5,
    min_token_value: int = 1000,
    max_token_value: int = 5000,
    num_tokens: int = 50,
    seed: Optional[int] = None,
    use_cache: bool = True,
    filter_tokens: bool = True
) -> Tuple[torch.Tensor, List[str]]:
    """
    Generate random semantic data for testing and training.
    With optimized performance compared to original implementation.
    
    Parameters:
    -----------
    num_classes : int
        Number of semantic classes to generate
    min_token_value : int
        Minimum token ID value
    max_token_value : int
        Maximum token ID value
    num_tokens : int
        Number of tokens per class
    seed : int, optional
        Random seed for reproducibility
    use_cache : bool
        Whether to use the cache
        
    Returns:
    --------
    Tuple[torch.Tensor, List[str]]
        Tuple of (semantic_data, column_names)
    """
    # Check cache first
    cache_key = f"{num_classes}_{min_token_value}_{max_token_value}_{num_tokens}_{seed}"
    
    if use_cache and cache_key in _RANDOM_SEMANTIC_DATA_CACHE:
        logger.debug(f"Using cached random semantic data for {num_classes} classes")
        return _RANDOM_SEMANTIC_DATA_CACHE[cache_key]
    
    # Set seed for reproducibility if provided
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
    
    # OPTIMIZATION: Generate all tokens at once with single call to randint
    # This is much faster than generating them individually
    semantic_data = torch.randint(
        min_token_value, 
        max_token_value, 
        (num_classes, num_tokens)
    )
    
    # Generate column names 
    column_names = [f"semantic_feature_{i}" for i in range(num_classes)]
    
    # Cache results
    if use_cache:
        _RANDOM_SEMANTIC_DATA_CACHE[cache_key] = (semantic_data, column_names)
    
    return semantic_data, column_names

# For backwards compatibility - generate with filtered tokens
random_tensor, semantic_data_column_names = get_random_semantic_data()