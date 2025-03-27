"""
Semantic prior data loader for TiCL models.

This module loads and processes non-numeric semantic column data from tokenized semantic features.
"""

import torch
import os
import json
from typing import Dict, Tuple, Optional, List, Set
import logging
import random
import numpy as np
import string
from transformers import CLIPTokenizerFast

logger = logging.getLogger(__name__)

# Global cache for loaded data to avoid repeated JSON loading
# IMPORTANT: We limit the cache size to prevent memory leaks
_SEMANTIC_DATA_CACHE = {
    "column_name_tokens": None,
    "column_value_tokens": None,
    "is_loaded": False,
    "log_file_path": None,
    "max_columns": None,
    "unwanted_token_ids": None,
    "tokenizer": None,
    # Track cache size in bytes for memory management 
    "cache_size_bytes": 0,
    # Limit cache size to prevent OOM issues - 500MB should be enough for training
    "max_cache_size_bytes": 500 * 1024 * 1024  # 500MB
}

def is_float(value: str) -> bool:
    """Check if a string can be converted to a float."""
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False

def is_int(value: str) -> bool:
    """Check if a string can be converted to an integer."""
    try:
        int(value)
        return True
    except (ValueError, TypeError):
        return False

def get_unwanted_token_ids(tokenizer: CLIPTokenizerFast) -> torch.Tensor:
    """
    Get or create a list of token IDs for tokens we want to filter out.
    This includes punctuation, numbers, and special tokens.
    
    Uses the global cache to avoid recomputing the list on every call.
    
    Parameters:
    -----------
    tokenizer : CLIPTokenizerFast
        The CLIP tokenizer
        
    Returns:
    --------
    torch.Tensor
        Tensor of token IDs to filter out (replace with -100)
    """
    global _SEMANTIC_DATA_CACHE
    
    # Check if we've already cached the unwanted token IDs
    if (_SEMANTIC_DATA_CACHE["unwanted_token_ids"] is not None and 
        _SEMANTIC_DATA_CACHE["tokenizer"] == tokenizer.name_or_path):
        return _SEMANTIC_DATA_CACHE["unwanted_token_ids"]
    
    # Define unwanted token categories
    
    # 1. Punctuation
    punctuation_tokens = []
    for p in string.punctuation:
        # Tokenize each punctuation character
        token_ids = tokenizer.encode(p, add_special_tokens=False)
        punctuation_tokens.extend(token_ids)
    
    # 2. Numbers
    number_tokens = []
    for i in range(10):
        # Tokenize digits 0-9
        token_ids = tokenizer.encode(str(i), add_special_tokens=False)
        number_tokens.extend(token_ids)
    
    # 3. Special characters and whitespace
    special_tokens = []
    for s in [' ', '\n', '\t', '\r', '<', '>', '/']:
        token_ids = tokenizer.encode(s, add_special_tokens=False)
        special_tokens.extend(token_ids)
    
    # 4. CLIP special tokens
    clip_special = []
    for token in ['<|startoftext|>', '<|endoftext|>', '<|pad|>']:
        if token in tokenizer.vocab:
            clip_special.append(tokenizer.vocab[token])
    
    # Combine all unwanted tokens
    all_unwanted = set(punctuation_tokens + number_tokens + special_tokens + clip_special)
    
    # Convert to tensor
    unwanted_tensor = torch.tensor(list(all_unwanted), dtype=torch.int32)
    
    # Add to cache
    _SEMANTIC_DATA_CACHE["unwanted_token_ids"] = unwanted_tensor
    _SEMANTIC_DATA_CACHE["tokenizer"] = tokenizer.name_or_path
    
    logger.info(f"Created unwanted token list with {len(unwanted_tensor)} tokens")
    
    return unwanted_tensor


def is_numeric_column(column_name: str) -> bool:
    """
    Determine if a column is likely to contain numeric data based on its name.
    
    This is a heuristic check that looks for common numeric column name patterns.
    Uses word boundaries to avoid false positives like 'phone' matching 'one'.
    """
    # Exact full column name matches
    exact_matches = [
        'count', 'amount', 'price', 'rate', 'age', 'year', 'month', 'day',
        'weight', 'height', 'size', 'length', 'width', 'depth', 'date',
        'time', 'score', 'rating', 'rank', 'id', 'index', 'code', 'zip',
        'temperature', 'distance', 'speed', 'volume', 'area', 'mass'
    ]
    
    # Word parts that should match with word boundaries
    # These will match when they appear as whole words or at word boundaries
    word_parts = [
        'count_', '_count', 'counter', 'counting',
        '_id', 'id_', '_code', 'code_',
        '_num', 'num_', '_qty', 'qty_',
        '_amt', 'amt_', '_val', 'val_',
        'price_', '_price', 'cost_', '_cost',
        'year_', '_year', 'date_', '_date',
        'time_', '_time', 'duration_', '_duration',
        'percent_', '_percent', 'ratio_', '_ratio',
        'score_', '_score', 'rating_', '_rating',
        'age_', '_age', 'weight_', '_weight',
        'height_', '_height', 'size_', '_size',
        'length_', '_length', 'width_', '_width',
        'depth_', '_depth'
    ]
    
    # Standard column name patterns that typically contain numeric data
    # These should only match when they're part of a compound word
    numeric_patterns = [
        'amount', 'quantity', 'number', 'numeric',
        'population', 'frequency', 'velocity',
        'acceleration', 'density',
        'pressure', 'energy', 'power', 'force', 
        'voltage', 'current', 'resistance',
        'conductance', 'capacitance', 'inductance',
        'flux', 'intensity', 'luminance', 'exposure', 
        'concentration', 'humidity', 'elevation', 
        'altitude', 'latitude', 'longitude',
        'grade', 'gpa', 'salary', 'income', 'revenue', 
        'expense', 'profit', 'loss', 'debt', 'asset', 
        'liability', 'equity', 'margin', 'timestamp', 'epoch'
    ]
    
    # List of known non-numeric columns that might trigger false positives
    non_numeric_columns = [
        'phone', 'telephone', 'email', 'address', 'name',
        'city', 'country', 'state', 'province', 'description',
        'title', 'url', 'link', 'category', 'tag'
    ]
    
    # Convert column name to lowercase for case-insensitive matching
    column_lower = column_name.lower()
    
    # Check if column is in the non-numeric list
    if column_lower in non_numeric_columns:
        return False
    
    # Check for exact full column name matches
    if column_lower in exact_matches:
        return True
    
    # Check for word boundary matches
    for word_part in word_parts:
        if word_part in column_lower:
            return True
    
    # Split into words and check for whole-word matches
    words = column_lower.replace('_', ' ').replace('-', ' ').split()
    for word in words:
        if word in numeric_patterns:
            return True
    
    # For compound words, check if any numeric pattern is a substring
    # This is more permissive but avoids matching substrings within words
    if '_' in column_lower:
        for pattern in numeric_patterns:
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
    Load semantic column data from completed_columns.json.
    
    This function loads tokenized column data, filtering out numeric columns,
    and returns two dictionaries mapping column names to:
    1. Column name tokenizations
    2. Column value tokenizations
    
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
    
    # Check if data is already in cache and we can use it
    cache_valid = (
        use_cache and 
        _SEMANTIC_DATA_CACHE["is_loaded"] and
        not force_reload and
        _SEMANTIC_DATA_CACHE["log_file_path"] == log_file_path and
        (max_columns is None or _SEMANTIC_DATA_CACHE["max_columns"] is None or 
         max_columns >= _SEMANTIC_DATA_CACHE["max_columns"])
    )
    
    if cache_valid:
        return _SEMANTIC_DATA_CACHE["column_name_tokens"], _SEMANTIC_DATA_CACHE["column_value_tokens"]
    
    if not os.path.exists(log_file_path):
        raise FileNotFoundError(f"Semantic column data file not found at {log_file_path}")
    
    # Load data from JSON file
    try:
        with open(log_file_path, 'r') as f:
            completed_columns = json.load(f)
    except json.JSONDecodeError:
        raise ValueError(f"Error parsing JSON file: {log_file_path}")
    
    logger.info(f"Loaded {len(completed_columns)} columns from {log_file_path}")
    
    # Initialize the CLIP tokenizer for column names
    tokenizer = CLIPTokenizerFast.from_pretrained(clip_tokenizer_name)
    
    # Prepare column name and value dictionaries
    column_name_tokens = {}
    column_value_tokens = {}
    
    # Determine maximum tensor size if not specified
    if target_tensor_size is None:
        sizes = [len(tensor_data) for tensor_data in completed_columns.values()]
        target_tensor_size = max(sizes) if sizes else 100
        logger.info(f"Using detected maximum tensor size: {target_tensor_size}")
    
    # Filter non-numeric columns first
    filtered_columns = {}
    
    for col_name, tensor_data in completed_columns.items():
        is_numeric = is_numeric_column(col_name)
        if not force_non_numeric or not is_numeric:
            filtered_columns[col_name] = tensor_data
    
    # If max_columns is set, limit the number of columns to process
    if max_columns is not None and max_columns < len(filtered_columns):
        sorted_cols = sorted(filtered_columns.keys())
        selected_cols = sorted_cols[:max_columns]
        filtered_columns = {col: filtered_columns[col] for col in selected_cols}
        logger.info(f"Limited to {len(filtered_columns)} columns due to max_columns={max_columns}")
    
    # Get or create the set of unwanted token IDs
    unwanted_token_ids = get_unwanted_token_ids(tokenizer)
    
    # Process each column
    for col_name, tensor_data in filtered_columns.items():
        # Convert tensor data to tensor
        col_tensor = torch.tensor(tensor_data, dtype=torch.int32)  # Use int32 to save memory
        
        # Filter out unwanted tokens with highly optimized vectorized approach
        # First create a boolean tensor to mark which tokens should be kept
        # This avoids the slow loop over all unwanted tokens
        
        # We'll create a lookup table where indices are token IDs and values are 1 for keep or 0 for filter
        # Find the maximum token ID to determine lookup table size
        max_token_id = max(col_tensor.max().item(), unwanted_token_ids.max().item()) + 1
        
        # Create lookup tensor (1 = keep, 0 = filter)
        # Default is to keep all tokens (1)
        token_filter = torch.ones(max_token_id, dtype=torch.int8)
        
        # Mark unwanted tokens as 0 (filter)
        token_filter[unwanted_token_ids] = 0
        
        # Use this lookup to create a mask where 0 = replace with -100, 1 = keep original
        # We handle negative indices separately since they can't be used in lookup table
        valid_indices = (col_tensor >= 0)
        mask = torch.ones_like(col_tensor, dtype=torch.int8)
        
        # Only apply lookup to non-negative indices
        mask[valid_indices] = token_filter[col_tensor[valid_indices]]
        
        # Apply the filter: where mask is 0, use -100, otherwise use original value
        col_tensor = torch.where(mask.bool(), col_tensor, torch.tensor(-100, dtype=col_tensor.dtype))
        
        # Ensure consistent tensor size
        if col_tensor.size(0) < target_tensor_size:
            # Pad with zeros if needed
            padding = torch.zeros(target_tensor_size - col_tensor.size(0), dtype=col_tensor.dtype)
            col_tensor = torch.cat([col_tensor, padding])
        elif col_tensor.size(0) > target_tensor_size:
            # Truncate if larger
            col_tensor = col_tensor[:target_tensor_size]
        
        # Process column name for tokens dictionary
        # First tokenize the column name
        col_name_clean = col_name.replace("_", " ").replace("-", " ")
        
        tokenizer_output = tokenizer(
            col_name_clean,
            return_tensors="pt",
            padding="max_length",
            max_length=50,  # Using a standard size for column names
            truncation=True
        )
        
        # Handle both dictionary output and object with attributes
        if isinstance(tokenizer_output, dict):
            col_name_tokens = tokenizer_output['input_ids'][0]
        else:
            col_name_tokens = tokenizer_output.input_ids[0]
        
        # Store in dictionaries
        column_name_tokens[col_name] = col_name_tokens
        column_value_tokens[col_name] = col_tensor
    
    logger.info(f"Loaded {len(column_name_tokens)} non-numeric columns with target size {target_tensor_size}")
    
    # Update the global cache
    if use_cache:
        _SEMANTIC_DATA_CACHE["column_name_tokens"] = column_name_tokens
        _SEMANTIC_DATA_CACHE["column_value_tokens"] = column_value_tokens
        _SEMANTIC_DATA_CACHE["is_loaded"] = True
        _SEMANTIC_DATA_CACHE["log_file_path"] = log_file_path
        _SEMANTIC_DATA_CACHE["max_columns"] = max_columns
    
    return column_name_tokens, column_value_tokens

def get_random_semantic_data(
    num_classes: int = 3,
    num_tokens: int = 50,
    tensor_size: int = 200,
    log_file_path: str = "ticl/datasets/completed_columns.json",
    max_tensor_size_mb: int = 10,  # Limit tensor size to 10MB by default
    return_column_names: bool = True,
    seed: Optional[int] = None,  # Random seed for reproducibility
    use_cache: bool = True,  # Whether to use cached data
    filter_tokens: bool = True  # Whether to filter unwanted tokens
) -> tuple:
    """
    Get random semantic data either from real columns or generate synthetic data.
    
    This is a compatibility function for code that expects semantic_prior_data_sample.random_tensor.
    
    Parameters:
    -----------
    num_classes : int
        Number of semantic classes
    num_tokens : int
        Number of tokens per class
    tensor_size : int
        Size of each tensor to return
    log_file_path : str
        Path to the completed_columns.json file
    max_tensor_size_mb : int
        Maximum size of the returned tensor in MB to avoid OOM issues
    return_column_names : bool
        Whether to return column names along with the tensor
    seed : int, optional
        Random seed for column selection and shuffling
    use_cache : bool
        Whether to use cached data or reload from file
        
    Returns:
    --------
    tuple
        If return_column_names is True:
          (semantic_data, column_names) where:
            - semantic_data: Tensor of shape [num_classes, tensor_size]
            - column_names: List of column names corresponding to each row in the tensor
        If return_column_names is False:
          just the semantic_data tensor
    """
    global _SEMANTIC_DATA_CACHE
    
    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
    
    # Calculate max size that would keep the whole tensor < max_tensor_size_mb
    max_allowed_size = int((max_tensor_size_mb * 1024 * 1024) / (num_classes * 4))  # 4 bytes per int32
    tensor_size = min(tensor_size, max_allowed_size)
    
    # First check if we have cached tensor of the right size
    cached_tensor_key = f"semantic_tensor_{num_classes}_{tensor_size}"
    cached_columns_key = f"semantic_columns_{num_classes}"
    
    if use_cache and cached_tensor_key in _SEMANTIC_DATA_CACHE and cached_columns_key in _SEMANTIC_DATA_CACHE:
        # We already have a cached tensor of the exact size and class count needed
        if return_column_names:
            return _SEMANTIC_DATA_CACHE[cached_tensor_key], _SEMANTIC_DATA_CACHE[cached_columns_key]
        else:
            return _SEMANTIC_DATA_CACHE[cached_tensor_key]
    
    try:
        # Try to load real data if file exists
        if os.path.exists(log_file_path):
            # Check if we have cached column data that we can use
            if _SEMANTIC_DATA_CACHE["is_loaded"] and use_cache and _SEMANTIC_DATA_CACHE["log_file_path"] == log_file_path:
                column_values = _SEMANTIC_DATA_CACHE["column_value_tokens"]
            else:
                # Load real data efficiently - only get the column values, not names
                _, column_values = load_semantic_prior_data(
                    log_file_path=log_file_path,
                    target_tensor_size=tensor_size,
                    force_non_numeric=True,
                    max_columns=None,
                    use_cache=use_cache
                )
            
            # If we have enough columns, sample from them
            if len(column_values) >= num_classes:
                # Select random columns efficiently
                columns = list(column_values.keys())
                
                # Create randomized indices for column selection
                if seed is not None:
                    # For reproducible selection
                    generator = torch.Generator()
                    generator.manual_seed(seed)
                    selected_indices = torch.randperm(len(columns), generator=generator)[:num_classes].tolist()
                else:
                    # Fresh random selection each time
                    selected_indices = torch.randperm(len(columns))[:num_classes].tolist()
                    
                selected_columns = [columns[i] for i in selected_indices]
                
                # Stack tensors efficiently
                tensors = [column_values[col] for col in selected_columns]
                semantic_tensor = torch.stack(tensors)
                
                # Force to int32 to save memory
                semantic_tensor = semantic_tensor.to(dtype=torch.int32)
                
                # Make sure the tensor stays on CPU
                if semantic_tensor.device.type != 'cpu':
                    semantic_tensor = semantic_tensor.to('cpu')
                
                # Cache the tensor for future use with this configuration
                if use_cache:
                    # Check tensor size before adding to cache
                    tensor_size_bytes = semantic_tensor.element_size() * semantic_tensor.nelement()
                    
                    # Check if adding this tensor would exceed cache size limit
                    if _SEMANTIC_DATA_CACHE["cache_size_bytes"] + tensor_size_bytes > _SEMANTIC_DATA_CACHE["max_cache_size_bytes"]:
                        # If cache is getting too big, clear it to prevent OOM
                        logger.warning(f"Semantic data cache limit reached ({_SEMANTIC_DATA_CACHE['cache_size_bytes']/1024/1024:.1f}MB), clearing cache")
                        # Clear all cached tensors except essential ones
                        keys_to_keep = ["is_loaded", "log_file_path", "max_columns", "unwanted_token_ids", "tokenizer", 
                                       "cache_size_bytes", "max_cache_size_bytes"]
                        keys_to_clear = [k for k in _SEMANTIC_DATA_CACHE.keys() if k not in keys_to_keep]
                        for k in keys_to_clear:
                            _SEMANTIC_DATA_CACHE[k] = None
                        _SEMANTIC_DATA_CACHE["cache_size_bytes"] = 0
                    
                    # Add to cache and update size tracking
                    _SEMANTIC_DATA_CACHE[cached_tensor_key] = semantic_tensor
                    _SEMANTIC_DATA_CACHE[cached_columns_key] = selected_columns
                    _SEMANTIC_DATA_CACHE["cache_size_bytes"] += tensor_size_bytes
                
                # Return both the tensor and column names
                if return_column_names:
                    return semantic_tensor, selected_columns
                else:
                    return semantic_tensor
    except Exception as e:
        # Only log at warning level if file exists but loading failed
        if os.path.exists(log_file_path):
            logger.warning(f"Could not load semantic data from file: {e}")
    
    # Check if we already generated synthetic data with these parameters
    if seed is not None and use_cache:
        synthetic_key = f"synthetic_{num_classes}_{tensor_size}_{seed}"
        if synthetic_key in _SEMANTIC_DATA_CACHE:
            # Return cached synthetic data
            if return_column_names:
                return _SEMANTIC_DATA_CACHE[synthetic_key], _SEMANTIC_DATA_CACHE[f"{synthetic_key}_columns"]
            else:
                return _SEMANTIC_DATA_CACHE[synthetic_key]
    
    # Fallback to synthetic data generation - optimize for speed
    # Direct generation without unnecessary operations
    if seed is not None:
        # Use generator for reproducible randomness
        generator = torch.Generator()
        generator.manual_seed(seed)
        # Generate directly in the right range to avoid extra multiplication
        random_tensor = torch.randint(1, 49405, (num_classes, tensor_size), 
                                      dtype=torch.int32, generator=generator)
    else:
        # Standard randomness, directly generate integers
        random_tensor = torch.randint(1, 49405, (num_classes, tensor_size), 
                                     dtype=torch.int32)
    
    # Filter unwanted tokens if requested
    if filter_tokens:
        # Initialize tokenizer if needed
        if _SEMANTIC_DATA_CACHE["tokenizer"] is None:
            tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
            _SEMANTIC_DATA_CACHE["tokenizer"] = tokenizer.name_or_path
        else:
            tokenizer = CLIPTokenizerFast.from_pretrained(_SEMANTIC_DATA_CACHE["tokenizer"])
        
        # Get unwanted token IDs (cached)
        unwanted_token_ids = get_unwanted_token_ids(tokenizer)
        
        # Apply efficient filtering with lookup table
        max_token_id = max(random_tensor.max().item(), unwanted_token_ids.max().item()) + 1
        token_filter = torch.ones(max_token_id, dtype=torch.int8)
        token_filter[unwanted_token_ids] = 0
        
        # Apply the filter using vectorized operations
        valid_indices = (random_tensor >= 0)
        mask = torch.ones_like(random_tensor, dtype=torch.int8)
        mask[valid_indices] = token_filter[random_tensor[valid_indices]]
        
        # Replace unwanted tokens with -100 (ignore index)
        random_tensor = torch.where(mask.bool(), random_tensor, torch.tensor(-100, dtype=random_tensor.dtype))
        
        logger.info(f"Filtered {(~mask.bool()).sum().item()} unwanted tokens from synthetic data")
    
    # Generate synthetic column names using minimal computation
    synthetic_columns = [f"synthetic_feature_{i}" for i in range(num_classes)]
    
    # Only generate more descriptive names if needed for return
    if return_column_names:
        synthetic_column_types = [
            "customer", "product", "sales", "review", "location", 
            "order", "time", "website", "device", "user"
        ]
        synthetic_attributes = [
            "id", "name", "type", "status", "count"
        ]
        
        # Only regenerate names for a subset to save time
        for i in range(min(num_classes, 10)):  # Only first 10 get descriptive names
            col_type = synthetic_column_types[i % len(synthetic_column_types)]
            attribute = synthetic_attributes[i % len(synthetic_attributes)]
            synthetic_columns[i] = f"{col_type}_{attribute}"
    
    # Cache synthetic data if using a seed
    if seed is not None and use_cache:
        # Check tensor size before adding to cache
        tensor_size_bytes = random_tensor.element_size() * random_tensor.nelement()
        
        # Check if adding this tensor would exceed cache size limit
        if _SEMANTIC_DATA_CACHE["cache_size_bytes"] + tensor_size_bytes > _SEMANTIC_DATA_CACHE["max_cache_size_bytes"]:
            # If cache is getting too big, clear it to prevent OOM
            logger.warning(f"Synthetic data cache limit reached ({_SEMANTIC_DATA_CACHE['cache_size_bytes']/1024/1024:.1f}MB), clearing cache")
            # Clear all cached tensors except essential ones
            keys_to_keep = ["is_loaded", "log_file_path", "max_columns", "unwanted_token_ids", "tokenizer", 
                           "cache_size_bytes", "max_cache_size_bytes"]
            keys_to_clear = [k for k in _SEMANTIC_DATA_CACHE.keys() if k not in keys_to_keep]
            for k in keys_to_clear:
                _SEMANTIC_DATA_CACHE[k] = None
            _SEMANTIC_DATA_CACHE["cache_size_bytes"] = 0
        
        # Add to cache and update size tracking
        _SEMANTIC_DATA_CACHE[synthetic_key] = random_tensor
        _SEMANTIC_DATA_CACHE[f"{synthetic_key}_columns"] = synthetic_columns
        _SEMANTIC_DATA_CACHE["cache_size_bytes"] += tensor_size_bytes
    
    # Return both the tensor and column names if requested
    if return_column_names:
        return random_tensor, synthetic_columns
    else:
        return random_tensor

# For backwards compatibility - generate with filtered tokens
random_tensor, column_names = get_random_semantic_data(filter_tokens=True)

# Export the column names for global access
semantic_data_column_names = column_names