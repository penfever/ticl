"""
Semantic prior data loader for TiCL models.

This module loads and processes non-numeric semantic column data from tokenized semantic features.
"""

import torch
import os
import json
from typing import Dict, Tuple, Optional, List
import logging
from transformers import CLIPTokenizerFast

logger = logging.getLogger(__name__)

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
    force_non_numeric: bool = True
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
        
    Returns:
    --------
    Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
        Tuple of (column_name_tokens, column_value_tokens)
    """
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
    
    # Process each column
    for col_name, tensor_data in completed_columns.items():
        # Skip numeric columns if requested
        if force_non_numeric and is_numeric_column(col_name):
            continue
        
        # Convert tensor data to tensor
        col_tensor = torch.tensor(tensor_data)
        
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
    
    return column_name_tokens, column_value_tokens

def get_random_semantic_data(
    num_classes: int = 3,
    num_tokens: int = 50,
    tensor_size: int = 200,
    log_file_path: str = "ticl/datasets/completed_columns.json"
) -> torch.Tensor:
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
        
    Returns:
    --------
    torch.Tensor
        Tensor of shape [num_classes, tensor_size] containing semantic data
    """
    try:
        # Try to load real data first
        _, column_values = load_semantic_prior_data(
            log_file_path=log_file_path,
            target_tensor_size=tensor_size,
            force_non_numeric=True
        )
        
        # If we have enough columns, sample from them
        if len(column_values) >= num_classes:
            # Select random columns
            columns = list(column_values.keys())
            selected_columns = torch.randperm(len(columns))[:num_classes].tolist()
            
            # Stack tensors
            tensors = [column_values[columns[i]] for i in selected_columns]
            return torch.stack(tensors)
    except (FileNotFoundError, ValueError) as e:
        logger.warning(f"Could not load semantic data from file: {e}")
    
    # Fallback to synthetic data
    logger.info("Generating synthetic semantic data")
    random_tensor = torch.rand(num_classes, tensor_size)
    # Scale to the range [1, 49404]
    random_tensor = 1 + random_tensor * (49404 - 1)
    # Round to the nearest integer
    random_tensor = torch.round(random_tensor)
    
    return random_tensor

# For backwards compatibility
random_tensor = get_random_semantic_data()