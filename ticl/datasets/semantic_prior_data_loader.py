"""
Semantic prior data loader for TiCL models.

This module loads and processes non-numeric semantic column data from tokenized semantic features.
"""

import torch
import os
import json
from typing import Dict, Tuple, Optional, List
import logging
import sys
import time
from transformers import CLIPTokenizerFast

# Use the memory profiling logger from utils
memory_logger = logging.getLogger("memory_profiling")

# If imported directly, configure logger
if not memory_logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    handler.setLevel(logging.INFO)  # Only show INFO and higher to console
    memory_logger.addHandler(handler)
    memory_logger.setLevel(logging.DEBUG)  # Log everything to file
    
    # Also add file handler for persistent logging
    try:
        file_handler = logging.FileHandler("semantic_data_loading.log")
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        file_handler.setLevel(logging.DEBUG)  # Include DEBUG messages in file
        memory_logger.addHandler(file_handler)
    except:
        memory_logger.warning("Could not create log file for semantic data loading")

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
    force_non_numeric: bool = True,
    max_columns: Optional[int] = None
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
        
    Returns:
    --------
    Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
        Tuple of (column_name_tokens, column_value_tokens)
    """
    memory_logger.debug(f"=== STARTING SEMANTIC DATA LOADING ===")
    memory_logger.debug(f"Parameters: log_file_path={log_file_path}, target_size={target_tensor_size}, max_columns={max_columns}")
    
    if not os.path.exists(log_file_path):
        memory_logger.error(f"File not found: {log_file_path}")
        raise FileNotFoundError(f"Semantic column data file not found at {log_file_path}")
    
    # Get file size for debugging
    file_size_mb = os.path.getsize(log_file_path) / (1024 * 1024)
    memory_logger.debug(f"JSON file size: {file_size_mb:.2f} MB")
    
    # Load data from JSON file
    try:
        memory_logger.debug(f"Loading JSON data from {log_file_path}")
        with open(log_file_path, 'r') as f:
            completed_columns = json.load(f)
        memory_logger.debug(f"Successfully loaded JSON with {len(completed_columns)} columns")
        
        # Sample and log a few columns for debugging
        sample_cols = list(completed_columns.keys())[:3]  # First 3 columns
        memory_logger.debug(f"SAMPLE COLUMNS: {sample_cols}")
        for col in sample_cols:
            data = completed_columns[col]
            memory_logger.debug(f"Column '{col}' sample data (first 20 tokens): {data[:20]}")
            memory_logger.debug(f"Column '{col}' data type: {type(data)}, length: {len(data)}")
    except json.JSONDecodeError:
        memory_logger.error(f"Error parsing JSON file: {log_file_path}")
        raise ValueError(f"Error parsing JSON file: {log_file_path}")
    
    logger.info(f"Loaded {len(completed_columns)} columns from {log_file_path}")
    
    # Initialize the CLIP tokenizer for column names
    memory_logger.debug(f"Initializing CLIP tokenizer: {clip_tokenizer_name}")
    tokenizer = CLIPTokenizerFast.from_pretrained(clip_tokenizer_name)
    
    # Prepare column name and value dictionaries
    column_name_tokens = {}
    column_value_tokens = {}
    
    # Determine maximum tensor size if not specified
    if target_tensor_size is None:
        memory_logger.debug("Detecting maximum tensor size from data")
        sizes = [len(tensor_data) for tensor_data in completed_columns.values()]
        target_tensor_size = max(sizes) if sizes else 100
        memory_logger.debug(f"Auto-detected maximum tensor size: {target_tensor_size}")
        logger.info(f"Using detected maximum tensor size: {target_tensor_size}")
    
    # Filter non-numeric columns first
    memory_logger.debug("Filtering non-numeric columns")
    filtered_columns = {}
    
    # Track column filtering decisions for debugging
    numeric_column_count = 0
    kept_column_count = 0
    
    for col_name, tensor_data in completed_columns.items():
        is_numeric = is_numeric_column(col_name)
        if is_numeric:
            numeric_column_count += 1
            memory_logger.debug(f"Column '{col_name}' detected as numeric - {'filtered out' if force_non_numeric else 'kept'}")
        
        if not force_non_numeric or not is_numeric:
            filtered_columns[col_name] = tensor_data
            kept_column_count += 1
    
    memory_logger.debug(f"Filtering stats: {numeric_column_count} numeric columns detected, {kept_column_count} columns kept")
    memory_logger.debug(f"Filtered from {len(completed_columns)} to {len(filtered_columns)} non-numeric columns")
    
    # If max_columns is set, limit the number of columns to process
    if max_columns is not None and max_columns < len(filtered_columns):
        memory_logger.debug(f"Limiting to {max_columns} columns (out of {len(filtered_columns)} filtered columns)")
        # Sort by column name for deterministic behavior
        sorted_cols = sorted(filtered_columns.keys())
        # Take the first max_columns
        selected_cols = sorted_cols[:max_columns]
        filtered_columns = {col: filtered_columns[col] for col in selected_cols}
        memory_logger.debug(f"Selected {len(filtered_columns)} columns after max_columns limit applied")
        memory_logger.debug(f"SELECTED COLUMNS (first 10): {list(filtered_columns.keys())[:10]}")
        logger.info(f"Limited to {len(filtered_columns)} columns due to max_columns={max_columns}")
    
    # Keep track of total memory used by tensors
    total_tensor_bytes = 0
    
    # Process each column
    memory_logger.debug(f"Processing {len(filtered_columns)} columns")
    for col_idx, (col_name, tensor_data) in enumerate(filtered_columns.items()):
        if col_idx % 10 == 0:
            memory_logger.debug(f"Processing column {col_idx}/{len(filtered_columns)}: {col_name}")
        
        # Convert tensor data to tensor and check size
        tensor_length = len(tensor_data)
        memory_logger.debug(f"Column {col_name}: Raw data length: {tensor_length}")
        
        # Log raw data sample for debugging
        if col_idx < 3:  # Only log first 3 columns to avoid too much output
            memory_logger.debug(f"Column {col_name} raw data sample: {tensor_data[:10]}")
        
        # Convert tensor data to tensor
        col_tensor = torch.tensor(tensor_data, dtype=torch.int32)  # Use int32 to save memory
        tensor_size_mb = col_tensor.element_size() * col_tensor.nelement() / (1024 * 1024)
        memory_logger.debug(f"Column {col_name}: Tensor size: {tensor_size_mb:.2f} MB, shape: {col_tensor.shape}, dtype: {col_tensor.dtype}")
        
        # Log tensor sample values
        if col_idx < 3:
            memory_logger.debug(f"Column {col_name} tensor sample: {col_tensor[:10]}")
        
        # Ensure consistent tensor size
        if col_tensor.size(0) < target_tensor_size:
            # Pad with zeros if needed
            padding = torch.zeros(target_tensor_size - col_tensor.size(0), dtype=col_tensor.dtype)
            col_tensor = torch.cat([col_tensor, padding])
            memory_logger.debug(f"Column {col_name}: Padded from {tensor_length} to {target_tensor_size}")
        elif col_tensor.size(0) > target_tensor_size:
            # Truncate if larger
            col_tensor = col_tensor[:target_tensor_size]
            memory_logger.debug(f"Column {col_name}: Truncated from {tensor_length} to {target_tensor_size}")
        
        # Process column name for tokens dictionary
        # First tokenize the column name
        col_name_clean = col_name.replace("_", " ").replace("-", " ")
        memory_logger.debug(f"Tokenizing column name: '{col_name}' → '{col_name_clean}'")
        
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
        
        # Log tokenization results for a few examples
        if col_idx < 3:
            token_ids = col_name_tokens.tolist()
            try:
                tokens = tokenizer.convert_ids_to_tokens(token_ids)
                memory_logger.debug(f"Column '{col_name}' tokenized as: {tokens[:10]}... (token_ids: {token_ids[:10]}...)")
            except:
                memory_logger.debug(f"Column '{col_name}' token_ids: {token_ids[:10]}...")
        
        # Store in dictionaries
        column_name_tokens[col_name] = col_name_tokens
        column_value_tokens[col_name] = col_tensor
        
        # Track memory usage
        tensor_bytes = col_tensor.element_size() * col_tensor.nelement()
        total_tensor_bytes += tensor_bytes
    
    total_tensor_mb = total_tensor_bytes / (1024 * 1024)
    memory_logger.debug(f"Total tensor memory usage: {total_tensor_mb:.2f} MB for {len(column_value_tokens)} columns")
    memory_logger.debug(f"Average tensor size: {total_tensor_mb / len(column_value_tokens):.2f} MB per column")
    logger.info(f"Loaded {len(column_name_tokens)} non-numeric columns with target size {target_tensor_size}")
    
    # Log a summary of the token distributions for debugging
    memory_logger.debug(f"=== SEMANTIC TOKEN DISTRIBUTION STATS ===")
    try:
        # Select a few sample columns
        sample_cols = list(column_value_tokens.keys())[:5]
        for col in sample_cols:
            tensor = column_value_tokens[col]
            unique_tokens = torch.unique(tensor, return_counts=True)
            token_ids = unique_tokens[0].tolist()
            token_counts = unique_tokens[1].tolist()
            
            # Count non-zero tokens
            non_zero = (tensor > 0).sum().item()
            zero_percent = 100 - (non_zero / tensor.numel() * 100)
            
            memory_logger.debug(f"Column '{col}': {len(token_ids)} unique tokens, {zero_percent:.1f}% zeros")
            memory_logger.debug(f"  Most frequent tokens: {token_ids[:5]} (counts: {token_counts[:5]})")
    except Exception as e:
        memory_logger.debug(f"Error analyzing token distribution: {e}")
    
    memory_logger.debug(f"=== COMPLETED SEMANTIC DATA LOADING ===")
    
    return column_name_tokens, column_value_tokens

def get_random_semantic_data(
    num_classes: int = 3,
    num_tokens: int = 50,
    tensor_size: int = 200,
    log_file_path: str = "ticl/datasets/completed_columns.json",
    max_tensor_size_mb: int = 10  # Limit tensor size to 10MB by default
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
    max_tensor_size_mb : int
        Maximum size of the returned tensor in MB to avoid OOM issues
        
    Returns:
    --------
    torch.Tensor
        Tensor of shape [num_classes, tensor_size] containing semantic data
    """
    memory_logger.debug(f"=== get_random_semantic_data called ===")
    memory_logger.debug(f"Called with: num_classes={num_classes}, num_tokens={num_tokens}, tensor_size={tensor_size}")
    memory_logger.debug(f"File path: {log_file_path}, max_size: {max_tensor_size_mb}MB")
    
    # Calculate max size that would keep the whole tensor < max_tensor_size_mb
    max_allowed_size = int((max_tensor_size_mb * 1024 * 1024) / (num_classes * 4))  # 4 bytes per int32
    if tensor_size > max_allowed_size:
        memory_logger.warning(f"Requested tensor_size={tensor_size} would exceed {max_tensor_size_mb}MB limit.")
        memory_logger.warning(f"Limiting tensor_size to {max_allowed_size}")
        tensor_size = max_allowed_size
    
    # Track which method is used to provide data
    method_used = "unknown"
    start_time = time.time()
    
    try:
        # Try to load real data first, but strictly limit to save memory
        memory_logger.debug(f"Attempting to load real semantic data from {log_file_path}")
        file_exists = os.path.exists(log_file_path)
        memory_logger.debug(f"File exists: {file_exists}")
        
        if not file_exists:
            raise FileNotFoundError(f"Semantic data file not found: {log_file_path}")
        
        # Log file stats
        file_size_mb = os.path.getsize(log_file_path) / (1024 * 1024)
        memory_logger.debug(f"Semantic data file size: {file_size_mb:.2f}MB")
        
        # Load real data
        memory_logger.debug(f"Loading semantic data with target_tensor_size={tensor_size}, max_columns={num_classes}")
        _, column_values = load_semantic_prior_data(
            log_file_path=log_file_path,
            target_tensor_size=tensor_size,
            force_non_numeric=True,
            max_columns=num_classes * 2  # Load 2x columns to ensure we have enough to sample from
        )
        
        # If we have enough columns, sample from them
        if len(column_values) >= num_classes:
            memory_logger.debug(f"Successfully loaded {len(column_values)} columns, selecting {num_classes}")
            method_used = "real_data"
            
            # Select random columns
            columns = list(column_values.keys())
            selected_indices = torch.randperm(len(columns))[:num_classes].tolist()
            selected_columns = [columns[i] for i in selected_indices]
            
            memory_logger.debug(f"Selected columns: {selected_columns}")
            
            # Log samples of the selected columns
            for i, col in enumerate(selected_columns[:2]):  # Log first 2 selected columns
                tensor = column_values[col]
                memory_logger.debug(f"Selected column {i} ('{col}'): Shape: {tensor.shape}, Sample: {tensor[:5]}...")
                
                # Count non-zero elements to understand sparsity
                non_zero = (tensor > 0).sum().item()
                sparsity = 100 - (non_zero / tensor.numel() * 100)
                memory_logger.debug(f"  Column '{col}' sparsity: {sparsity:.1f}% zeros")
            
            # Stack tensors
            memory_logger.debug("Stacking tensors from selected columns")
            tensors = [column_values[col] for col in selected_columns]
            semantic_tensor = torch.stack(tensors)
            
            # Force to int32 to save memory
            semantic_tensor = semantic_tensor.to(dtype=torch.int32)
            
            # Calculate size
            tensor_size_mb = semantic_tensor.element_size() * semantic_tensor.nelement() / (1024 * 1024)
            memory_logger.debug(f"Created semantic tensor with shape {semantic_tensor.shape}, " 
                             f"size={tensor_size_mb:.2f}MB, dtype={semantic_tensor.dtype}")
            
            # Log token distribution
            try:
                unique_tokens = torch.unique(semantic_tensor).tolist()
                memory_logger.debug(f"Number of unique tokens: {len(unique_tokens)}")
                memory_logger.debug(f"Token range: min={min(unique_tokens)}, max={max(unique_tokens)}")
                
                # Check distribution of token values
                token_counts = {}
                for i in range(0, 50000, 5000):  # Check distribution in 5000-wide bins
                    count = ((semantic_tensor >= i) & (semantic_tensor < i + 5000)).sum().item()
                    token_counts[f"{i}-{i+4999}"] = count
                
                memory_logger.debug(f"Token distribution: {token_counts}")
            except Exception as e:
                memory_logger.debug(f"Error analyzing token distribution: {e}")
            
            # Make sure the tensor stays on CPU
            if semantic_tensor.device.type != 'cpu':
                memory_logger.warning(f"Moving semantic tensor from {semantic_tensor.device} to CPU")
                semantic_tensor = semantic_tensor.to('cpu')
            
            processing_time = time.time() - start_time
            memory_logger.debug(f"Semantic data processing took {processing_time:.2f}s (method: {method_used})")
            return semantic_tensor
    except (FileNotFoundError, ValueError, Exception) as e:
        memory_logger.error(f"Could not load semantic data from file: {e}")
        logger.warning(f"Could not load semantic data from file: {e}")
    
    # Fallback to synthetic data
    method_used = "synthetic"
    memory_logger.debug("Generating synthetic semantic data")
    logger.info("Generating synthetic semantic data")
    
    # Create random tensor with controlled size
    memory_logger.debug(f"Creating random tensor with shape [{num_classes}, {tensor_size}]")
    random_tensor = torch.rand(num_classes, tensor_size)
    
    # Scale to the range [1, 49404] (CLIP vocabulary size range)
    memory_logger.debug(f"Scaling random values to token ID range [1, 49404]")
    random_tensor = 1 + random_tensor * (49404 - 1)
    
    # Round to the nearest integer
    random_tensor = torch.round(random_tensor).to(dtype=torch.int32)  # Use int32 to save memory
    
    # Calculate memory usage
    tensor_size_mb = random_tensor.element_size() * random_tensor.nelement() / (1024 * 1024)
    memory_logger.debug(f"Created synthetic tensor with shape {random_tensor.shape}, "
                      f"size={tensor_size_mb:.2f}MB, dtype={random_tensor.dtype}")
    
    # Log sample of synthetic data
    memory_logger.debug(f"Synthetic data sample (first 3 classes, first 10 tokens):")
    for i in range(min(3, num_classes)):
        memory_logger.debug(f"Class {i}: {random_tensor[i, :10].tolist()}")
    
    processing_time = time.time() - start_time
    memory_logger.debug(f"Semantic data processing took {processing_time:.2f}s (method: {method_used})")
    
    return random_tensor

# For backwards compatibility
random_tensor = get_random_semantic_data()