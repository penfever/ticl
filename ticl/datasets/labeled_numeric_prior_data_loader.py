"""
Labeled numeric prior data loader for TiCL models.

This module loads and processes numeric column data from tokenized semantic features
and provides them in a format ready for use in the model.
"""

import torch
import os
import json
import re
from typing import Dict, Tuple, Optional, List, Any, Union
import logging
import numpy as np

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
    """
    numeric_indicators = [
        'count', 'amount', 'price', 'rate', 'age', 'year', 'month', 'day',
        'weight', 'height', 'size', 'length', 'width', 'depth', 'date',
        'time', 'duration', 'percentage', 'percent', 'ratio', 'number',
        'score', 'rating', 'rank', 'id', 'index', 'code', 'zip', 'postal',
        'population', 'frequency', 'temperature', 'distance', 'velocity',
        'speed', 'acceleration', 'volume', 'area', 'density', 'mass',
        'pressure', 'energy', 'power', 'force', 'charge', 'voltage',
        'current', 'resistance', 'conductance', 'capacitance', 'inductance',
        'flux', 'intensity', 'luminance', 'exposure', 'dose', 'concentration',
        'ph', 'humidity', 'elevation', 'altitude', 'latitude', 'longitude',
        'grade', 'gpa', 'salary', 'income', 'revenue', 'cost', 'expense',
        'profit', 'loss', 'debt', 'asset', 'liability', 'equity', 'margin',
        'timestamp', 'epoch'
    ]
    
    # Convert column name to lowercase for case-insensitive matching
    column_lower = column_name.lower()
    
    # Check if the column name contains any numeric indicators
    for indicator in numeric_indicators:
        if indicator in column_lower:
            return True
    
    return False

def extract_numeric_values(tokens_data: List[Any]) -> List[float]:
    """
    Extract numeric values from tokenized column data.
    
    This function attempts to find and extract numeric values from the tokenized
    data, typically from the CLIP tokenizer output.
    
    Parameters:
    -----------
    tokens_data : List[Any]
        List of token data that might contain numeric values
        
    Returns:
    --------
    List[float]
        List of extracted numeric values
    """
    numeric_values = []
    
    # Convert tokens to string to search for patterns
    tokens_str = str(tokens_data)
    
    # Use regex to find numeric patterns
    # Look for various numeric formats:
    # - Integers: 123, -456
    # - Decimals: 123.45, -67.89
    # - Scientific notation: 1.23e5, -4.56e-7
    pattern = r'-?\d+\.?\d*(?:[eE][-+]?\d+)?'
    
    matches = re.findall(pattern, tokens_str)
    
    for match in matches:
        try:
            numeric_values.append(float(match))
        except (ValueError, TypeError):
            # Skip values that can't be converted to float
            continue
    
    return numeric_values

def load_numeric_prior_data(
    log_file_path: str = "ticl/datasets/completed_columns.json",
    force_numeric: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Load numeric column data from completed_columns.json.
    
    This function loads tokenized column data, filtering for numeric columns,
    and returns a dictionary mapping column names to tensors of numeric values.
    
    Parameters:
    -----------
    log_file_path : str
        Path to the completed_columns.json file
    force_numeric : bool
        If True, only include columns that appear to be numeric
        
    Returns:
    --------
    Dict[str, torch.Tensor]
        Dictionary mapping column names to tensors of numeric values
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
    
    # Prepare numeric data dictionary
    numeric_data = {}
    
    # Process each column
    for col_name, tensor_data in completed_columns.items():
        # Only include numeric columns if requested
        if force_numeric and not is_numeric_column(col_name):
            continue
        
        # Extract numeric values from the tokenized data
        numeric_values = extract_numeric_values(tensor_data)
        
        # Only include columns with sufficient numeric values
        if len(numeric_values) >= 5:  # Require at least 5 values
            numeric_data[col_name] = torch.tensor(numeric_values, dtype=torch.float32)
    
    logger.info(f"Extracted numeric data for {len(numeric_data)} columns")
    
    return numeric_data

# The example data dictionary below is used for backwards compatibility
# It maps column names to tensors with realistic numeric values
labeled_numeric_data = {
    # Financial/Economic data columns
    "income": torch.tensor([
        25000, 35000, 45000, 55000, 65000, 75000, 85000, 95000, 
        105000, 150000, 200000, 250000, 300000, 500000, 1000000
    ], dtype=torch.float32),
    
    "price": torch.tensor([
        9.99, 19.99, 24.99, 29.99, 49.99, 99.99, 149.99, 
        199.99, 299.99, 499.99, 999.99, 1999.99
    ], dtype=torch.float32),
    
    "interest_rate": torch.tensor([
        0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 
        0.045, 0.05, 0.055, 0.06, 0.07, 0.08, 0.09, 0.1
    ], dtype=torch.float32),
    
    "debt": torch.tensor([
        0, 1000, 5000, 10000, 15000, 20000, 25000, 30000, 
        40000, 50000, 75000, 100000, 150000, 200000
    ], dtype=torch.float32),
    
    # Health/Medical data columns
    "age": torch.tensor([
        1, 2, 3, 5, 10, 15, 18, 21, 25, 30, 35, 40, 
        45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100
    ], dtype=torch.float32),
    
    "weight_kg": torch.tensor([
        2.5, 5, 10, 20, 30, 40, 50, 60, 70, 
        80, 90, 100, 110, 120, 130, 140, 150
    ], dtype=torch.float32),
    
    "height_cm": torch.tensor([
        40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 
        140, 150, 160, 170, 180, 190, 200, 210, 220
    ], dtype=torch.float32),
    
    "blood_pressure_systolic": torch.tensor([
        80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200
    ], dtype=torch.float32),
    
    "blood_pressure_diastolic": torch.tensor([
        50, 60, 70, 80, 90, 100, 110, 120
    ], dtype=torch.float32),
    
    "temperature_celsius": torch.tensor([
        35.0, 35.5, 36.0, 36.5, 37.0, 37.5, 38.0, 
        38.5, 39.0, 39.5, 40.0, 40.5, 41.0, 41.5
    ], dtype=torch.float32),
    
    # Scientific/Measurement data columns
    "rainfall_mm": torch.tensor([
        0, 1, 2, 5, 10, 15, 20, 25, 30, 50, 
        75, 100, 150, 200, 250, 300, 400, 500
    ], dtype=torch.float32),
    
    "temperature_kelvin": torch.tensor([
        173.15, 193.15, 213.15, 233.15, 253.15, 273.15, 
        293.15, 303.15, 313.15, 333.15, 353.15, 373.15, 
        473.15, 573.15, 673.15, 773.15
    ], dtype=torch.float32),
    
    "distance_km": torch.tensor([
        0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 
        200, 500, 1000, 2000, 5000, 10000
    ], dtype=torch.float32),
    
    # Statistical data columns
    "percentage": torch.tensor([
        0, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 
        55, 60, 65, 70, 75, 80, 85, 90, 95, 99, 100
    ], dtype=torch.float32),
    
    "probability": torch.tensor([
        0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 
        0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 
        0.9, 0.95, 0.99, 1.0
    ], dtype=torch.float32),
    
    "count": torch.tensor([
        0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 50, 
        100, 200, 500, 1000, 5000, 10000
    ], dtype=torch.float32),
    
    # Time-related data columns
    "year": torch.tensor([
        1900, 1950, 1970, 1980, 1990, 2000, 2010, 
        2015, 2020, 2021, 2022, 2023, 2024, 2025
    ], dtype=torch.float32),
    
    "month": torch.tensor([
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
    ], dtype=torch.float32),
    
    "day": torch.tensor([
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 
        29, 30, 31
    ], dtype=torch.float32),
    
    "hour": torch.tensor([
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 
        12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
    ], dtype=torch.float32),
}

# Add metadata about each column type
column_metadata = {
    "income": {
        "units": "USD",
        "description": "Annual income in US dollars",
        "domain": "finance"
    },
    "price": {
        "units": "USD",
        "description": "Price of an item in US dollars",
        "domain": "retail"
    },
    "interest_rate": {
        "units": "decimal",
        "description": "Interest rate as a decimal (0.05 = 5%)",
        "domain": "finance"
    },
    "debt": {
        "units": "USD",
        "description": "Amount of debt in US dollars",
        "domain": "finance"
    },
    "age": {
        "units": "years",
        "description": "Age in years",
        "domain": "demographics"
    },
    "weight_kg": {
        "units": "kg",
        "description": "Weight in kilograms",
        "domain": "health"
    },
    "height_cm": {
        "units": "cm",
        "description": "Height in centimeters",
        "domain": "health"
    },
    "blood_pressure_systolic": {
        "units": "mmHg",
        "description": "Systolic blood pressure in mmHg",
        "domain": "health"
    },
    "blood_pressure_diastolic": {
        "units": "mmHg",
        "description": "Diastolic blood pressure in mmHg",
        "domain": "health"
    },
    "temperature_celsius": {
        "units": "°C",
        "description": "Body temperature in degrees Celsius",
        "domain": "health"
    },
    "rainfall_mm": {
        "units": "mm",
        "description": "Rainfall in millimeters",
        "domain": "meteorology"
    },
    "temperature_kelvin": {
        "units": "K",
        "description": "Temperature in Kelvin",
        "domain": "physics"
    },
    "distance_km": {
        "units": "km",
        "description": "Distance in kilometers",
        "domain": "geography"
    },
    "percentage": {
        "units": "%",
        "description": "Value as a percentage (0-100)",
        "domain": "statistics"
    },
    "probability": {
        "units": "decimal",
        "description": "Probability as a decimal (0-1)",
        "domain": "statistics"
    },
    "count": {
        "units": "",
        "description": "Count or frequency of items",
        "domain": "statistics"
    },
    "year": {
        "units": "",
        "description": "Calendar year",
        "domain": "time"
    },
    "month": {
        "units": "",
        "description": "Month of the year (1-12)",
        "domain": "time"
    },
    "day": {
        "units": "",
        "description": "Day of the month (1-31)",
        "domain": "time"
    },
    "hour": {
        "units": "",
        "description": "Hour of the day (0-23)",
        "domain": "time"
    }
}

def get_random_value(column_name: str) -> float:
    """
    Get a random value from a numeric column.
    
    Parameters:
    -----------
    column_name : str
        Name of the column to get a value from
        
    Returns:
    --------
    float
        Random value from the column, or None if not found
    """
    if column_name in labeled_numeric_data:
        values = labeled_numeric_data[column_name]
        idx = torch.randint(0, len(values), (1,))
        return values[idx].item()
    else:
        return None

def get_random_batch(column_name: str, batch_size: int) -> torch.Tensor:
    """
    Get a batch of random values from a numeric column.
    
    Parameters:
    -----------
    column_name : str
        Name of the column to get values from
    batch_size : int
        Number of values to return
        
    Returns:
    --------
    torch.Tensor
        Tensor of random values, or None if column not found
    """
    if column_name in labeled_numeric_data:
        values = labeled_numeric_data[column_name]
        indices = torch.randint(0, len(values), (batch_size,))
        return values[indices]
    else:
        return None

def get_columns_by_domain(domain: str) -> List[str]:
    """
    Get column names by domain.
    
    Parameters:
    -----------
    domain : str
        Domain to filter by
        
    Returns:
    --------
    List[str]
        List of column names in the domain
    """
    return [col for col, meta in column_metadata.items() 
            if meta['domain'] == domain]

def get_noisy_value(column_name: str, noise_factor: float = 0.1) -> float:
    """
    Get a value close to a real value from the column with some random noise.
    
    Parameters:
    -----------
    column_name : str
        Name of the column to get a value from
    noise_factor : float
        Factor to scale the noise by
        
    Returns:
    --------
    float
        Noisy value, or None if column not found
    """
    if column_name in labeled_numeric_data:
        values = labeled_numeric_data[column_name]
        idx = torch.randint(0, len(values), (1,))
        value = values[idx].item()
        
        # Add noise - different scale for different columns
        noise_scale = abs(value * noise_factor)
        if noise_scale == 0:  # For zero values
            noise_scale = 0.1
            
        noise = np.random.normal(0, noise_scale)
        return value + noise
    else:
        return None