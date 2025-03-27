"""
Statistical column mapping for semantic aware models.
This module provides utilities for mapping statistical features to column names
and generating causal relationships between feature values and target classes.
"""

import torch
import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union

from ticl.datasets.labeled_numeric_prior_data_loader import labeled_numeric_data, column_metadata, get_columns_by_domain
from ticl.datasets.statistical_feature_encoding import calculate_feature_statistics, encode_numerical_features_as_tokens

class StatisticalColumnMapper:
    """
    Maps statistical features to column names and generates causal relationships.
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize the statistical column mapper.
        
        Parameters:
        -----------
        random_seed : int, optional
            Random seed for reproducibility
        """
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        # Load data from labeled_numeric_prior_data_loader
        self.column_data = labeled_numeric_data
        self.column_metadata = column_metadata
        
        # Track assignments for consistent mapping
        self.feature_to_column_map = {}
        self.column_to_feature_map = {}
        
        # Track value ranges for each feature
        self.feature_value_ranges = {}
        
    def assign_column_names(self, feature_stats: Dict[int, Dict[str, Any]]) -> Dict[int, str]:
        """
        Assign meaningful column names to numeric features based on their statistical properties.
        
        Parameters:
        -----------
        feature_stats : Dict[int, Dict[str, Any]]
            Statistics for each feature as returned by calculate_feature_statistics
            
        Returns:
        --------
        Dict[int, str]
            Mapping from feature index to column name
        """
        # Determine which features need column name assignment
        features_to_assign = [idx for idx in feature_stats.keys() 
                             if idx not in self.feature_to_column_map]
        
        # Get all available column names
        available_columns = set(self.column_data.keys()) - set(self.column_to_feature_map.values())
        available_columns = list(available_columns)
        
        if not available_columns:
            # If all columns are used, reset and use any column
            available_columns = list(self.column_data.keys())
        
        # Create mapping for each feature
        column_mapping = {}
        
        for idx in features_to_assign:
            stats = feature_stats[idx]
            
            # Try to find a suitable column based on statistical properties
            selected_column = None
            
            # Get value range
            value_min = stats['min']
            value_max = stats['max']
            value_mean = stats['mean']
            
            # Look for columns with similar ranges
            best_match_score = float('inf')
            
            for col_name in available_columns:
                col_values = self.column_data[col_name]
                col_min = col_values.min().item()
                col_max = col_values.max().item()
                col_mean = col_values.mean().item()
                
                # Calculate score based on how well ranges match
                # Lower score is better
                range_score = abs((value_max - value_min) / max(1, value_mean) - 
                                 (col_max - col_min) / max(1, col_mean))
                
                if range_score < best_match_score:
                    best_match_score = range_score
                    selected_column = col_name
            
            if selected_column:
                # Assign the column
                column_mapping[idx] = selected_column
                self.feature_to_column_map[idx] = selected_column
                self.column_to_feature_map[selected_column] = idx
                
                # Remove from available columns
                available_columns.remove(selected_column)
                
                # If no more columns, break
                if not available_columns:
                    break
        
        # For any remaining features, assign random columns
        for idx in features_to_assign:
            if idx not in column_mapping:
                if available_columns:
                    selected_column = random.choice(available_columns)
                    column_mapping[idx] = selected_column
                    self.feature_to_column_map[idx] = selected_column
                    self.column_to_feature_map[selected_column] = idx
                    available_columns.remove(selected_column)
                else:
                    # If no columns left, use a random one
                    selected_column = random.choice(list(self.column_data.keys()))
                    column_mapping[idx] = selected_column
                    self.feature_to_column_map[idx] = selected_column
        
        return column_mapping
    
    def get_column_value_descriptions(self, column_name: str, num_ranges: int = 3) -> List[str]:
        """
        Get descriptive terms for different value ranges of a column.
        
        Parameters:
        -----------
        column_name : str
            Name of the column
        num_ranges : int
            Number of value ranges to describe (typically 3 for low/medium/high)
            
        Returns:
        --------
        List[str]
            Descriptive terms for each value range
        """
        if column_name not in self.column_data:
            return ["low value", "medium value", "high value"]
        
        # Get column values and metadata
        values = self.column_data[column_name]
        meta = self.column_metadata.get(column_name, {})
        
        # Get domain and units
        domain = meta.get('domain', '')
        units = meta.get('units', '')
        
        # Calculate range boundaries
        min_val = values.min().item()
        max_val = values.max().item()
        range_size = (max_val - min_val) / num_ranges
        
        # Generate descriptive terms
        descriptions = []
        
        for i in range(num_ranges):
            lower_bound = min_val + i * range_size
            upper_bound = min_val + (i + 1) * range_size
            
            # Format values with appropriate precision
            if abs(lower_bound) < 0.1 or abs(upper_bound) < 0.1:
                lower_str = f"{lower_bound:.4f}"
                upper_str = f"{upper_bound:.4f}"
            elif abs(lower_bound) < 1 or abs(upper_bound) < 1:
                lower_str = f"{lower_bound:.2f}"
                upper_str = f"{upper_bound:.2f}"
            else:
                lower_str = f"{int(lower_bound)}"
                upper_str = f"{int(upper_bound)}"
            
            # Add units if available
            if units:
                lower_str = f"{lower_str} {units}"
                upper_str = f"{upper_str} {units}"
            
            # Build description based on range position
            if i == 0:
                term = f"low {column_name} ({lower_str} to {upper_str})"
            elif i == num_ranges - 1:
                term = f"high {column_name} ({lower_str} to {upper_str})"
            else:
                term = f"medium {column_name} ({lower_str} to {upper_str})"
            
            descriptions.append(term)
        
        return descriptions
    
    def get_statistical_token_descriptions(self, token_mapping: Dict) -> Dict[str, List[str]]:
        """
        Get descriptive terms for statistical tokens.
        
        Parameters:
        -----------
        token_mapping : Dict
            Token mapping information as returned by encode_numerical_features_as_tokens
            
        Returns:
        --------
        Dict[str, List[str]]
            Mapping from token type to descriptive terms
        """
        descriptions = {
            'low_tokens': [],
            'med_tokens': [],
            'high_tokens': []
        }
        
        for feature_idx, col_idx in enumerate(token_mapping.get('feature_indices', [])):
            # Find the column name for this feature
            if col_idx in self.feature_to_column_map:
                column_name = self.feature_to_column_map[col_idx]
                
                # Get descriptions for this column's value ranges
                value_descriptions = self.get_column_value_descriptions(column_name)
                
                # Map to token types
                if len(value_descriptions) >= 3:
                    token_type_idx = min(feature_idx, 2)  # Only support up to 3 token types
                    
                    if token_type_idx == 0:
                        descriptions['low_tokens'].append(value_descriptions[0])
                        descriptions['med_tokens'].append(value_descriptions[1])
                        descriptions['high_tokens'].append(value_descriptions[2])
                    elif token_type_idx == 1:
                        descriptions['low_tokens'].append(value_descriptions[0])
                        descriptions['med_tokens'].append(value_descriptions[1])
                        descriptions['high_tokens'].append(value_descriptions[2])
            
        # If no descriptions were added, use defaults
        if not descriptions['low_tokens']:
            descriptions['low_tokens'] = ["low value"]
        if not descriptions['med_tokens']:
            descriptions['med_tokens'] = ["medium value"]
        if not descriptions['high_tokens']:
            descriptions['high_tokens'] = ["high value"]
            
        return descriptions

    def create_token_description_mapping(self, token_mapping: Dict) -> Dict[int, str]:
        """
        Create a mapping from token values to descriptive terms.
        
        Parameters:
        -----------
        token_mapping : Dict
            Token mapping information as returned by encode_numerical_features_as_tokens
            
        Returns:
        --------
        Dict[int, str]
            Mapping from token value to descriptive term
        """
        token_descriptions = {}
        
        # Get column names and descriptions for each token type
        feature_indices = token_mapping.get('feature_indices', [])
        
        for i, feature_idx in enumerate(feature_indices):
            # Get the column name
            if feature_idx in self.feature_to_column_map:
                column_name = self.feature_to_column_map[feature_idx]
                
                # Get token values for this feature
                low_token = token_mapping.get('low_tokens', [])[i] if i < len(token_mapping.get('low_tokens', [])) else None
                med_token = token_mapping.get('med_tokens', [])[i] if i < len(token_mapping.get('med_tokens', [])) else None
                high_token = token_mapping.get('high_tokens', [])[i] if i < len(token_mapping.get('high_tokens', [])) else None
                
                # Create descriptions
                value_descriptions = self.get_column_value_descriptions(column_name)
                
                # Map tokens to descriptions
                if low_token is not None and len(value_descriptions) > 0:
                    token_descriptions[low_token] = value_descriptions[0]
                    
                if med_token is not None and len(value_descriptions) > 1:
                    token_descriptions[med_token] = value_descriptions[1]
                    
                if high_token is not None and len(value_descriptions) > 2:
                    token_descriptions[high_token] = value_descriptions[2]
        
        return token_descriptions
    
    def get_feature_column_mapping(self) -> Dict[int, str]:
        """
        Get the current mapping from feature indices to column names.
        
        Returns:
        --------
        Dict[int, str]
            Mapping from feature index to column name
        """
        return self.feature_to_column_map.copy()