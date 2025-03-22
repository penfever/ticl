#!/usr/bin/env python
"""
Utility for integrating different semantic feature sets into the semantic prior data loader.
This script helps combine and prepare different semantic feature types for use in training.
"""

import os
import sys
import torch
import json
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Optional

# Add the parent directory to the path to import from ticl
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ticl.datasets.semantic_prior_data_loader import (
    get_unwanted_token_ids,
    is_float,
    is_int
)
from transformers import CLIPTokenizerFast

class SemanticFeatureIntegrator:
    """
    Utility class for integrating different semantic feature sets.
    """
    
    def __init__(
        self,
        clip_tokenizer: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        max_tokens: int = 200
    ):
        """
        Initialize the integrator.
        
        Args:
            clip_tokenizer: CLIP tokenizer name
            device: Computing device for tensor operations
            max_tokens: Maximum number of tokens to keep
        """
        self.tokenizer = CLIPTokenizerFast.from_pretrained(clip_tokenizer)
        self.max_tokens = max_tokens
        
        # Determine device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch, 'xpu') and torch.xpu.is_available():
                self.device = "xpu"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        # Get unwanted token IDs for filtering
        self.unwanted_token_ids = get_unwanted_token_ids(self.tokenizer)
    
    def filter_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Filter unwanted tokens from a tensor.
        
        Args:
            tokens: Input token tensor
            
        Returns:
            Filtered token tensor
        """
        # Create a mask of tokens to keep (not in unwanted list)
        mask = torch.ones_like(tokens, dtype=torch.bool)
        for token_id in self.unwanted_token_ids:
            mask &= (tokens != token_id)
        
        # Apply mask to keep only non-unwanted tokens
        filtered_tokens = tokens.clone()
        filtered_tokens[~mask] = 0
        
        return filtered_tokens
    
    def load_feature_set(self, feature_file: str) -> torch.Tensor:
        """
        Load a semantic feature set from a file.
        
        Args:
            feature_file: Path to the feature file (.pt)
            
        Returns:
            Tensor of features
        """
        try:
            features = torch.load(feature_file, map_location=self.device)
            print(f"Loaded {features.shape} features from {feature_file}")
            return features
        except Exception as e:
            print(f"Error loading {feature_file}: {e}")
            return None
    
    def combine_feature_sets(
        self,
        primary_features: torch.Tensor,
        secondary_features: torch.Tensor,
        primary_weight: float = 0.7,
        method: str = "weighted_sum"
    ) -> torch.Tensor:
        """
        Combine two feature sets.
        
        Args:
            primary_features: Primary feature tensor
            secondary_features: Secondary feature tensor
            primary_weight: Weight for the primary features (0-1)
            method: Combination method ("weighted_sum", "concatenate", or "alternate")
            
        Returns:
            Combined feature tensor
        """
        if primary_features.shape != secondary_features.shape:
            raise ValueError(f"Feature shapes don't match: {primary_features.shape} vs {secondary_features.shape}")
        
        if method == "weighted_sum":
            # Weighted sum of the features
            secondary_weight = 1.0 - primary_weight
            combined = primary_weight * primary_features + secondary_weight * secondary_features
            
        elif method == "concatenate":
            # Concatenate half of each feature set
            half_tokens = primary_features.shape[1] // 2
            combined = torch.cat([
                primary_features[:, :half_tokens],
                secondary_features[:, half_tokens:]
            ], dim=1)
            
        elif method == "alternate":
            # Alternate between primary and secondary features
            combined = torch.zeros_like(primary_features)
            combined[:, ::2] = primary_features[:, ::2]  # Even indices
            combined[:, 1::2] = secondary_features[:, 1::2]  # Odd indices
            
        else:
            raise ValueError(f"Unknown combination method: {method}")
        
        return combined
    
    def create_enhanced_feature_file(
        self,
        feature_files: List[str],
        output_file: str,
        column_names: Optional[List[str]] = None,
        combination_method: str = "weighted_sum",
        weights: Optional[List[float]] = None,
        filter_unwanted: bool = True
    ) -> None:
        """
        Create an enhanced feature file by combining multiple feature sets.
        
        Args:
            feature_files: List of feature files to combine
            output_file: Path to save the combined features
            column_names: List of column names (used for logging)
            combination_method: Method for combining features
            weights: List of weights for each feature file (must sum to 1)
            filter_unwanted: Whether to filter unwanted tokens
        """
        if len(feature_files) == 0:
            raise ValueError("No feature files provided")
        
        # Validate weights
        if weights is None:
            # Equal weighting
            weights = [1.0 / len(feature_files)] * len(feature_files)
        elif len(weights) != len(feature_files):
            raise ValueError(f"Number of weights ({len(weights)}) doesn't match number of feature files ({len(feature_files)})")
        elif abs(sum(weights) - 1.0) > 1e-4:
            print(f"Warning: Weights sum to {sum(weights)}, normalizing to 1.0")
            weights = [w / sum(weights) for w in weights]
        
        # Load all feature sets
        all_features = []
        for file_path in feature_files:
            features = self.load_feature_set(file_path)
            if features is not None:
                all_features.append(features)
            else:
                print(f"Skipping invalid feature file: {file_path}")
        
        if len(all_features) == 0:
            raise ValueError("No valid feature files loaded")
        
        # Check shapes
        shapes = [f.shape for f in all_features]
        if len(set(shapes)) > 1:
            raise ValueError(f"Feature shapes don't match: {shapes}")
        
        # Combine features
        if len(all_features) == 1:
            # Only one feature set, no need to combine
            combined_features = all_features[0]
        else:
            if combination_method == "weighted_sum":
                # Weighted sum of all feature sets
                combined_features = torch.zeros_like(all_features[0])
                for i, features in enumerate(all_features):
                    combined_features += weights[i] * features
                    
            elif combination_method == "concatenate":
                # Split each feature set and concatenate parts
                num_features = len(all_features)
                tokens_per_feature = all_features[0].shape[1] // num_features
                
                combined_parts = []
                for i, features in enumerate(all_features):
                    start_idx = i * tokens_per_feature
                    end_idx = (i + 1) * tokens_per_feature if i < num_features - 1 else all_features[0].shape[1]
                    combined_parts.append(features[:, start_idx:end_idx])
                
                combined_features = torch.cat(combined_parts, dim=1)
                
            elif combination_method == "alternate":
                # Alternate between feature sets
                combined_features = torch.zeros_like(all_features[0])
                num_features = len(all_features)
                
                for i in range(all_features[0].shape[1]):
                    feature_idx = i % num_features
                    combined_features[:, i] = all_features[feature_idx][:, i]
                    
            else:
                raise ValueError(f"Unknown combination method: {combination_method}")
        
        # Filter unwanted tokens if requested
        if filter_unwanted:
            print("Filtering unwanted tokens...")
            combined_features = self.filter_tokens(combined_features)
        
        # Save the combined features
        torch.save(combined_features, output_file)
        print(f"Saved combined features with shape {combined_features.shape} to {output_file}")
        
        # Print sample of the features
        if column_names is not None and len(column_names) == combined_features.shape[0]:
            print("\nSample of combined features:")
            for i in range(min(5, len(column_names))):
                col_name = column_names[i]
                # Get non-zero tokens
                tokens = combined_features[i]
                non_zero_tokens = tokens[tokens != 0]
                
                # Decode tokens
                decoded = self.tokenizer.decode(non_zero_tokens)
                print(f"{col_name}: {decoded[:100]}...")

def main():
    parser = argparse.ArgumentParser(description="Integrate different semantic feature sets")
    
    parser.add_argument(
        "--feature-files",
        type=str,
        nargs="+",
        required=True,
        help="Paths to the feature files to combine"
    )
    
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to save the combined features"
    )
    
    parser.add_argument(
        "--combination-method",
        type=str,
        choices=["weighted_sum", "concatenate", "alternate"],
        default="weighted_sum",
        help="Method for combining features"
    )
    
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=None,
        help="Weights for each feature file (must sum to 1)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Computing device for tensor operations"
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Maximum number of tokens to keep"
    )
    
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Don't filter unwanted tokens"
    )
    
    parser.add_argument(
        "--column-names-file",
        type=str,
        default=None,
        help="JSON file containing column names"
    )
    
    args = parser.parse_args()
    
    # Load column names if provided
    column_names = None
    if args.column_names_file:
        try:
            with open(args.column_names_file, 'r') as f:
                column_names = json.load(f)
        except Exception as e:
            print(f"Error loading column names: {e}")
    
    # Create the integrator
    integrator = SemanticFeatureIntegrator(
        device=args.device,
        max_tokens=args.max_tokens
    )
    
    # Create the enhanced feature file
    integrator.create_enhanced_feature_file(
        feature_files=args.feature_files,
        output_file=args.output_file,
        column_names=column_names,
        combination_method=args.combination_method,
        weights=args.weights,
        filter_unwanted=not args.no_filter
    )

if __name__ == "__main__":
    main()