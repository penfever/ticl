#!/usr/bin/env python
"""
Analyze statistical features and their integration into class descriptions.

This script generates synthetic data with causal features, analyzes their
statistical properties, and shows how these properties are incorporated into
class descriptions for semantic learning.
"""

import torch
import numpy as np
import argparse
import os
import sys
import json
from pprint import pprint

# Add the parent directory to the path so we can import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ticl.priors.mlp import MLPPrior
from ticl.priors.classification_adapter import ClassificationAdapterPrior
from ticl.priors.feature_statistical_analyzer import (
    analyze_numerical_feature,
    analyze_categorical_feature,
    map_numeric_stats_to_terms,
    map_categorical_stats_to_terms,
    get_class_description_from_stats
)


def analyze_feature(feature_values, class_labels, feature_idx, is_categorical=False):
    """Analyze a feature and print its statistical properties."""
    print(f"\n{'=' * 80}")
    print(f"Feature {feature_idx} ({'Categorical' if is_categorical else 'Numerical'})")
    print(f"{'-' * 80}")
    
    # Analyze the feature
    if is_categorical:
        stats = analyze_categorical_feature(
            feature_values, 
            class_labels,
            feature_name=f"feature_{feature_idx}",
            is_causal=True
        )
    else:
        stats = analyze_numerical_feature(
            feature_values, 
            class_labels,
            feature_name=f"feature_{feature_idx}",
            is_causal=True
        )
    
    # Print basic statistics
    print(f"Basic Statistics:")
    if not is_categorical:
        print(f"  Mean: {stats.get('mean', 'N/A')}")
        print(f"  Median: {stats.get('median', 'N/A')}")
        print(f"  Std Dev: {stats.get('std', 'N/A')}")
        print(f"  Min: {stats.get('min', 'N/A')}")
        print(f"  Max: {stats.get('max', 'N/A')}")
    else:
        print(f"  Num Categories: {stats.get('num_categories', 'N/A')}")
        print(f"  Mode: {stats.get('mode', 'N/A')}")
        
    # Print class-specific statistics
    print(f"\nClass-Specific Statistics:")
    if 'class_stats' in stats:
        for class_id, class_stats in stats['class_stats'].items():
            print(f"  Class {class_id}:")
            for stat_name, stat_value in class_stats.items():
                if stat_name != 'percentiles':
                    print(f"    {stat_name}: {stat_value}")
    elif 'conditional_probs' in stats:
        for cat, class_probs in stats['conditional_probs'].items():
            print(f"  Category {cat}:")
            for class_id, prob in class_probs.items():
                print(f"    P(Class {class_id} | Cat {cat}) = {prob:.4f}")
    
    # Print correlation information
    print(f"\nCorrelation Information:")
    if not is_categorical:
        print(f"  Correlation: {stats.get('correlation', 'N/A')}")
    else:
        print(f"  Mutual Information: {stats.get('mutual_information', 'N/A')}")
        print(f"  Normalized MI: {stats.get('normalized_mutual_information', 'N/A')}")
    print(f"  Correlation Type: {stats.get('correlation_type', 'N/A')}")
    
    # Generate and print terms for each class
    print(f"\nGenerated Terms for Classes:")
    unique_classes = set()
    if 'class_stats' in stats:
        unique_classes = set(stats['class_stats'].keys())
    elif 'conditional_probs' in stats:
        for class_probs in stats['conditional_probs'].values():
            unique_classes.update(class_probs.keys())
    
    for class_id in sorted(unique_classes):
        if not is_categorical:
            terms = map_numeric_stats_to_terms(stats, f"feature_{feature_idx}", class_id)
        else:
            terms = map_categorical_stats_to_terms(stats, f"feature_{feature_idx}", class_id)
        
        print(f"  Class {class_id}:")
        for term in terms:
            print(f"    - {term}")
    
    return stats


def analyze_class_descriptions(info):
    """Analyze class descriptions created from statistical terms."""
    print(f"\n{'=' * 80}")
    print(f"Class Descriptions with Statistical Terms")
    print(f"{'-' * 80}")
    
    if 'class_token_patterns' not in info:
        print("No class token patterns found in info.")
        return
    
    # Extract class token patterns
    patterns = info['class_token_patterns']
    
    for class_id, pattern in patterns.items():
        print(f"\nClass {class_id}:")
        
        # Print basic class information
        print(f"  Original Name: {pattern.get('class_name', 'N/A')}")
        print(f"  Column Name: {pattern.get('column_name', 'N/A')}")
        
        # Print enhanced class description if available
        if 'enhanced_class_name' in pattern:
            print(f"  Enhanced Description: {pattern['enhanced_class_name']}")
        
        # Print statistical terms if available
        if 'statistical_terms' in pattern:
            print(f"  Statistical Terms:")
            for term in pattern['statistical_terms']:
                print(f"    - {term}")


def main():
    """Main function to generate and analyze data with statistical features."""
    parser = argparse.ArgumentParser(description='Analyze statistical features in synthetic data.')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for synthetic data')
    parser.add_argument('--num-features', type=int, default=10, help='Number of features')
    parser.add_argument('--num-samples', type=int, default=100, help='Number of samples')
    parser.add_argument('--output', type=str, default='statistical_analysis.json', help='Output file for results')
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    print(f"Generating synthetic data with {args.num_features} features and {args.num_samples} samples...")
    
    # Create MLPPrior with a complete config
    mlp_config = {
        "sampling": "normal",
        "num_layers": 3,
        "prior_mlp_hidden_dim": 128,
        "prior_mlp_activations": torch.nn.ReLU,
        "noise_std": 0.1,
        "y_is_effect": True,
        "pre_sample_weights": True,
        "prior_mlp_dropout_prob": 0.1,
        "pre_sample_causes": True,
        "prior_mlp_scale_weights_sqrt": True,
        "random_feature_rotation": False,
        "add_uninformative_features": False,
        "is_causal": True,
        "num_causes": 5,
        "block_wise_dropout": False,
        "init_std": 1.0,
        "sort_features": True,
        "in_clique": False
    }
    base_prior = MLPPrior(config=mlp_config)
    
    # Create ClassificationAdapterPrior with semantic features enabled
    config = {
        'max_num_classes': 2,
        'num_classes': 2,
        'output_multiclass_ordered_p': 0.0,
        'multiclass_max_steps': 10,
        'multiclass_type': 'rank',
        'categorical_feature_p': 0.3,  # 30% chance for categorical features
        'semantic_feature_p': 1.0,     # Always include semantic features
        'nan_prob_no_reason': 0.0,
        'nan_prob_a_reason': 0.0,
        'set_value_to_nan': 0.9,
        'num_features_sampler': 'uniform',
        'pad_zeros': True,
        'balanced': False,
        'feature_curriculum': False
    }
    adapter = ClassificationAdapterPrior(base_prior, **config)
    
    # Generate data
    device = torch.device('cpu')
    x, y, y_, info = adapter.get_batch(
        batch_size=args.batch_size,
        n_samples=args.num_samples,
        num_features=args.num_features,
        device=device,
        single_eval_pos=args.num_samples // 2
    )
    
    print(f"Data generated with shape: {x.shape}")
    
    # Analyze feature statistics if available
    if 'feature_stats' in info:
        print("\nAnalyzing feature statistics...")
        
        for batch_id, batch_stats in info['feature_stats'].items():
            print(f"\n{'#' * 100}")
            print(f"Batch {batch_id}")
            print(f"{'#' * 100}")
            
            # Analyze each feature
            for feature_idx, feature_stats in batch_stats.items():
                feature_type = feature_stats.get('type', 'unknown')
                is_categorical = feature_type == 'categorical'
                
                # Extract feature values and labels for this batch
                feature_values = x[:, batch_id, int(feature_idx)]
                batch_labels = y[:, batch_id]
                
                # Analyze the feature (this will re-run the analysis for demonstration)
                analyze_feature(feature_values, batch_labels, feature_idx, is_categorical)
    else:
        print("No feature statistics found in the data. Statistical analysis was not run.")
    
    # Analyze class descriptions
    analyze_class_descriptions(info)
    
    # Skip saving results to file to avoid JSON serialization issues
    print("\nSkipping results file save to avoid serialization issues with tensor objects")


if __name__ == "__main__":
    main()