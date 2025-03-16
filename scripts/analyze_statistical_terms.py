#!/usr/bin/env python
"""
Analyze statistical features in a simple example.

This script provides a simplified demonstration of how statistical 
features are analyzed and mapped to semantic terms.
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

from ticl.priors.feature_statistical_analyzer import (
    analyze_numerical_feature,
    analyze_categorical_feature,
    map_numeric_stats_to_terms,
    map_categorical_stats_to_terms,
    get_class_description_from_stats
)


def main():
    """Main function to demonstrate statistical feature analysis."""
    parser = argparse.ArgumentParser(description='Demonstrate statistical feature analysis.')
    parser.add_argument('--output', type=str, default='statistical_terms_demo.json', help='Output file for results')
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("Demonstrating statistical feature analysis with synthetic data...")
    
    # Create synthetic numerical feature with two-class discrimination
    # Class 0 will have lower values, Class 1 will have higher values
    num_samples = 100
    feature_values = torch.zeros(num_samples)
    class_labels = torch.zeros(num_samples)
    
    # Assign class 0 to first half of samples, class 1 to second half
    class_labels[num_samples//2:] = 1
    
    # Generate values: class 0 has mean 3, class 1 has mean 8
    feature_values[:num_samples//2] = torch.normal(3.0, 1.0, (num_samples//2,))
    feature_values[num_samples//2:] = torch.normal(8.0, 1.0, (num_samples//2,))
    
    # Analyze the numerical feature
    print("\n===== Analyzing Numerical Feature =====")
    stats = analyze_numerical_feature(
        feature_values,
        class_labels,
        feature_name="temperature",
        is_causal=True
    )
    
    # Print basic statistics
    print(f"\nBasic Statistics:")
    print(f"  Mean: {stats['mean']:.2f}")
    print(f"  Median: {stats['median']:.2f}")
    print(f"  Std Dev: {stats['std']:.2f}")
    print(f"  Range: {stats['min']:.2f} to {stats['max']:.2f}")
    
    # Print class-specific statistics
    print(f"\nClass-Specific Statistics:")
    for class_id, class_stats in stats['class_stats'].items():
        print(f"  Class {int(class_id)}:")
        print(f"    Mean: {class_stats['mean']:.2f}")
        if 'median' in class_stats:
            print(f"    Median: {class_stats['median']:.2f}")
        if 'std' in class_stats:
            print(f"    Std Dev: {class_stats['std']:.2f}")
    
    # Print correlation information
    print(f"\nCorrelation Information:")
    print(f"  Correlation: {stats['correlation']:.4f}")
    print(f"  Correlation Type: {stats['correlation_type']}")
    
    # Generate and print terms for each class
    print(f"\nGenerated Terms for Classes:")
    class_terms = {}
    for class_id in [0, 1]:
        terms = map_numeric_stats_to_terms(stats, "temperature", class_id)
        class_terms[class_id] = terms
        
        print(f"  Class {class_id}:")
        for term in terms:
            print(f"    - {term}")
    
    # Create a categorical feature where:
    # Category 0 -> mostly class 0
    # Category 1 -> mostly class 0
    # Category 2 -> mostly class 1
    # Category 3 -> mostly class 1
    print("\n\n===== Analyzing Categorical Feature =====")
    cat_feature = torch.zeros(num_samples, dtype=torch.long)
    
    # Assign categories with class bias
    for i in range(num_samples):
        if class_labels[i] == 0:
            # For class 0, mostly categories 0 and 1
            cat_feature[i] = 0 if np.random.random() < 0.7 else 1
        else:
            # For class 1, mostly categories 2 and 3
            cat_feature[i] = 2 if np.random.random() < 0.7 else 3
    
    # Analyze the categorical feature
    cat_stats = analyze_categorical_feature(
        cat_feature,
        class_labels,
        feature_name="color",
        is_causal=True,
        num_categories=4
    )
    
    # Print basic statistics
    print(f"\nBasic Statistics:")
    print(f"  Number of Categories: {cat_stats['num_categories']}")
    print(f"  Mode (most frequent category): {cat_stats['mode']}")
    
    # Print category counts
    print(f"\nCategory Counts:")
    for cat, count in cat_stats['category_counts'].items():
        print(f"  Category {cat}: {count}")
    
    # Print conditional probabilities
    print(f"\nConditional Probabilities (P(class | category)):")
    for cat, class_probs in cat_stats['conditional_probs'].items():
        print(f"  Category {cat}:")
        for class_id, prob in class_probs.items():
            print(f"    P(Class {int(class_id)} | Category {cat}) = {prob:.4f}")
    
    # Print correlation information
    print(f"\nCorrelation Information:")
    print(f"  Mutual Information: {cat_stats['mutual_information']:.4f}")
    print(f"  Normalized MI: {cat_stats['normalized_mutual_information']:.4f}")
    print(f"  Correlation Type: {cat_stats['correlation_type']}")
    
    # Generate and print terms for each class
    print(f"\nGenerated Terms for Classes:")
    for class_id in [0, 1]:
        terms = map_categorical_stats_to_terms(cat_stats, "color", class_id)
        class_terms[class_id] = class_terms.get(class_id, []) + terms
        
        print(f"  Class {class_id}:")
        for term in terms:
            print(f"    - {term}")
    
    # Create enhanced class descriptions
    print("\n\n===== Enhanced Class Descriptions =====")
    
    # Create a feature stats dictionary with both features
    feature_stats = {
        0: {  # Numerical feature
            'type': 'numerical',
            'feature_name': 'temperature',
            'is_causal': True,
            **stats
        },
        1: {  # Categorical feature
            'type': 'categorical',
            'feature_name': 'color',
            'is_causal': True,
            **cat_stats
        }
    }
    
    # Generate class descriptions
    for class_id in [0, 1]:
        terms = get_class_description_from_stats(feature_stats, class_id)
        
        # Create a sample class description
        class_name = f"Class_{class_id}"
        column_name = "sample_data"
        
        # Combine terms into an enhanced description
        enhanced_description = f"{class_name} ({column_name})"
        if terms:
            selected_terms = terms[:min(5, len(terms))]
            stats_desc = ", ".join(selected_terms)
            enhanced_description += f": {stats_desc}"
        
        print(f"\nClass {class_id}:")
        print(f"  Original Name: {class_name}")
        print(f"  Column Name: {column_name}")
        print(f"  Enhanced Description: {enhanced_description}")
        print(f"  Statistical Terms:")
        for term in terms:
            print(f"    - {term}")
    
    # Save results to file
    if args.output:
        # Prepare results for saving
        results = {
            'numerical_feature': {
                'name': 'temperature',
                'stats': {k: v for k, v in stats.items() if k not in ['class_stats', 'percentiles']},
                'class_terms': {int(class_id): terms for class_id, terms in stats.get('class_stats', {}).items()}
            },
            'categorical_feature': {
                'name': 'color',
                'stats': {k: v for k, v in cat_stats.items() if k not in ['conditional_probs', 'predictive_categories']},
                'conditional_probs': {int(cat): {int(c): p for c, p in probs.items()} 
                                      for cat, probs in cat_stats.get('conditional_probs', {}).items()}
            },
            'class_terms': {int(class_id): terms for class_id, terms in class_terms.items()}
        }
        
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()