"""
Feature Statistical Analyzer for TiCL models.

This module analyzes numerical and categorical features to detect statistical properties
and maps them to semantic terms for enhancing class descriptions.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

logger = logging.getLogger(__name__)

# Constants for percentile thresholds
VERY_LOW_THRESHOLD = 10
LOW_THRESHOLD = 25
MEDIUM_LOW_THRESHOLD = 40
MEDIUM_THRESHOLD = 50
MEDIUM_HIGH_THRESHOLD = 60
HIGH_THRESHOLD = 75
VERY_HIGH_THRESHOLD = 90

# Constants for correlation strength
STRONG_CORRELATION = 0.6
MODERATE_CORRELATION = 0.3
WEAK_CORRELATION = 0.1

def analyze_numerical_feature(
    feature_values: torch.Tensor,
    class_labels: torch.Tensor,
    feature_name: str = "feature",
    is_causal: bool = False
) -> Dict[str, Any]:
    """
    Analyze a numerical feature to detect statistical properties and relationship to classes.
    
    Args:
        feature_values: Tensor of feature values
        class_labels: Tensor of class labels
        feature_name: Name of the feature
        is_causal: Whether the feature is causal
        
    Returns:
        Dictionary of statistical properties and their associations with classes
    """
    stats = {}
    
    # Ensure tensors are on the same device
    if feature_values.device != class_labels.device:
        class_labels = class_labels.to(feature_values.device)
    
    # Convert to numpy for easier analysis
    values_np = feature_values.detach().cpu().numpy().flatten()
    labels_np = class_labels.detach().cpu().numpy().flatten()
    
    # Filter out invalid values (NaN, inf, etc.)
    valid_mask = np.isfinite(values_np) & np.isfinite(labels_np)
    values_np = values_np[valid_mask]
    labels_np = labels_np[valid_mask]
    
    if len(values_np) == 0:
        return {"error": "No valid values for analysis"}
    
    # Basic statistics
    stats['mean'] = float(np.mean(values_np))
    stats['median'] = float(np.median(values_np))
    stats['std'] = float(np.std(values_np))
    stats['min'] = float(np.min(values_np))
    stats['max'] = float(np.max(values_np))
    
    # Percentile thresholds for terms like "high", "low", etc.
    stats['percentiles'] = {
        'very_low': float(np.percentile(values_np, VERY_LOW_THRESHOLD)),
        'low': float(np.percentile(values_np, LOW_THRESHOLD)),
        'medium_low': float(np.percentile(values_np, MEDIUM_LOW_THRESHOLD)),
        'medium': float(np.percentile(values_np, MEDIUM_THRESHOLD)),
        'medium_high': float(np.percentile(values_np, MEDIUM_HIGH_THRESHOLD)),
        'high': float(np.percentile(values_np, HIGH_THRESHOLD)),
        'very_high': float(np.percentile(values_np, VERY_HIGH_THRESHOLD))
    }
    
    # For causal features, analyze relationship with classes
    if is_causal:
        # Get unique class labels
        unique_classes = np.unique(labels_np)
        
        # Calculate class-specific statistics
        class_stats = {}
        for class_label in unique_classes:
            class_mask = labels_np == class_label
            if not np.any(class_mask):
                continue
                
            class_values = values_np[class_mask]
            class_stats[int(class_label)] = {
                'mean': float(np.mean(class_values)),
                'median': float(np.median(class_values)),
                'std': float(np.std(class_values)),
                'percentiles': {
                    'very_low': float(np.percentile(class_values, VERY_LOW_THRESHOLD)),
                    'low': float(np.percentile(class_values, LOW_THRESHOLD)),
                    'medium': float(np.percentile(class_values, MEDIUM_THRESHOLD)),
                    'high': float(np.percentile(class_values, HIGH_THRESHOLD)),
                    'very_high': float(np.percentile(class_values, VERY_HIGH_THRESHOLD))
                }
            }
        stats['class_stats'] = class_stats
        
        # Calculate correlation with class
        if len(unique_classes) == 2:  # Binary classification
            # Calculate point-biserial correlation (equivalent to Pearson for binary)
            try:
                correlation = float(np.corrcoef(values_np, labels_np)[0, 1])
                stats['correlation'] = correlation
                stats['correlation_type'] = get_correlation_type(correlation)
            except:
                stats['correlation'] = 0.0
                stats['correlation_type'] = 'none'
        else:  # Multi-class
            # Calculate correlation ratio (effect size / eta squared)
            try:
                class_means = np.array([np.mean(values_np[labels_np == c]) for c in unique_classes])
                class_counts = np.array([np.sum(labels_np == c) for c in unique_classes])
                
                overall_mean = np.mean(values_np)
                between_group_variance = np.sum(class_counts * (class_means - overall_mean)**2) / len(values_np)
                total_variance = np.var(values_np)
                
                correlation_ratio = between_group_variance / total_variance if total_variance > 0 else 0
                stats['correlation_ratio'] = float(correlation_ratio)
                stats['correlation_type'] = get_correlation_type(correlation_ratio)
            except:
                stats['correlation_ratio'] = 0.0
                stats['correlation_type'] = 'none'
    
    # Feature importance analysis
    # For causal features, determine which class this feature is most important for
    if is_causal and 'class_stats' in stats and len(stats['class_stats']) >= 2:
        class_means = {c: stats['class_stats'][c]['mean'] for c in stats['class_stats']}
        
        # Find the classes with min and max means
        min_class = min(class_means, key=class_means.get)
        max_class = max(class_means, key=class_means.get)
        
        # Calculate separation (how distinct the classes are based on this feature)
        if len(class_means) == 2:
            class_ids = list(class_means.keys())
            stats['separation'] = abs(class_means[class_ids[0]] - class_means[class_ids[1]]) / stats['std'] if stats['std'] > 0 else 0
        else:
            stats['separation'] = (stats['max'] - stats['min']) / stats['std'] if stats['std'] > 0 else 0
        
        # Determine feature importance for each class
        stats['importance_by_class'] = {}
        overall_mean = stats['mean']
        
        for class_id, class_mean in class_means.items():
            # How far is this class's mean from the overall mean (in std devs)
            if stats['std'] > 0:
                z_score = abs(class_mean - overall_mean) / stats['std']
                if z_score > 2.0:
                    importance = 'very_high'
                elif z_score > 1.5:
                    importance = 'high'
                elif z_score > 1.0:
                    importance = 'moderate'
                elif z_score > 0.5:
                    importance = 'low'
                else:
                    importance = 'very_low'
            else:
                importance = 'unknown'
                
            stats['importance_by_class'][class_id] = {
                'importance': importance,
                'z_score': float(z_score) if stats['std'] > 0 else 0.0,
                'relative_position': 'high' if class_mean > overall_mean else 'low'
            }
    
    # Add feature name for reference
    stats['feature_name'] = feature_name
    
    return stats

def get_correlation_type(corr_value: float) -> str:
    """Determine the type of correlation based on the correlation value."""
    abs_corr = abs(corr_value)
    
    if abs_corr >= STRONG_CORRELATION:
        prefix = 'strong_'
    elif abs_corr >= MODERATE_CORRELATION:
        prefix = 'moderate_'
    elif abs_corr >= WEAK_CORRELATION:
        prefix = 'weak_'
    else:
        return 'no_correlation'
    
    if corr_value >= 0:
        return prefix + 'positive'
    else:
        return prefix + 'negative'

def analyze_categorical_feature(
    feature_values: torch.Tensor,
    class_labels: torch.Tensor,
    feature_name: str = "feature",
    is_causal: bool = False,
    num_categories: Optional[int] = None
) -> Dict[str, Any]:
    """
    Analyze a categorical feature to detect statistical properties and relationship to classes.
    
    Args:
        feature_values: Tensor of feature values (integers representing categories)
        class_labels: Tensor of class labels
        feature_name: Name of the feature
        is_causal: Whether the feature is causal
        num_categories: Number of categories if known (otherwise inferred)
        
    Returns:
        Dictionary of statistical properties and their associations with classes
    """
    stats = {}
    
    # Ensure tensors are on the same device
    if feature_values.device != class_labels.device:
        class_labels = class_labels.to(feature_values.device)
    
    # Convert to numpy for easier analysis
    values_np = feature_values.detach().cpu().numpy().flatten()
    labels_np = class_labels.detach().cpu().numpy().flatten()
    
    # Filter out invalid values
    valid_mask = np.isfinite(values_np) & np.isfinite(labels_np)
    values_np = values_np[valid_mask].astype(int)
    labels_np = labels_np[valid_mask].astype(int)
    
    if len(values_np) == 0:
        return {"error": "No valid values for analysis"}
    
    # Determine categories
    if num_categories is None:
        categories = np.unique(values_np)
        num_categories = len(categories)
    else:
        categories = np.arange(num_categories)
    
    # Basic statistics
    stats['num_categories'] = num_categories
    stats['mode'] = int(np.argmax(np.bincount(values_np)))
    
    # Calculate category frequencies
    category_counts = {}
    for cat in categories:
        count = np.sum(values_np == cat)
        category_counts[int(cat)] = int(count)
    stats['category_counts'] = category_counts
    
    # For causal features, analyze relationship with classes
    if is_causal:
        # Get unique class labels
        unique_classes = np.unique(labels_np)
        
        # Calculate conditional probabilities: P(class | category)
        conditional_probs = {}
        for cat in categories:
            cat_mask = values_np == cat
            cat_count = np.sum(cat_mask)
            
            if cat_count > 0:
                class_counts = {}
                for class_id in unique_classes:
                    count = np.sum((values_np == cat) & (labels_np == class_id))
                    prob = float(count / cat_count)
                    class_counts[int(class_id)] = prob
                
                conditional_probs[int(cat)] = class_counts
        
        stats['conditional_probs'] = conditional_probs
        
        # Calculate mutual information and normalized mutual information
        try:
            # Calculate mutual information
            mi = 0.0
            for cat in categories:
                cat_prob = np.mean(values_np == cat)
                if cat_prob > 0:
                    for class_id in unique_classes:
                        class_prob = np.mean(labels_np == class_id)
                        if class_prob > 0:
                            joint_prob = np.mean((values_np == cat) & (labels_np == class_id))
                            if joint_prob > 0:
                                mi += joint_prob * np.log(joint_prob / (cat_prob * class_prob))
            
            # Calculate entropy of feature
            h_feature = 0.0
            for cat in categories:
                cat_prob = np.mean(values_np == cat)
                if cat_prob > 0:
                    h_feature -= cat_prob * np.log(cat_prob)
            
            # Calculate entropy of class
            h_class = 0.0
            for class_id in unique_classes:
                class_prob = np.mean(labels_np == class_id)
                if class_prob > 0:
                    h_class -= class_prob * np.log(class_prob)
            
            # Normalized mutual information
            if min(h_feature, h_class) > 0:
                nmi = mi / min(h_feature, h_class)
            else:
                nmi = 0.0
                
            stats['mutual_information'] = float(mi)
            stats['normalized_mutual_information'] = float(nmi)
            stats['correlation_type'] = get_correlation_type(nmi)
        except:
            stats['mutual_information'] = 0.0
            stats['normalized_mutual_information'] = 0.0
            stats['correlation_type'] = 'no_correlation'
        
        # Find the most predictive categories for each class
        predictive_categories = {}
        for class_id in unique_classes:
            class_categories = []
            for cat, probs in conditional_probs.items():
                if class_id in probs and probs[class_id] > 0.5:
                    class_categories.append((cat, probs[class_id]))
            
            # Sort by probability descending
            class_categories.sort(key=lambda x: x[1], reverse=True)
            predictive_categories[int(class_id)] = class_categories
        
        stats['predictive_categories'] = predictive_categories
    
    # Add feature name for reference
    stats['feature_name'] = feature_name
    
    return stats

def map_numeric_stats_to_terms(
    stats: Dict[str, Any],
    feature_name: str,
    class_id: int
) -> List[str]:
    """
    Map statistical properties of numerical features to semantic terms for a specific class.
    
    Args:
        stats: Statistics dictionary from analyze_numerical_feature
        feature_name: Name of the feature
        class_id: Class ID to generate terms for
        
    Returns:
        List of semantic terms relevant to this class for this feature
    """
    terms = []
    
    # If feature isn't causal or no class-specific stats, return empty list
    if not stats.get('class_stats') or class_id not in stats['class_stats']:
        return terms
    
    # Get class and overall statistics
    class_stats = stats['class_stats'][class_id]
    overall_mean = stats['mean']
    overall_median = stats['median']
    class_mean = class_stats['mean']
    
    # Basic relative position terms
    if class_mean > overall_mean:
        basic_term = f"high {feature_name}"
        terms.append(basic_term)
        terms.append(f"above average {feature_name}")
        
        # Add more specific magnitude terms based on percentiles
        if class_mean > stats['percentiles']['very_high']:
            terms.append(f"very high {feature_name}")
            terms.append(f"extremely high {feature_name}")
            terms.append(f"maximum {feature_name}")
        elif class_mean > stats['percentiles']['high']:
            terms.append(f"high {feature_name}")
            terms.append(f"elevated {feature_name}")
    else:
        basic_term = f"low {feature_name}"
        terms.append(basic_term)
        terms.append(f"below average {feature_name}")
        
        # Add more specific magnitude terms based on percentiles
        if class_mean < stats['percentiles']['very_low']:
            terms.append(f"very low {feature_name}")
            terms.append(f"extremely low {feature_name}")
            terms.append(f"minimum {feature_name}")
        elif class_mean < stats['percentiles']['low']:
            terms.append(f"low {feature_name}")
            terms.append(f"reduced {feature_name}")
    
    # Add trend terms if correlation information exists
    if 'correlation_type' in stats:
        corr_type = stats['correlation_type']
        
        if corr_type == 'strong_positive' and class_id == 1:
            terms.append(f"increasing {feature_name}")
            terms.append(f"directly correlated with {feature_name}")
        elif corr_type == 'strong_negative' and class_id == 1:
            terms.append(f"decreasing {feature_name}")
            terms.append(f"inversely correlated with {feature_name}")
        elif corr_type == 'moderate_positive' and class_id == 1:
            terms.append(f"moderately increasing {feature_name}")
        elif corr_type == 'moderate_negative' and class_id == 1:
            terms.append(f"moderately decreasing {feature_name}")
    
    # Add standard deviation / distribution terms
    if 'std' in class_stats and class_stats['std'] > stats['std'] * 1.2:
        terms.append(f"variable {feature_name}")
        terms.append(f"wide range of {feature_name}")
    elif 'std' in class_stats and class_stats['std'] < stats['std'] * 0.8:
        terms.append(f"consistent {feature_name}")
        terms.append(f"narrow range of {feature_name}")
    
    # Add important feature terms
    if 'importance_by_class' in stats and class_id in stats['importance_by_class']:
        importance = stats['importance_by_class'][class_id]['importance']
        if importance in ['high', 'very_high']:
            terms.append(f"characteristic {feature_name}")
            terms.append(f"defining {feature_name}")
    
    # Add specific numeric range if appropriate
    if abs(class_stats['mean'] - overall_mean) > stats['std']:
        # Format nicely with appropriate precision
        if abs(class_stats['mean']) < 0.01:
            mean_str = f"{class_stats['mean']:.6f}"
        elif abs(class_stats['mean']) < 1:
            mean_str = f"{class_stats['mean']:.4f}"
        elif abs(class_stats['mean']) < 10:
            mean_str = f"{class_stats['mean']:.2f}"
        else:
            mean_str = f"{int(class_stats['mean'])}"
            
        terms.append(f"{feature_name} around {mean_str}")
    
    return terms

def map_categorical_stats_to_terms(
    stats: Dict[str, Any],
    feature_name: str,
    class_id: int
) -> List[str]:
    """
    Map statistical properties of categorical features to semantic terms for a specific class.
    
    Args:
        stats: Statistics dictionary from analyze_categorical_feature
        feature_name: Name of the feature
        class_id: Class ID to generate terms for
        
    Returns:
        List of semantic terms relevant to this class for this feature
    """
    terms = []
    
    # If no predictive categories for this class, return empty list
    if 'predictive_categories' not in stats or class_id not in stats['predictive_categories']:
        return terms
    
    # Get predictive categories for this class
    pred_categories = stats['predictive_categories'][class_id]
    
    # If there are strong predictive categories, use them
    if pred_categories and len(pred_categories) > 0:
        # Get the top category and its probability
        top_cat, top_prob = pred_categories[0]
        
        # Format the category name
        cat_name = f"category_{top_cat}"
        
        # Basic association term
        terms.append(f"{feature_name} {cat_name}")
        
        # Add strength terms based on probability
        if top_prob > 0.9:
            terms.append(f"strong {feature_name} {cat_name}")
            terms.append(f"almost always {feature_name} {cat_name}")
        elif top_prob > 0.7:
            terms.append(f"frequent {feature_name} {cat_name}")
            terms.append(f"usually {feature_name} {cat_name}")
        elif top_prob > 0.5:
            terms.append(f"common {feature_name} {cat_name}")
            terms.append(f"often {feature_name} {cat_name}")
    
    # Add importance terms based on mutual information
    if 'normalized_mutual_information' in stats:
        nmi = stats['normalized_mutual_information']
        if nmi > 0.8:
            terms.append(f"highly predictive {feature_name}")
        elif nmi > 0.5:
            terms.append(f"moderately predictive {feature_name}")
        elif nmi > 0.2:
            terms.append(f"somewhat predictive {feature_name}")
    
    return terms

def get_class_description_from_stats(
    feature_stats: Dict[str, Dict[str, Any]],
    class_id: int
) -> List[str]:
    """
    Generate class description terms from statistical properties of all features.
    
    Args:
        feature_stats: Dictionary mapping feature indices to their statistics
        class_id: Class ID to generate terms for
        
    Returns:
        List of semantic terms relevant to this class across all features
    """
    class_terms = []
    
    # Process each feature
    for feature_idx, stats in feature_stats.items():
        # Skip if this feature doesn't have a strong relationship with the class
        if not stats.get('is_causal', False):
            continue
            
        feature_name = stats.get('feature_name', f"feature_{feature_idx}")
        
        # Apply the appropriate mapping based on feature type
        if stats.get('type') == 'categorical':
            feature_terms = map_categorical_stats_to_terms(stats, feature_name, class_id)
        else:  # default to numerical
            feature_terms = map_numeric_stats_to_terms(stats, feature_name, class_id)
        
        # Add the terms to our list
        class_terms.extend(feature_terms)
    
    return class_terms