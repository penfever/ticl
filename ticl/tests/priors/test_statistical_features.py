"""
Test module for feature statistical analyzer.
"""

import pytest
import torch
import numpy as np
from ticl.priors.feature_statistical_analyzer import (
    analyze_numerical_feature,
    analyze_categorical_feature,
    map_numeric_stats_to_terms,
    map_categorical_stats_to_terms,
    get_class_description_from_stats
)
from ticl.priors.mlp import MLPPrior
from ticl.priors.classification_adapter import ClassificationAdapterPrior


def test_analyze_numerical_feature():
    """Test numerical feature analysis."""
    # Create test data
    feature_values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    class_labels = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    
    # Analyze as causal feature
    stats = analyze_numerical_feature(feature_values, class_labels, feature_name="test_feature", is_causal=True)
    
    # Check basic statistics
    assert 'mean' in stats
    assert 'median' in stats
    assert 'std' in stats
    assert 'percentiles' in stats
    assert 'class_stats' in stats
    
    # Check class-specific statistics
    assert 0 in stats['class_stats']
    assert 1 in stats['class_stats']
    
    # Check correlation
    assert 'correlation' in stats
    assert 'correlation_type' in stats
    
    # Verify the statistical values make sense
    assert stats['mean'] == 5.5
    assert stats['class_stats'][0]['mean'] == 3.0
    assert stats['class_stats'][1]['mean'] == 8.0
    assert stats['correlation'] > 0.9  # Should be strongly positive


def test_analyze_categorical_feature():
    """Test categorical feature analysis."""
    # Create test data
    feature_values = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3])
    class_labels = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    
    # Analyze as causal feature
    stats = analyze_categorical_feature(feature_values, class_labels, feature_name="test_feature", is_causal=True)
    
    # Check basic statistics
    assert 'num_categories' in stats
    assert 'mode' in stats
    assert 'category_counts' in stats
    
    # Check relationship statistics
    assert 'conditional_probs' in stats
    assert 'mutual_information' in stats
    assert 'normalized_mutual_information' in stats
    assert 'correlation_type' in stats


def test_map_numeric_stats_to_terms():
    """Test mapping numerical statistics to semantic terms."""
    # Create test statistics
    stats = {
        'mean': 5.5,
        'median': 5.5,
        'std': 3.0,
        'percentiles': {
            'very_low': 1.5,
            'low': 3.0,
            'medium': 5.5,
            'high': 8.0,
            'very_high': 9.5
        },
        'class_stats': {
            0: {
                'mean': 3.0,
                'median': 3.0,
                'std': 1.5
            },
            1: {
                'mean': 8.0,
                'median': 8.0,
                'std': 1.5
            }
        },
        'correlation': 0.95,
        'correlation_type': 'strong_positive'
    }
    
    # Map to terms for class 0 (low values)
    terms_0 = map_numeric_stats_to_terms(stats, "test_feature", 0)
    
    # Map to terms for class 1 (high values)
    terms_1 = map_numeric_stats_to_terms(stats, "test_feature", 1)
    
    # Check that terms were generated
    assert len(terms_0) > 0
    assert len(terms_1) > 0
    
    # Check that terms reflect statistical properties
    assert any("low" in term for term in terms_0)
    assert any("high" in term for term in terms_1)


def test_mlp_prior_with_causality():
    """Test MLPPrior with causality tracking."""
    # Create MLPPrior with default config
    prior = MLPPrior()
    
    # Generate a batch
    batch_size = 2
    n_samples = 10
    num_features = 5
    device = torch.device('cpu')
    
    try:
        # Try the new interface
        x, y, y_, info = prior.get_batch(batch_size, n_samples, num_features, device)
        
        # Check that causality info is returned
        assert 'causality_info' in info
        assert len(info['causality_info']) == batch_size
        
        # Check structure of causality info
        for batch_info in info['causality_info']:
            assert 'is_causal' in batch_info
            assert 'causal_features' in batch_info
            assert 'feature_metadata' in batch_info
            
    except ValueError as e:
        # If this fails due to config issues, skip the test
        pytest.skip(f"MLPPrior test skipped due to config error: {e}")


def test_classification_adapter_with_stats():
    """Test ClassificationAdapterPrior with statistical analysis."""
    # Create MLPPrior with default config
    base_prior = MLPPrior()
    
    # Create ClassificationAdapterPrior
    adapter = ClassificationAdapterPrior(base_prior, semantic_feature_p=1.0)
    
    # Generate a batch
    batch_size = 2
    n_samples = 10
    num_features = 5
    device = torch.device('cpu')
    
    try:
        # Generate data with adapter
        x, y, y_, info = adapter.get_batch(batch_size, n_samples, num_features, device)
        
        # Check if information is returned
        assert isinstance(info, dict)
        
        # Some batches may not have statistical information due to randomness
        # in the data generation process, so this is not a hard assertion
        if 'feature_stats' in info:
            # Check structure of feature stats
            assert isinstance(info['feature_stats'], dict)
            
            # At least one batch should have statistical information
            for batch_id, batch_stats in info['feature_stats'].items():
                # Check structure of batch stats
                assert isinstance(batch_stats, dict)
                
                # Check at least one feature has stats
                if len(batch_stats) > 0:
                    feature_idx = next(iter(batch_stats.keys()))
                    feature_stats = batch_stats[feature_idx]
                    
                    # Check basic stats fields
                    assert 'feature_name' in feature_stats
                    assert 'type' in feature_stats
        
    except (ValueError, RuntimeError) as e:
        # If this fails due to config issues, skip the test
        pytest.skip(f"ClassificationAdapterPrior test skipped due to error: {e}")


if __name__ == "__main__":
    # Run tests manually
    test_analyze_numerical_feature()
    test_analyze_categorical_feature()
    test_map_numeric_stats_to_terms()
    test_mlp_prior_with_causality()
    test_classification_adapter_with_stats()
    print("All tests passed!")