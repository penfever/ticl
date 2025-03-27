"""
Utility functions for TabFlex testing
"""
import torch

def get_nan_value(set_to_nan):
    """Get a NaN value based on the configuration"""
    if set_to_nan == 'nan':
        return float('nan')
    elif set_to_nan == 'zero':
        return 0.0
    else:
        return float('nan')

def normalize_by_used_features_f(x, num_features_used, num_features):
    """Scale features based on used/total feature ratio"""
    return x * (num_features / num_features_used) ** 0.5

def normalize_data(x):
    """Simple data normalization"""
    return x

def remove_outliers(x, categorical_features=None):
    """Remove outliers from data"""
    if categorical_features is None:
        categorical_features = []
    return x