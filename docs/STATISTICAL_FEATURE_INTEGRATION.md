# Statistical Feature Integration in Semantic Models

This document explains how numerical features are integrated with semantic understanding in the model.

## Overview

In many tabular datasets, numerical features (like age, income, etc.) have inherent semantic meaning, but this meaning is defined by their statistical distribution rather than just their presence/absence. The Statistical Feature Integration system enhances the model's ability to understand numerical features by:

1. Detecting which features are numerical
2. Calculating statistical properties of these features (min, max, mean, quantiles, etc.)
3. Encoding these features as special tokens that represent their value ranges
4. Training the model to associate these statistical patterns with semantic meaning

## How It Works

### 1. Feature Statistics Calculation

During both training and inference, the system:
- Identifies numerical features based on their distribution patterns
- Calculates key statistics for each feature
- Determines appropriate value ranges (low, medium, high) using quartiles

### 2. Statistical Token Encoding

- Three reserved "semantic features" are used to encode statistical information
- Numerical values are converted to tokens representing their position in the distribution
- Token ranges are consistent to ensure the model can learn meaningful associations:
  - First feature: 10 (low), 20 (medium), 30 (high)
  - Second feature: 40 (low), 50 (medium), 60 (high)
  - Additional metadata in third feature

### 3. Contrastive Statistical Loss

A distribution-aware contrastive loss component ensures that:
- Similar numerical ranges have similar embeddings
- Different numerical ranges have distinct embeddings
- These embeddings align correctly with text descriptions

## Usage

To enable and control statistical feature integration:

1. Ensure semantic features are enabled (semantic_feature_p > 0)
2. Configure the loss weights for the statistical component:

```python
# In configuration:
config = {
    # ... other settings
    "semantic_feature_p": 0.8,  # Enable semantic features
    "semantic_weight": 0.2,     # Weight for general semantic feature alignment
    "statistical_weight": 0.1   # Weight for statistical feature alignment
}
```

## Benefits

- Better understanding of numerical features beyond simple presence/absence
- More meaningful zero-shot generalization to new datasets with similar numeric patterns
- Improved alignment between numeric data and textual descriptions
- Enhanced model interpretability by connecting numbers to their semantic meaning

## Implementation Details

The integration happens in several components:

1. `/ticl/datasets/statistical_feature_encoding.py`: Core utilities for encoding numeric features
2. `/ticl/priors/classification_adapter.py`: Integration during data generation and training
3. `/ticl/models/semantic_aware_model.py`: Enhanced loss function with statistical component
4. `/ticl/prediction/semantic.py`: Integration during inference

## Example

When a dataset has a numeric feature like "age" that ranges from 25 to 85, the system:

1. Creates quantiles (e.g., 35, 60) to divide values into "young", "middle-aged", and "elderly"
2. Encodes values in these ranges with tokens 10, 20, or 30 respectively
3. Trains the model to associate these tokens with their semantic meaning
4. During inference, uses this understanding to correctly interpret new age values

This allows the model to develop a more nuanced understanding of what numeric values mean in context.