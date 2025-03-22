# Statistical Column Mapping

This document describes the statistical column mapping system used in the TiCL framework to enhance semantic understanding of numerical features.

## Overview

Statistical column mapping is a feature that assigns meaningful column names to statistical features and generates causal relationships between feature values and target classes. This helps the model better understand the semantic meaning of numerical features and their relationship to classification tasks.

## Components

The system consists of the following key components:

1. **StatisticalColumnMapper**: A class that maps statistical features to column names from labeled_numeric_prior_data_loader and generates value range descriptions.

2. **Enhanced Semantic Column Generation**: Extended with a new curation strategy 'statistical_causal_relationships' that uses LLM prompting to generate causal relationships between statistical feature values and target classes.

3. **Statistical Feature Encoding**: Updated to include column names in the token mapping information.

## Workflow

The system works as follows:

1. Numeric features are identified and their statistics are calculated.

2. The `StatisticalColumnMapper` assigns meaningful column names (e.g., "age", "income", "price") to these numerical features based on statistical properties.

3. For each column, value ranges are identified (e.g., "low age (0-18 years)", "medium age (19-65 years)", "high age (66-100 years)").

4. The enhanced semantic column generation generates causal relationships between these value ranges and potential target classes using LLM prompting.

5. These causal relationships are stored and can be used for:
   - Enhancing model understanding of numeric features
   - Generating semantic embeddings for contrastive learning
   - Improving interpretability of model predictions

## Usage Example

```python
from ticl.datasets.statistical_column_mapping import StatisticalColumnMapper
from ticl.datasets.statistical_feature_encoding import calculate_feature_statistics
from ticl.datasets.enhanced_semantic_column_generation import EnhancedColumnSemanticTokenizer

# Calculate feature statistics
feature_stats = calculate_feature_statistics(X)

# Create a column mapper
column_mapper = StatisticalColumnMapper()

# Assign column names to features
column_mapping = column_mapper.assign_column_names(feature_stats)

# Initialize tokenizer with statistical_causal_relationships strategy
tokenizer = EnhancedColumnSemanticTokenizer(
    provider="gemini",  # or another provider
    curation_strategy="statistical_causal_relationships",
    stat_causal_path="causal_relationships.json"
)

# For a specific column
column_name = column_mapping[0]  # Get column name for feature index 0
causal_relationships = tokenizer.generate_statistical_causal_relationships(column_name)

# Process the column to get tokenized values
tokenized_values = tokenizer.process_column(column_name)
```

## Causal Relationship Structure

The causal relationships are stored in a JSON format with the following structure:

```json
{
  "column_name": {
    "low column_name (value_range)": [
      {"class": "class_name_1", "causal_explanation": "Explanation of causal mechanism"},
      {"class": "class_name_2", "causal_explanation": "Explanation of causal mechanism"}
    ],
    "medium column_name (value_range)": [
      {"class": "class_name_3", "causal_explanation": "Explanation of causal mechanism"},
      {"class": "class_name_4", "causal_explanation": "Explanation of causal mechanism"}
    ],
    "high column_name (value_range)": [
      {"class": "class_name_5", "causal_explanation": "Explanation of causal mechanism"},
      {"class": "class_name_6", "causal_explanation": "Explanation of causal mechanism"}
    ]
  }
}
```

## Benefits

1. **Improved Semantic Understanding**: The model can better understand the semantic meaning of numerical features through their column names and value ranges.

2. **Enhanced Contrastive Learning**: The causal relationships provide a rich source of semantic information for contrastive learning.

3. **Better Interpretability**: The causal relationships make model predictions more interpretable by explaining how specific feature values contribute to class predictions.

4. **Bridge Between Numeric and Semantic Features**: Helps bridge the gap between numerical features and text-based semantic understanding.

## Command Line Usage

```bash
python -m ticl.datasets.enhanced_semantic_column_generation \
  --provider gemini \
  --curation-strategy statistical_causal_relationships \
  --stat-causal-path causal_relationships.json \
  --output-file tokenized_statistical_features.pt
```