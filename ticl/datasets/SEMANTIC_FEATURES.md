# Semantic Feature Generation and Integration

This directory contains utilities for generating, testing, and integrating different types of semantic features for tabular models. The core purpose is to establish meaningful causal relationships between column values and their semantic representations.

## Overview

The tools in this directory enable:

1. Generating semantic features using different curation strategies
2. Testing different semantic feature generation approaches
3. Integrating multiple feature types to create enhanced causal relationships

## Available Curation Strategies

The following curation strategies are available:

- **standard**: Basic approach that generates values likely to appear in a column
- **numeric_properties**: Focuses on statistical properties of numeric data (magnitude, distribution, trends)
- **causal_relationships**: Explicitly models causal relationships involving the column
- **conceptual_clusters**: Groups related terms into semantic clusters
- **contrastive_pairs**: Creates pairs of semantically opposite concepts

## Usage

### 1. Generate Semantic Features

Use the enhanced semantic column generation script to create feature sets with different curation strategies:

```bash
python -m ticl.datasets.enhanced_semantic_column_generation \
  --provider gemini \
  --curation-strategy numeric_properties \
  --output-file numeric_properties_features.pt \
  --parallel
```

Available strategies: `standard`, `numeric_properties`, `causal_relationships`, `conceptual_clusters`, `contrastive_pairs`

### 2. Test Curation Strategies

Test different curation strategies on a small set of sample columns:

```bash
python -m ticl.datasets.test_curation_strategies \
  --provider gemini \
  --strategies numeric_properties causal_relationships \
  --output-dir ./test_results \
  --verbose
```

This generates sample outputs for each strategy to help you evaluate which approaches work best.

### 3. Integrate Feature Sets

Combine multiple feature sets to create enhanced semantic representations:

```bash
python -m ticl.datasets.semantic_feature_integration \
  --feature-files standard_features.pt numeric_properties_features.pt \
  --output-file combined_features.pt \
  --combination-method weighted_sum \
  --weights 0.3 0.7
```

Combination methods:
- `weighted_sum`: Weighted combination of feature sets
- `concatenate`: Concatenates portions of each feature set
- `alternate`: Alternates tokens from different feature sets

## Training with Enhanced Features

To train a model with the enhanced semantic features, use:

```bash
python -m ticl.fit_model tabpfn -E 1 --semantic-feature-p 0.3 -n 100 -b 4 \
  --progress-bar -U 1 --reduce-lr-on-spike True --validate True \
  --semantic-features-file combined_features.pt
```

## Concept

The key concept behind these different curation strategies is to establish more meaningful causal relationships between column values and their semantic representations.

For example:
- A column named "temperature" with `numeric_properties` strategy will have features like "increasing", "high", "above average"
- A column named "medical_condition" with `causal_relationships` strategy will have features like "caused by virus", "leads to fever"

This creates a richer semantic space that helps the model better understand the causal nature of the data.