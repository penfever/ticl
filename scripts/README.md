# Utility Scripts

This directory contains various utility scripts for analyzing, testing, and visualizing the TiCL framework.

## Performance Testing Scripts

- `test_semantic_performance.py`: Benchmark and profile the semantic feature generation process 
- `test_high_feature_count.py`: Test performance with large numbers of features
- `test_item_bottleneck.py`: Test performance of `.item()` call optimization
- `profile_model_performance.py`: General model performance profiling

## Visualization Scripts

- `semantic_feature_analysis.py`: Analyze and visualize semantic features 
- `visualize_semantic_contrastive.py`: Create CLIP-like visualizations of token-class relationships
- `test_contrastive_data.py`: Test and visualize contrastive data

## Data Processing Scripts

- `update_process_columns.py`: Process and update column data
- `update_script.py`: General update utility script

## Usage

Most scripts support command line arguments. To see available options:

```bash
python -m scripts.<script_name> --help
```

For example:

```bash
python -m scripts.test_semantic_performance --help
```

## Output Location

All output files are saved to the `logs` directory:

- Runtime logs: `logs/semantic_performance.log`, etc.
- Visualizations: `logs/visualizations/`
- Training outputs: `logs/training/`