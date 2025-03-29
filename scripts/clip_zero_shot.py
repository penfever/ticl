#!/usr/bin/env python3
"""
Zero-shot classification of tabular data using CLIP text embeddings.

This script computes cosine similarity between CLIP text embeddings of:
1. Class names from each dataset
2. Text representations of each row in the dataset

It then evaluates the accuracy of the classification based on these similarities.
"""

import os
import sys
import json
import logging
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report
from transformers import CLIPTokenizer, CLIPTextModel
from tqdm import tqdm
import glob

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("clip_zero_shot")

def setup_clip_model(device='cpu', checkpoint_path=None):
    """Set up CLIP text model and tokenizer on the specified device.
    
    Parameters:
    -----------
    device : str
        Device to run model on (cpu, cuda, mps, or auto)
    checkpoint_path : str, optional
        Path to a saved model checkpoint to load weights from
        
    Returns:
    --------
    tuple
        (model, tokenizer) - The loaded CLIP text model and tokenizer
    """
    logger.info(f"Setting up CLIP text model on {device}")
    
    # Load pretrained model and tokenizer
    model_name = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    model = CLIPTextModel.from_pretrained(model_name)
    
    # Move model to appropriate device
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    # Load weights from checkpoint if provided
    if checkpoint_path:
        try:
            logger.info(f"Loading model weights from checkpoint: {checkpoint_path}")
            # Check if checkpoint exists
            if not os.path.exists(checkpoint_path):
                logger.error(f"Checkpoint file not found: {checkpoint_path}")
                logger.warning("Falling back to pretrained weights")
            else:
                # Load checkpoint
                checkpoint = torch.load(checkpoint_path, map_location=device)
                
                # Based on examination, the checkpoint is a tuple with an OrderedDict as first element
                if isinstance(checkpoint, tuple) and len(checkpoint) > 0 and isinstance(checkpoint[0], dict):
                    state_dict = checkpoint[0]
                    
                    # Look for CLIP text model weights in the state dict
                    clip_weights = {}
                    clip_prefix = 'clip_text_model.text_model.'
                    
                    # Filter weights from checkpoint
                    for key, value in state_dict.items():
                        # Check if this is a CLIP text model weight
                        if key.startswith(clip_prefix):
                            # Remove the prefix to match the CLIP model's state dict keys
                            # The structure is clip_text_model.text_model.XXX, so we remove
                            # clip_text_model. to get text_model.XXX
                            clip_key = key[len('clip_text_model.'):]
                            clip_weights[clip_key] = value
                    
                    if clip_weights:
                        # Load weights into model
                        missing_keys, unexpected_keys = model.load_state_dict(clip_weights, strict=False)
                        logger.info(f"Successfully loaded CLIP text model weights from checkpoint")
                        logger.info(f"Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
                    else:
                        logger.warning(f"No CLIP text model weights found in checkpoint")
                else:
                    logger.warning(f"Checkpoint structure not as expected. Found type: {type(checkpoint)}")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            import traceback
            traceback.print_exc()
            logger.warning("Falling back to pretrained weights")
    
    # Disable gradient calculation for inference
    for param in model.parameters():
        param.requires_grad = False
        
    logger.info(f"Successfully initialized CLIP text model")
    return model, tokenizer

def get_text_embedding(text, model, tokenizer, device='cpu'):
    """Get text embedding from CLIP model."""
    # Tokenize text
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=77,
        return_tensors="pt"
    ).to(device)
    
    # Get embedding
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.pooler_output
        
    # Normalize embeddings for cosine similarity
    normalized_embeddings = F.normalize(embeddings, dim=1)
    
    return normalized_embeddings

def load_dataset(dataset_path, format=None):
    """Load dataset from file path."""
    if format is None:
        # Determine format from file extension
        if dataset_path.endswith('.csv'):
            format = 'csv'
        elif dataset_path.endswith('.json'):
            format = 'json'
        elif dataset_path.endswith('.parquet'):
            format = 'parquet'
        else:
            raise ValueError(f"Unknown file format for {dataset_path}")
    
    # Load dataset based on format
    if format == 'csv':
        return pd.read_csv(dataset_path)
    elif format == 'json':
        return pd.read_json(dataset_path)
    elif format == 'parquet':
        return pd.read_parquet(dataset_path)
    else:
        raise ValueError(f"Unsupported format: {format}")

def get_dataset_metadata(dataset_dir):
    """Get class column and class names from dataset metadata."""
    config_path = os.path.join(dataset_dir, 'config_data.json')
    
    # Use default metadata if configuration doesn't exist
    if not os.path.exists(config_path):
        logger.warning(f"No config found at {config_path}, using dataset name as class name")
        dataset_name = os.path.basename(dataset_dir)
        # Return default configuration
        return {
            'class_column': 'target',
            'class_names': None,
            'dataset_name': dataset_name
        }
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Extract class column and class names
    # In Carte datasets, the target column is specified in 'target_name'
    class_column = config.get('target_name', config.get('class_column', 'target'))
    class_names = config.get('class_names', None)
    dataset_name = config.get('dataset_name', os.path.basename(dataset_dir))
    
    # Also extract the entity name if available
    entity_name = config.get('entity_name', None)
    
    return {
        'class_column': class_column,
        'class_names': class_names,
        'dataset_name': dataset_name,
        'entity_name': entity_name,
        'task': config.get('task', 'classification')
    }

def row_to_text(row, include_columns=None, exclude_columns=None, entity_name=None, highlight_key_columns=None):
    """
    Convert a pandas Series (row) to formatted text.
    
    Parameters:
    -----------
    row : pandas.Series
        The row to convert
    include_columns : list, optional
        Only include these columns
    exclude_columns : list, optional
        Exclude these columns
    entity_name : str, optional
        Name of the entity column (will be highlighted in the text)
    highlight_key_columns : list, optional
        List of columns to highlight at the beginning of the text
    
    Returns:
    --------
    str
        Formatted text description of the row
    """
    # Start with entity name if available
    if entity_name and entity_name in row and not pd.isna(row[entity_name]):
        prefix = f"This is {row[entity_name]}, which has the following attributes: "
    else:
        prefix = "This item has the following attributes: "
    
    # Handle highlighted columns first if provided
    highlighted_parts = []
    if highlight_key_columns:
        for col in highlight_key_columns:
            if col in row and not pd.isna(row[col]):
                value = row[col]
                # Format value based on type
                if isinstance(value, (int, float)):
                    if float(value).is_integer():
                        value_str = str(int(value))
                    else:
                        value_str = f"{value:.4f}".rstrip('0').rstrip('.')
                else:
                    value_str = str(value)
                
                # Add to highlighted parts with special formatting
                highlighted_parts.append(f"{col.replace('_', ' ')} is {value_str}")
    
    # Create a highlighted section if needed
    highlighted_section = ""
    if highlighted_parts:
        highlighted_section = "Most importantly, " + ", and ".join(highlighted_parts) + ". "
    
    # Process remaining columns
    text_parts = []
    
    # Filter columns if needed
    columns = row.index
    if include_columns:
        columns = [col for col in columns if col in include_columns]
    if exclude_columns:
        columns = [col for col in columns if col not in exclude_columns]
    
    # Skip columns already in highlight section
    if highlight_key_columns:
        columns = [col for col in columns if col not in highlight_key_columns]
        
    # Skip entity name column as it's already included in the prefix
    if entity_name:
        columns = [col for col in columns if col != entity_name]
    
    # Format each column
    for col in columns:
        value = row[col]
        # Skip NaN values
        if pd.isna(value):
            continue
        # Format value based on type
        if isinstance(value, (int, float)):
            if float(value).is_integer():
                value_str = str(int(value))
            else:
                value_str = f"{value:.4f}".rstrip('0').rstrip('.')
        else:
            value_str = str(value)
        
        # Add to text parts with more natural language
        text_parts.append(f"its {col.replace('_', ' ')} is {value_str}")
    
    # Combine all parts
    details = ", ".join(text_parts)
    
    return prefix + highlighted_section + details

def get_class_texts(class_names, dataset_name=None):
    """Generate descriptive text for each class."""
    if isinstance(class_names, list):
        # If class_names is a list, use each name directly
        class_texts = {}
        for i, name in enumerate(class_names):
            class_texts[i] = f"This is the {name} class."
    elif isinstance(class_names, dict):
        # If class_names is a mapping, use provided mapping
        class_texts = {}
        for idx, name in class_names.items():
            idx_key = int(idx) if isinstance(idx, str) and idx.isdigit() else idx
            class_texts[idx_key] = f"This is the {name} class."
    else:
        # If class_names is None or unsupported type, generate generic class texts
        unique_classes = set()
        # If we know the dataset name, include it
        dataset_prefix = f"in the {dataset_name} dataset " if dataset_name else ""
        
        class_texts = {
            i: f"This is class {i} {dataset_prefix}representing category #{i}."
            for i in range(10)  # Support up to 10 classes by default
        }
    
    return class_texts

def get_datasets_from_carte(carte_dir):
    """Get all datasets from the Carte benchmark directory."""
    datasets = []
    
    # Check if carte_dir exists
    if not os.path.exists(carte_dir):
        logger.error(f"Carte benchmark directory not found: {carte_dir}")
        return datasets
    
    # Look only for subdirectories, not individual files
    for item in os.listdir(carte_dir):
        item_path = os.path.join(carte_dir, item)
        
        # Skip hidden files or non-directories
        if item.startswith('.') or not os.path.isdir(item_path):
            continue
        
        # Process directories
        dataset_name = os.path.basename(item_path)
        
        # Prefer to use CSV files if they exist
        csv_files = glob.glob(os.path.join(item_path, '*.csv'))
        if csv_files:
            # Try to find a CSV file with the same name as the directory
            main_csv = None
            for csv_file in csv_files:
                if os.path.basename(csv_file).startswith(dataset_name):
                    main_csv = csv_file
                    break
            
            # If no match, use the first CSV file
            if main_csv is None and csv_files:
                main_csv = csv_files[0]
                
            if main_csv:
                # Get dataset metadata
                metadata = get_dataset_metadata(item_path)
                metadata['data_path'] = main_csv
                metadata['dataset_dir'] = item_path
                metadata['dataset_name'] = dataset_name
                
                datasets.append(metadata)
                continue
        
        # Otherwise look for any data files
        data_files = []
        for ext in ['*.csv', '*.json', '*.parquet']:
            data_files.extend(glob.glob(os.path.join(item_path, ext)))
        
        if data_files:
            # Pick the first data file as the dataset
            data_file = data_files[0]
            
            # Get dataset metadata
            metadata = get_dataset_metadata(item_path)
            metadata['data_path'] = data_file
            metadata['dataset_dir'] = item_path
            metadata['dataset_name'] = dataset_name
            
            datasets.append(metadata)
    
    # Sort datasets by name for consistent ordering
    datasets = sorted(datasets, key=lambda x: x['dataset_name'])
    
    logger.info(f"Found {len(datasets)} datasets in Carte benchmark directory:")
    for i, dataset in enumerate(datasets):
        logger.info(f"  {i+1}. {dataset['dataset_name']} ({os.path.basename(dataset['data_path'])})")
    
    return datasets

def evaluate_dataset_with_clip(dataset_info, model, tokenizer, device='cpu'):
    """Evaluate a dataset using CLIP zero-shot classification."""
    # Load dataset
    logger.info(f"Loading dataset: {dataset_info['dataset_name']}")
    try:
        df = load_dataset(dataset_info['data_path'])
    except Exception as e:
        logger.error(f"Error loading dataset {dataset_info['dataset_name']}: {e}")
        return None
    
    # Get class column
    class_column = dataset_info['class_column']
    
    # Make sure class column exists
    if class_column not in df.columns:
        logger.error(f"Class column '{class_column}' not found in dataset {dataset_info['dataset_name']}")
        # Try to find a column with 'target' or 'class' in the name
        for col in df.columns:
            if 'target' in col.lower() or 'class' in col.lower() or 'label' in col.lower() or 'rating' in col.lower():
                logger.info(f"Using '{col}' as the class column instead")
                class_column = col
                break
        else:
            logger.error(f"Could not find a suitable class column. Available columns: {df.columns.tolist()}")
            return None
    
    # Extract labels
    y_true = df[class_column].values
    
    # Get unique classes
    unique_classes = sorted(df[class_column].unique())
    logger.info(f"Found {len(unique_classes)} unique classes: {unique_classes}")
    
    # Generate class texts
    if dataset_info['class_names'] is None:
        # Use unique values to generate class names
        dataset_info['class_names'] = {cls: str(cls) for cls in unique_classes}
    
    # If class_names is a dict but keys don't match unique_classes, remap it
    if isinstance(dataset_info['class_names'], dict):
        # Convert string keys to original data type if needed
        class_names_remapped = {}
        for key, value in dataset_info['class_names'].items():
            # Try to convert string keys to match the dtype in unique_classes
            if isinstance(key, str) and key.isdigit() and all(not isinstance(cls, str) for cls in unique_classes):
                try:
                    # Convert to int or float as needed
                    if all(isinstance(cls, int) for cls in unique_classes):
                        key = int(key)
                    else:
                        key = float(key)
                except:
                    pass
            
            class_names_remapped[key] = value
            
        dataset_info['class_names'] = class_names_remapped
    
    class_texts = get_class_texts(dataset_info['class_names'], dataset_info['dataset_name'])
    
    # Make sure we have text for each class
    for cls in unique_classes:
        if cls not in class_texts:
            class_texts[cls] = f"This item belongs to class {cls}."
    
    # Get class embeddings
    logger.info(f"Computing embeddings for {len(class_texts)} classes")
    class_embeddings = {}
    for cls, text in class_texts.items():
        # Print the class text so we can see what's being used
        logger.debug(f"Class {cls} text: {text}")
        embedding = get_text_embedding(text, model, tokenizer, device)
        class_embeddings[cls] = embedding
    
    # Create tensor with all class embeddings
    # Use a consistent sort function that can handle mixed types
    class_indices = list(class_embeddings.keys())
    
    # Convert all keys to strings for consistent sorting
    class_indices.sort(key=lambda x: str(x))
    
    class_embeddings_tensor = torch.cat([class_embeddings[cls] for cls in class_indices], dim=0)
    
    # Exclude the class column from row text
    exclude_columns = [class_column]
    
    # Get entity name and identify important columns
    entity_name = dataset_info.get('entity_name', None)
    
    # Identify important columns (features likely to be most relevant)
    important_columns = []
    column_importances = {}
    
    # Simple heuristic to identify important columns - those with likely descriptive content
    for col in df.columns:
        if col == class_column or col == entity_name:
            continue
            
        # Check column name for indicators of importance
        importance = 0
        keywords = ['name', 'desc', 'feature', 'description', 'category', 'type', 'quality']
        for keyword in keywords:
            if keyword in col.lower():
                importance += 5
        
        # Check for text columns, which are likely important
        if df[col].dtype == 'object' and df[col].notna().any():
            # Sample the first non-null value
            sample = df[col][df[col].notna()].iloc[0]
            if isinstance(sample, str) and len(sample) > 5:
                importance += 3
                
        # Check for known useful columns based on domain knowledge
        domain_columns = ['origin', 'roast', 'aroma', 'flavor', 'body', 'acid', 'aftertaste', 
                         'style', 'cuisine', 'specialties', 'category', 'genre']
        if col.lower() in domain_columns:
            importance += 4
            
        column_importances[col] = importance
    
    # Select top columns by importance
    important_columns = sorted(column_importances.keys(), 
                              key=lambda x: column_importances[x], 
                              reverse=True)[:5]  # Take top 5
    
    logger.info(f"Using entity column: {entity_name}")
    logger.info(f"Selected important columns: {important_columns}")
    
    # Process each row
    logger.info(f"Processing {len(df)} rows")
    predictions = []
    similarities_all = []
    row_texts_sample = []
    
    batch_size = 16  # Process rows in batches for efficiency
    for i in tqdm(range(0, len(df), batch_size)):
        batch_df = df.iloc[i:i+batch_size]
        batch_texts = []
        
        # Convert each row to text
        for _, row in batch_df.iterrows():
            row_text = row_to_text(
                row, 
                exclude_columns=exclude_columns, 
                entity_name=entity_name,
                highlight_key_columns=important_columns
            )
            batch_texts.append(row_text)
            
            # Save some sample texts for debugging
            if len(row_texts_sample) < 3:
                row_texts_sample.append(row_text)
        
        # Get embeddings for all texts in batch
        row_embeddings = torch.cat([
            get_text_embedding(text, model, tokenizer, device)
            for text in batch_texts
        ], dim=0)
        
        # Calculate similarities
        similarities = torch.matmul(row_embeddings, class_embeddings_tensor.t())
        
        # Store similarities for confidence analysis
        similarities_all.append(similarities.cpu())
        
        # Get predictions (indices of highest similarity)
        batch_preds = similarities.argmax(dim=1).cpu().numpy()
        
        # Convert indices back to class values
        batch_preds = [class_indices[idx] for idx in batch_preds]
        predictions.extend(batch_preds)
    
    # Print sample row texts for inspection
    logger.info("Sample row text representations:")
    for i, text in enumerate(row_texts_sample):
        logger.info(f"Row {i+1}: {text[:500]}...")
    
    # Convert predictions to numpy array
    y_pred = np.array(predictions)
    
    # Combine all similarities
    if similarities_all:
        similarities_tensor = torch.cat(similarities_all, dim=0)
        
        # Calculate confidence metrics
        softmax_probs = F.softmax(similarities_tensor, dim=1)
        top_probs, _ = torch.max(softmax_probs, dim=1)
        mean_confidence = top_probs.mean().item()
        
        # Get entropy of distribution (higher entropy = lower confidence)
        entropy = -torch.sum(softmax_probs * torch.log(softmax_probs + 1e-10), dim=1)
        mean_entropy = entropy.mean().item()
        
        # Calculate margin (difference between top two probabilities)
        sorted_probs, _ = torch.sort(softmax_probs, dim=1, descending=True)
        margins = sorted_probs[:, 0] - sorted_probs[:, 1]
        mean_margin = margins.mean().item()
    else:
        mean_confidence = 0.0
        mean_entropy = 0.0
        mean_margin = 0.0
    
    # Evaluate predictions
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'mean_confidence': mean_confidence,
        'mean_entropy': mean_entropy,
        'mean_margin': mean_margin,
        'report': classification_report(y_true, y_pred, output_dict=True)
    }
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Zero-shot classification of tabular data using CLIP")
    parser.add_argument('--carte-dir', type=str, default="benchmarks/carte",
                        help="Path to Carte benchmark directory")
    parser.add_argument('--dataset', type=str, default=None,
                        help="Specific dataset to evaluate (if None, evaluate all)")
    parser.add_argument('--device', type=str, default="auto",
                        help="Device to run model on (cpu, cuda, mps, or auto)")
    parser.add_argument('--output-file', type=str, default="clip_zero_shot_results.json",
                        help="Path to output file for results")
    parser.add_argument('--debug', action='store_true',
                        help="Enable debug logging")
    parser.add_argument('--save-checkpoints', action='store_true',
                        help="Save intermediate results after each dataset")
    parser.add_argument('--batch-size', type=int, default=16,
                        help="Batch size for processing rows")
    parser.add_argument('--checkpoint-path', type=str, 
                        default="/Users/benfeuer/Library/CloudStorage/GoogleDrive-penfever@gmail.com/My Drive/Current Papers/tabular-fm-llm/ticl/ticl/models_diff/tabpfn_semanticfeaturep0.3_03_27_2025_16_11_51_epoch_60.cpkt",
                        help="Path to model checkpoint for loading CLIP weights")
    args = parser.parse_args()
    
    # Set up logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Set device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Set up CLIP model
    model, tokenizer = setup_clip_model(device, args.checkpoint_path)
    
    # Get datasets
    carte_dir = os.path.expanduser(args.carte_dir)
    if not os.path.isabs(carte_dir):
        # Convert relative path to absolute
        current_dir = os.path.dirname(os.path.abspath(__file__))
        carte_dir = os.path.join(current_dir, carte_dir)
    
    datasets = get_datasets_from_carte(carte_dir)
    
    # Filter to specific dataset if requested
    if args.dataset:
        datasets = [d for d in datasets if d['dataset_name'] == args.dataset]
        if not datasets:
            logger.error(f"Dataset '{args.dataset}' not found.")
            logger.error(f"Available datasets: {', '.join([d['dataset_name'] for d in datasets])}")
            return 1
    
    # Create output directory if it doesn't exist
    if args.output_file:
        output_dir = os.path.dirname(args.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
    
    # Evaluate datasets
    results = {}
    for i, dataset_info in enumerate(datasets):
        logger.info(f"Evaluating dataset {i+1}/{len(datasets)}: {dataset_info['dataset_name']}")
        
        try:
            metrics = evaluate_dataset_with_clip(dataset_info, model, tokenizer, device)
            if metrics:
                results[dataset_info['dataset_name']] = metrics
                
                # Print results
                logger.info(f"Results for {dataset_info['dataset_name']}:")
                logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
                logger.info(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
                logger.info(f"  F1 (Macro): {metrics['f1_macro']:.4f}")
                logger.info(f"  F1 (Weighted): {metrics['f1_weighted']:.4f}")
                logger.info(f"  Mean Confidence: {metrics['mean_confidence']:.4f}")
                logger.info(f"  Mean Margin: {metrics['mean_margin']:.4f}")
                
                # Save checkpoint if requested
                if args.save_checkpoints and args.output_file:
                    checkpoint_file = f"{os.path.splitext(args.output_file)[0]}_checkpoint.json"
                    try:
                        # Make a deep copy of results for the checkpoint
                        checkpoint_results = {}
                        for dataset_name, dataset_metrics in results.items():
                            checkpoint_results[dataset_name] = dict(dataset_metrics)
                            if 'report' in checkpoint_results[dataset_name]:
                                checkpoint_results[dataset_name]['report'] = str(checkpoint_results[dataset_name]['report'])
                        
                        with open(checkpoint_file, 'w') as f:
                            json.dump(checkpoint_results, f, indent=2)
                        logger.info(f"Checkpoint saved to {checkpoint_file}")
                    except Exception as e:
                        logger.error(f"Error saving checkpoint: {e}")
        except Exception as e:
            logger.error(f"Error evaluating dataset {dataset_info['dataset_name']}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    if args.output_file:
        try:
            # Prepare results for saving
            serializable_results = {}
            for dataset_name, metrics in results.items():
                serializable_results[dataset_name] = dict(metrics)
                if 'report' in serializable_results[dataset_name]:
                    serializable_results[dataset_name]['report'] = str(serializable_results[dataset_name]['report'])
            
            with open(args.output_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            logger.info(f"Final results saved to {args.output_file}")
            
            # Also save a CSV summary for easy viewing
            csv_file = f"{os.path.splitext(args.output_file)[0]}_summary.csv"
            with open(csv_file, 'w') as f:
                # Write header
                f.write("Dataset,Accuracy,Balanced Accuracy,F1 (Macro),F1 (Weighted),Mean Confidence,Mean Margin\n")
                
                # Write data for each dataset
                for dataset_name, metrics in results.items():
                    f.write(f"{dataset_name},{metrics['accuracy']:.4f},{metrics['balanced_accuracy']:.4f},"
                           f"{metrics['f1_macro']:.4f},{metrics['f1_weighted']:.4f},"
                           f"{metrics['mean_confidence']:.4f},{metrics['mean_margin']:.4f}\n")
                
            logger.info(f"Summary CSV saved to {csv_file}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            import traceback
            traceback.print_exc()
    
    # Print overall summary
    logger.info("\n===== OVERALL RESULTS =====")
    if results:
        # Calculate averages
        avg_acc = sum(m['accuracy'] for m in results.values()) / len(results)
        avg_bal_acc = sum(m['balanced_accuracy'] for m in results.values()) / len(results)
        avg_f1_macro = sum(m['f1_macro'] for m in results.values()) / len(results)
        avg_f1_weighted = sum(m['f1_weighted'] for m in results.values()) / len(results)
        
        logger.info(f"Evaluated {len(results)}/{len(datasets)} datasets")
        logger.info(f"Average Accuracy: {avg_acc:.4f}")
        logger.info(f"Average Balanced Accuracy: {avg_bal_acc:.4f}")
        logger.info(f"Average F1 (Macro): {avg_f1_macro:.4f}")
        logger.info(f"Average F1 (Weighted): {avg_f1_weighted:.4f}")
    else:
        logger.info("No successful evaluations")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())