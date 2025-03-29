#!/usr/bin/env python
"""
Load and process data files in the carte benchmarks directory.

This script:
1. Loads .pickle and .parquet files and displays their contents
2. Loads the config_data.json file and extracts the target_name
3. Finds the corresponding CSV file and extracts unique values from the target column
4. Adds semantic descriptions for each unique value to config_data.json
"""

import os
import sys
import argparse
import pickle
import json
import pandas as pd
import numpy as np
from pprint import pprint
import time
import asyncio

# Add the parent directory to the path so we can import local modules if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import LLM client functions from semantic_column_generation
from ticl.datasets.semantic_column_generation import (
    AnthropicClient, GeminiClient, TogetherClient, LocalGenerationClient
)

def load_secrets():
    import yaml
    import os
    secrets = {}
    with open('../secrets.yml', 'r') as file:
        secrets = yaml.safe_load(file)
    for k, v in secrets.items():
        os.environ[k] = v


def load_pickle_file(file_path):
    """Load data from a pickle file."""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print(f"\nSuccessfully loaded pickle file: {file_path}")
        return data
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return None


def load_parquet_file(file_path):
    """Load data from a parquet file into a pandas DataFrame."""
    try:
        data = pd.read_parquet(file_path)
        print(f"\nSuccessfully loaded parquet file: {file_path}")
        return data
    except Exception as e:
        print(f"Error loading parquet file: {e}")
        return None


def load_csv_file(file_path):
    """Load data from a CSV file into a pandas DataFrame."""
    try:
        data = pd.read_csv(file_path)
        print(f"\nSuccessfully loaded CSV file: {file_path}")
        return data
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None


def load_json_file(file_path):
    """Load data from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(f"\nSuccessfully loaded JSON file: {file_path}")
        return data
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return None


def save_json_file(file_path, data):
    """Save data to a JSON file."""
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\nSuccessfully saved JSON file: {file_path}")
        return True
    except Exception as e:
        print(f"Error saving JSON file: {e}")
        return False


def display_pickle_preview(data):
    """Display a preview of the pickle data."""
    print("\nPickle Data Preview:")
    print(f"{'-' * 80}")
    
    if data is None:
        print("No data to display.")
        return
    
    # Check data type to determine display format
    if isinstance(data, dict):
        print(f"Data Type: Dictionary with {len(data)} keys")
        print("\nKeys:")
        for i, key in enumerate(data.keys()):
            if i >= 5:  # Limit to first 5 keys
                print(f"... and {len(data) - 5} more keys")
                break
            print(f"  - {key}")
        
        # Try to display a sample value
        if data and list(data.keys()):
            sample_key = list(data.keys())[0]
            print(f"\nSample Value (for key '{sample_key}'):")
            try:
                pprint(data[sample_key], depth=2, compact=True, width=80)
            except:
                print("  [Unable to display sample value]")
    
    elif isinstance(data, list):
        print(f"Data Type: List with {len(data)} items")
        print("\nFirst few items:")
        for i, item in enumerate(data[:5]):  # Limit to first 5 items
            print(f"  {i}: {str(item)[:100]}")
        if len(data) > 5:
            print(f"... and {len(data) - 5} more items")
    
    else:
        print(f"Data Type: {type(data)}")
        try:
            print("\nPreview:")
            pprint(data, depth=1, compact=True, width=80)
        except:
            print("  [Unable to display data preview]")


def display_parquet_preview(data):
    """Display a preview of the parquet DataFrame."""
    print("\nParquet Data Preview:")
    print(f"{'-' * 80}")
    
    if data is None or data.empty:
        print("No data to display or empty DataFrame.")
        return
    
    # Display DataFrame info
    print(f"DataFrame Shape: {data.shape} (rows, columns)")
    print(f"DataFrame Columns: {', '.join(data.columns[:10])}")
    if len(data.columns) > 10:
        print(f"... and {len(data.columns) - 10} more columns")
    
    # Display data types
    print("\nData Types:")
    for col, dtype in list(data.dtypes.items())[:5]:
        print(f"  {col}: {dtype}")
    if len(data.dtypes) > 5:
        print(f"... and {len(data.dtypes) - 5} more columns")
    
    # Display first few rows
    print("\nFirst 5 rows:")
    try:
        print(data.head().to_string())
    except:
        print("  [Unable to display rows]")


def find_files(directory):
    """Find relevant files in the given directory."""
    pickle_files = []
    parquet_files = []
    csv_files = []
    config_file = None
    
    if not os.path.exists(directory):
        print(f"Directory does not exist: {directory}")
        return pickle_files, parquet_files, csv_files, config_file
    
    # Get parent directory name for CSV file
    parent_dir_name = os.path.basename(os.path.normpath(directory))
    expected_csv_name = f"{parent_dir_name}.csv"
    
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            if file.endswith('.pickle') or file.endswith('.pkl'):
                pickle_files.append(file_path)
            elif file.endswith('.parquet'):
                parquet_files.append(file_path)
            elif file.lower() == "config_data.json":
                config_file = file_path
            elif file.lower() == expected_csv_name.lower():
                csv_files.append(file_path)
            elif file.endswith('.csv'):
                csv_files.append(file_path)
    
    return pickle_files, parquet_files, csv_files, config_file


def initialize_llm_client(provider="anthropic", model=None):
    """
    Initialize an LLM client for generating semantic descriptions.
    
    Args:
        provider: LLM provider ('anthropic', 'gemini', or 'together')
        model: Model name (if None, uses a default model)
        
    Returns:
        An initialized LLM client
    """
    # Load secrets from secrets.yml
    try:
        load_secrets()
    except Exception as e:
        print(f"Warning: Failed to load secrets file. Some models may not work: {e}")
    
    # Set default models based on provider
    provider_defaults = {
        "local": "Qwen/Qwen2.5-7B-Instruct",
        "gemini": "gemini-2.0-flash",
        "together": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "anthropic": "claude-3-haiku-20240307"
    }
    
    if model is None:
        model = provider_defaults.get(provider, provider_defaults["anthropic"])
    
    # Initialize the appropriate client
    try:
        if provider == "local":
            client = LocalGenerationClient(model, "cpu")
        elif provider == "gemini":
            client = GeminiClient(model)
        elif provider == "together":
            client = TogetherClient(model)
        elif provider == "anthropic":
            client = AnthropicClient(model)
        else:
            print(f"Unsupported provider: {provider}, falling back to Anthropic")
            client = AnthropicClient(provider_defaults["anthropic"])
        
        print(f"Initialized {provider} client with model {model}")
        return client
    except Exception as e:
        print(f"Failed to initialize {provider} client: {e}")
        print("Falling back to rule-based descriptions")
        return None


def generate_class_descriptions_with_llm(client, values, column_name, dataset_name, dataset_description=None):
    """
    Generate semantic descriptions for the unique values using an LLM.
    
    Args:
        client: The LLM client to use for generation
        values: List of unique values
        column_name: Name of the target column
        dataset_name: Name of the dataset
        dataset_description: Optional description of the dataset
        
    Returns:
        Dictionary mapping values to descriptions
    """
    # Filter out NaN values
    values = [v for v in values if not (isinstance(v, float) and np.isnan(v))]
    
    # Limit number of values to avoid token limits
    if len(values) > 50:
        print(f"Found {len(values)} unique values, sampling 50 for LLM prompt")
        # Take a stratified sample if numeric, otherwise random
        if all(isinstance(v, (int, float)) for v in values):
            # Sort and take a stratified sample
            sorted_values = sorted(values)
            stride = max(1, len(sorted_values) // 50)
            sampled_values = sorted_values[::stride][:50]
        else:
            # Random sample for categorical
            import random
            sampled_values = random.sample(values, 50)
    else:
        sampled_values = values
    
    # Check if values are numeric or categorical
    is_numeric = all(isinstance(v, (int, float)) for v in values)
    
    # Create a tailored prompt based on the type of data
    if is_numeric:
        value_list = ", ".join([str(val) for val in sorted(sampled_values)[:20]])
        
        prompt = f"""
        You are analyzing the '{column_name}' column in the {dataset_name} dataset.

        {dataset_description or ""}
        
        The column contains numeric values. Here are some examples: {value_list}...
        
        For each unique value in the column, create a detailed semantic description that:
        1. Contextualizes what this value means in this dataset
        2. Provides domain-specific interpretation
        3. Gives enough detail for a machine learning model to use for semantic understanding
        
        Please create concise descriptions (1-2 sentences) for each of the following values.
        Format your response as a JSON object with values as keys and descriptions as values:
        
        {{
          "value1": "Description for value1",
          "value2": "Description for value2",
          ...
        }}
        
        The actual values to describe are: {sorted(set(values))}
        """
    else:
        value_list = ", ".join([f"'{str(val)}'" for val in sorted(sampled_values)[:20]])
        
        prompt = f"""
        You are analyzing the '{column_name}' column in the {dataset_name} dataset.

        {dataset_description or ""}
        
        The column contains categorical values. Here are some examples: {value_list}...
        
        For each unique category in the column, create a detailed semantic description that:
        1. Contextualizes what this category means in this dataset
        2. Provides domain-specific interpretation
        3. Gives enough detail for a machine learning model to use for semantic understanding
        
        Please create concise descriptions (1-2 sentences) for each of the following categories.
        Format your response as a JSON object with values as keys and descriptions as values:
        
        {{
          "value1": "Description for value1",
          "value2": "Description for value2",
          ...
        }}
        
        The actual values to describe are: {sorted(set(values))}
        """
    
    # Generate descriptions using the LLM
    try:
        generated_text = client.generate(
            prompt,
            max_tokens=4000,
            temperature=0.2,
        )
        
        # Extract JSON from the response
        try:
            # Find JSON block in the response (may be wrapped in backticks)
            import re
            json_match = re.search(r'```(?:json)?(.*?)```', generated_text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                # Try to find outer braces if not in code block
                start_idx = generated_text.find('{')
                end_idx = generated_text.rfind('}') + 1
                
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    json_str = generated_text[start_idx:end_idx]
                else:
                    raise ValueError("Could not find JSON in LLM response")
            
            # Parse the JSON
            descriptions = json.loads(json_str)
            
            # Convert all keys to strings if they aren't already
            descriptions = {str(k): v for k, v in descriptions.items()}
            
            # Verify we have descriptions for all values
            missing_values = set(str(v) for v in values) - set(descriptions.keys())
            if missing_values and len(missing_values) < 20:
                print(f"Warning: Missing descriptions for {len(missing_values)} values. Generating rule-based descriptions for these.")
                # Fill in missing values with rule-based descriptions
                rule_descriptions = generate_class_descriptions_rule_based(values, column_name, dataset_name)
                for val in missing_values:
                    if val in rule_descriptions:
                        descriptions[val] = rule_descriptions[val]
            
            return descriptions
            
        except Exception as e:
            print(f"Error parsing LLM response as JSON: {e}")
            print("Falling back to rule-based descriptions")
            return generate_class_descriptions_rule_based(values, column_name, dataset_name)
            
    except Exception as e:
        print(f"LLM generation failed: {e}")
        print("Falling back to rule-based descriptions")
        return generate_class_descriptions_rule_based(values, column_name, dataset_name)


def generate_class_descriptions_rule_based(values, column_name, dataset_name):
    """
    Generate semantic descriptions for the unique values in the target column using rules.
    
    For numeric values, we'll categorize them into meaningful ranges.
    For categorical values, we'll generate descriptions based on the unique values.
    """
    # Filter out NaN values
    values = [v for v in values if not (isinstance(v, float) and np.isnan(v))]
    
    # Check if the values are numeric or categorical
    is_numeric = all(isinstance(v, (int, float)) for v in values)
    
    if is_numeric:
        # For numeric values, create meaningful ranges
        min_val = min(values)
        max_val = max(values)
        
        # Determine the number of bins (categories) based on data range
        range_span = max_val - min_val
        
        if range_span <= 5:  # Small range, use individual values
            unique_values = sorted(set(values))
            descriptions = {}
            for val in unique_values:
                if val < 0:
                    descriptions[str(val)] = f"Below average {column_name} - players who performed worse than league average"
                elif val == 0:
                    descriptions[str(val)] = f"Average {column_name} - players who performed at the league average level"
                elif 0 < val <= 5:
                    descriptions[str(val)] = f"Above average {column_name} - solid role players with positive contributions"
                elif 5 < val <= 15:
                    descriptions[str(val)] = f"High {column_name} - quality starters or significant rotation players"
                elif 15 < val <= 30:
                    descriptions[str(val)] = f"Very high {column_name} - All-Star caliber players with major impact"
                elif 30 < val <= 50:
                    descriptions[str(val)] = f"Elite {column_name} - superstar players with franchise-altering impact"
                else:
                    descriptions[str(val)] = f"Historic {column_name} - Hall of Fame level players with legendary careers"
        
        else:  # Larger range, create bins
            # Create bins for classification
            if dataset_name == "nba_draft" and column_name == "value_over_replacement":
                # Custom bins for VOR in NBA data
                bins = [
                    (float('-inf'), -5, "Negative impact players - significantly below replacement level"),
                    (-5, 0, "Replacement level players - minimal positive contribution"),
                    (0, 10, "Role players - positive contributors but not stars"),
                    (10, 25, "Quality starters - solid NBA starters or sixth men"),
                    (25, 50, "All-Stars - top tier talents and occasional All-Stars"),
                    (50, 75, "Superstars - perennial All-Stars and franchise cornerstones"),
                    (75, float('inf'), "Hall of Fame level - all-time greats with exceptional careers")
                ]
            else:
                # Generic approach for other numeric data
                # Create equal-width bins
                num_bins = min(7, len(set(values)))  # Maximum 7 bins
                bin_width = range_span / num_bins
                
                bins = []
                for i in range(num_bins):
                    lower = min_val + i * bin_width
                    upper = min_val + (i+1) * bin_width if i < num_bins-1 else float('inf')
                    
                    # Generate description based on relative position
                    if i == 0:
                        description = f"Very low {column_name} - bottom {100/num_bins:.0f}% of values"
                    elif i < num_bins // 3:
                        description = f"Low {column_name} - below average values"
                    elif i < (2 * num_bins) // 3:
                        description = f"Medium {column_name} - average values"
                    elif i < num_bins - 1:
                        description = f"High {column_name} - above average values"
                    else:
                        description = f"Very high {column_name} - top {100/num_bins:.0f}% of values"
                    
                    bins.append((lower, upper, description))
            
            # Assign each value to a bin
            descriptions = {}
            for val in sorted(set(values)):
                for lower, upper, description in bins:
                    if lower <= val < upper:
                        descriptions[str(val)] = description
                        break
    else:
        # For categorical values
        descriptions = {}
        for val in sorted(set(values)):
            str_val = str(val)
            if dataset_name == "nba_draft":
                # Example for NBA data if there were categorical values
                descriptions[str_val] = f"Players classified as '{val}' in the NBA draft dataset"
            else:
                # Generic description for categorical values
                descriptions[str_val] = f"Items with {column_name} value of '{val}' in the {dataset_name} dataset"
    
    return descriptions


def main():
    """Main function to load and process data files."""
    parser = argparse.ArgumentParser(
        description='Load and process data files in the carte benchmarks directory.'
    )
    parser.add_argument(
        '--dir', 
        type=str, 
        default='benchmarks/carte/nba_draft',
        help='Directory path containing the data files'
    )
    parser.add_argument(
        '--provider',
        type=str,
        choices=['anthropic', 'gemini', 'together', 'local', 'none'],
        default='anthropic',
        help='LLM provider to use for generating descriptions'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Specific model to use with the provider (if not specified, uses default model)'
    )
    parser.add_argument(
        '--description',
        type=str,
        default=None,
        help='Optional description of the dataset to help with LLM generation'
    )
    parser.add_argument(
        '--rule-based',
        action='store_true',
        help='Use rule-based descriptions instead of LLM, even if LLM is available'
    )
    args = parser.parse_args()
    
    # Get absolute path of the directory
    # First check if path is absolute or relative
    if os.path.isabs(args.dir):
        directory = args.dir
    else:
        # If relative, construct path relative to project root
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        directory = os.path.join(project_root, args.dir)
    
    # Get dataset name from directory
    dataset_name = os.path.basename(os.path.normpath(directory))
    
    print(f"Looking for data files in: {directory}")
    
    # Find relevant files
    pickle_files, parquet_files, csv_files, config_file = find_files(directory)
    
    if not config_file:
        print("Error: config_data.json not found in the directory.")
        return
    
    if not csv_files:
        print(f"Error: No CSV files found in the directory. Expected {dataset_name}.csv")
        return
    
    print(f"Found {len(pickle_files)} pickle files, {len(parquet_files)} parquet files, " 
          f"{len(csv_files)} CSV files, and config_data.json")
    
    # Load and display pickle files
    for file_path in pickle_files:
        print(f"\n{'=' * 80}")
        print(f"Processing file: {os.path.basename(file_path)}")
        data = load_pickle_file(file_path)
        display_pickle_preview(data)
    
    # Load and display parquet files
    for file_path in parquet_files:
        print(f"\n{'=' * 80}")
        print(f"Processing file: {os.path.basename(file_path)}")
        data = load_parquet_file(file_path)
        display_parquet_preview(data)
    
    # Load config_data.json to get target_name
    config_data = load_json_file(config_file)
    if not config_data or 'target_name' not in config_data:
        print("Error: config_data.json does not contain 'target_name' field.")
        return
    
    target_name = config_data['target_name']
    print(f"\nTarget column name from config_data.json: {target_name}")
    
    # Find the expected CSV file based on the directory name
    expected_csv_name = f"{dataset_name}.csv"
    csv_file_path = None
    
    for file_path in csv_files:
        if os.path.basename(file_path).lower() == expected_csv_name.lower():
            csv_file_path = file_path
            break
    
    if not csv_file_path and csv_files:
        # If we didn't find the expected name but have other CSVs, use the first one
        csv_file_path = csv_files[0]
    
    if not csv_file_path:
        print(f"Error: CSV file {expected_csv_name} not found in the directory.")
        return
    
    # Load the CSV file and get unique values from the target column
    csv_data = load_csv_file(csv_file_path)
    if csv_data is None:
        print("Error loading CSV file.")
        return
    
    if target_name not in csv_data.columns:
        print(f"Error: Target column '{target_name}' not found in the CSV file.")
        return
    
    unique_values = csv_data[target_name].unique()
    print(f"\nFound {len(unique_values)} unique values in the '{target_name}' column.")
    print("Sample values:", unique_values[:5])
    
    # Generate semantic descriptions for each unique value
    print("\nGenerating semantic descriptions for unique values...")
    
    # Use LLM if requested and available
    if not args.rule_based and args.provider.lower() != 'none':
        # Initialize LLM client
        llm_client = initialize_llm_client(provider=args.provider, model=args.model)
        
        if llm_client:
            print(f"Using {args.provider} LLM to generate class descriptions")
            dataset_description = args.description or f"This is a dataset about {dataset_name}."
            
            class_descriptions = generate_class_descriptions_with_llm(
                llm_client, 
                unique_values, 
                target_name, 
                dataset_name,
                dataset_description
            )
        else:
            print("Failed to initialize LLM client, falling back to rule-based descriptions")
            class_descriptions = generate_class_descriptions_rule_based(unique_values, target_name, dataset_name)
    else:
        print("Using rule-based descriptions")
        class_descriptions = generate_class_descriptions_rule_based(unique_values, target_name, dataset_name)
    
    # Add class_names to config_data
    config_data['class_names'] = class_descriptions
    
    # Save updated config_data.json
    print("\nSaving updated config_data.json...")
    save_json_file(config_file, config_data)
    
    # Display a sample of the added descriptions
    print("\nSample of added class descriptions:")
    sample_count = min(5, len(class_descriptions))
    sample_items = list(class_descriptions.items())[:sample_count]
    for value, description in sample_items:
        print(f"  - {value}: {description}")
    
    if len(class_descriptions) > sample_count:
        print(f"... and {len(class_descriptions) - sample_count} more descriptions")
    
    print("\nDone!")


if __name__ == "__main__":
    main()