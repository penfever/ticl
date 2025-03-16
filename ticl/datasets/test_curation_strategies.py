#!/usr/bin/env python
"""
Test script for comparing different semantic curation strategies.
This script generates semantic features for a small set of test columns
using different curation strategies and shows the resulting tokens.
"""

import os
import sys
import torch
import json
from transformers import CLIPTokenizerFast
from argparse import ArgumentParser

# Add the parent directory to the path to import from ticl
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ticl.datasets.enhanced_semantic_column_generation import (
    EnhancedColumnSemanticTokenizer, PROMPT_TEMPLATES
)
from ticl.utils import load_secrets

# Test columns representing different types of data
TEST_COLUMNS = [
    # Numeric columns
    "temperature",
    "blood_pressure",
    "price",
    "age",
    "distance",
    # Categorical columns
    "gender",
    "country",
    "color",
    # Text columns
    "description",
    "comment",
    # Date/time columns
    "birth_date",
    "event_time",
    # Mixed/complex columns
    "medical_diagnosis",
    "financial_transaction",
    "customer_feedback",
    "product_rating"
]

def decode_tokens(tokenizer, tokens):
    """Convert token IDs back to text for visualization."""
    # Filter out padding tokens (0)
    non_zero_tokens = tokens[tokens != 0]
    
    # If there are no non-zero tokens, return an empty string
    if len(non_zero_tokens) == 0:
        return ""
    
    # Decode the tokens
    return tokenizer.decode(non_zero_tokens)

def test_curation_strategy(strategy, provider="local", model=None, output_dir="./outputs", verbose=True):
    """
    Test a specific curation strategy on the test columns.
    
    Args:
        strategy: Curation strategy to test
        provider: Model provider to use
        model: Model name to use (or None for default)
        output_dir: Directory to save output files
        verbose: Whether to print detailed output
        
    Returns:
        Dictionary with results for each column
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n=== Testing '{strategy}' curation strategy ===")
    
    # Initialize the tokenizer with the specified strategy
    tokenizer = EnhancedColumnSemanticTokenizer(
        provider=provider,
        model=model,
        max_tokens=200,
        curation_strategy=strategy
    )
    
    # Store results
    results = {}
    
    # Process each test column
    for column_name in TEST_COLUMNS:
        if verbose:
            print(f"\nProcessing column: {column_name}")
            print(f"Prompt template: {tokenizer.prompt_template[:100]}...")
        
        # Generate tokens for the column
        tokens = tokenizer.process_column(column_name, max_non_zero_tokens=100)
        
        # Split the tokens into column name tokens and value tokens
        col_name_tokens = tokens[:tokenizer.max_tokens]
        value_tokens = tokens[tokenizer.max_tokens:]
        
        # Decode the tokens
        clip_tokenizer = tokenizer.clip_tokenizer
        col_name_text = decode_tokens(clip_tokenizer, col_name_tokens)
        values_text = decode_tokens(clip_tokenizer, value_tokens)
        
        if verbose:
            print(f"Column name tokens: {col_name_text}")
            print(f"Sample values: {values_text[:200]}...")
        
        # Store results
        results[column_name] = {
            "column_name": column_name,
            "column_name_tokens": col_name_tokens.tolist(),
            "column_name_text": col_name_text,
            "value_tokens": value_tokens.tolist(),
            "value_text": values_text
        }
    
    # Save results to a JSON file
    output_file = os.path.join(output_dir, f"{strategy}_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    return results

def main():
    parser = ArgumentParser(description="Test different semantic curation strategies")
    parser.add_argument(
        "--provider", 
        type=str, 
        choices=["local", "gemini", "together", "anthropic"], 
        default="local",
        help="Model provider ('local', 'gemini', 'together', or 'anthropic')"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default=None,
        help="Model name for the specified provider"
    )
    parser.add_argument(
        "--strategies", 
        type=str, 
        nargs="+",
        choices=list(PROMPT_TEMPLATES.keys()),
        default=list(PROMPT_TEMPLATES.keys()),
        help="Strategies to test (default: all)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./curation_test_outputs",
        help="Directory to save output files"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Print detailed output"
    )
    args = parser.parse_args()
    
    # Load secrets for API access
    load_secrets()
    
    print(f"Testing {len(args.strategies)} curation strategies on {len(TEST_COLUMNS)} test columns")
    print(f"Provider: {args.provider}")
    print(f"Model: {args.model or 'default'}")
    
    # Test each strategy
    for strategy in args.strategies:
        test_curation_strategy(
            strategy=strategy,
            provider=args.provider,
            model=args.model,
            output_dir=args.output_dir,
            verbose=args.verbose
        )
    
    print("\nDone!")

if __name__ == "__main__":
    main()