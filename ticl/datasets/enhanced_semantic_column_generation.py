import torch
from transformers import pipeline, CLIPTokenizerFast
from typing import List, Optional, Dict, Any
import ast
import argparse
import os
import json
import random
import asyncio
import time
import re
import numpy as np
from tqdm import tqdm
from datetime import datetime

from ticl.datasets.processed_schema_types import SCHEMA_TYPES
from ticl.utils import load_secrets
from ticl.datasets.semantic_column_generation import (
    TextGenerationClient, LocalGenerationClient, GeminiClient, 
    TogetherClient, AnthropicClient, ColumnSemanticTokenizer,
    truncate_tensor
)
from ticl.datasets.statistical_column_mapping import StatisticalColumnMapper

# Define enhanced prompts for different curation strategies
PROMPT_TEMPLATES = {
    'standard': """
                {column_name} is the name of column in a table found on the web. 
                Please give me a Python formatted list of around 100 unique values (words, integers, floating point numbers,
                phrases, dates, stringifed numbers like zipcodes, etc) that would be likely to appear in this column with this name.
                Where appropriate, also include synonyms, antonyms, hierarchically related concepts, and non-English translations. Be creative!
                If {column_name} doesn't mean anything to you, return an empty list.
                """,
    
    'numeric_properties': """
                {column_name} is the name of column in a table found on the web that likely contains numeric values.
                Please give me a Python formatted list of descriptive terms for the statistical properties of these numbers, such as:
                - Terms describing magnitude: "high", "low", "medium", "peak value", "minimum", etc.
                - Terms describing distribution: "normal distribution", "skewed", "outliers", "variance", etc.
                - Terms describing trends: "increasing", "decreasing", "stable", "volatile", "seasonal", etc.
                - Terms describing correlations: "positively correlated with", "inversely related to", etc.
                - Terms describing relative positions: "above average", "below threshold", "within normal range", etc.
                
                Be sure to include 80-100 unique descriptive terms that would be semantically meaningful for data analysis.
                If {column_name} doesn't seem like it would contain numeric values, focus on qualitative descriptors instead.
                """,
    
    'causal_relationships': """
                {column_name} is the name of column in a table found on the web.
                Please give me a Python formatted list of 80-100 causal relationships that might involve this column, such as:
                - "causes X" (where this column might be a causal factor for X)
                - "caused by Y" (where this column might be an effect of Y)
                - "influences Z" (where this column has a partial causal effect on Z)
                - "affected by W" (where this column is partially affected by W)
                - "predicts P" (where this column is predictive of P)
                - "indicated by Q" (where this column is indicated by Q)
                
                For each causal relationship, include the relationship type and the related concept.
                Be sure to consider both forward and reverse causal connections.
                If {column_name} doesn't mean anything to you or causal relationships don't make sense for it, return more general semantic associations instead.
                """,
                
    'statistical_causal_relationships': """
                I need to understand the causal relationships between specific values of {column_name} and potential target classes.
                
                The column {column_name} has the following value ranges:
                {value_ranges}
                
                For a classification task that might use {column_name} as a feature, please generate a Python dictionary that maps 
                each value range to potential target classes it could causally influence, along with an explanation of the causal mechanism.
                
                For example, if the column is "income" with ranges "low income", "medium income", and "high income", you might return:
                ```python
                {
                    "low income (0-30,000 USD)": [
                        {"class": "default_risk_high", "causal_explanation": "Low income reduces financial buffer, increasing default risk."},
                        {"class": "budget_shopper", "causal_explanation": "Lower income necessitates more budget-conscious purchasing."}
                    ],
                    "medium income (30,000-70,000 USD)": [
                        {"class": "default_risk_medium", "causal_explanation": "Medium income provides some financial stability."},
                        {"class": "value_shopper", "causal_explanation": "Seeks value but can occasionally afford premium products."}
                    ],
                    "high income (70,000+ USD)": [
                        {"class": "default_risk_low", "causal_explanation": "High income creates financial security and lower default risk."},
                        {"class": "luxury_consumer", "causal_explanation": "Higher income enables luxury purchases and premium services."}
                    ]
                }
                ```
                
                Please create meaningful, realistic causal relationships between each value range and 2-4 potential target classes.
                Focus on relationships that are:
                1. Plausible based on real-world knowledge
                2. Specific enough to be useful in a machine learning context
                3. Diverse, showing different ways the feature might influence outcomes
                
                Format your response as a valid Python dictionary with the exact structure shown in the example.
                """,
                
    'conceptual_clusters': """
                {column_name} is the name of column in a table found on the web.
                Please give me a Python formatted list of conceptual clusters related to this column, organized as follows:
                
                [
                  {{"cluster": "Cluster Name 1", "terms": ["term1", "term2", "term3", ...]}},
                  {{"cluster": "Cluster Name 2", "terms": ["term4", "term5", "term6", ...]}},
                  ...
                ]
                
                For example, if the column is "medical_symptoms", clusters might include:
                - "Respiratory": ["cough", "shortness of breath", "wheezing", ...]
                - "Digestive": ["nausea", "vomiting", "abdominal pain", ...]
                
                Create 5-10 meaningful clusters with 10-15 terms each that are semantically related to {column_name}.
                If {column_name} doesn't make sense to you, create general conceptual clusters that might be relevant to tabular data analysis.
                """,
                
    'contrastive_pairs': """
                {column_name} is the name of column in a table found on the web.
                Please give me a Python formatted list of contrastive pairs related to this column.
                Each pair should represent semantically opposite or contrasting concepts that might appear in or be related to this column.
                
                Format your response as:
                [
                  ["concept1a", "concept1b"],
                  ["concept2a", "concept2b"],
                  ...
                ]
                
                For example, if the column is "temperature", contrastive pairs might include:
                ["hot", "cold"], ["warming", "cooling"], ["thermal expansion", "thermal contraction"]
                
                Create 40-50 meaningful contrastive pairs that are semantically related to {column_name}.
                Ensure the pairs represent a true semantic contrast or opposition relevant to the column concept.
                If {column_name} doesn't make sense to you, create general contrasting pairs that might be relevant to tabular data.
                """,
                
    'metadata_description': """
                You are an expert data scientist who understands tabular data extremely well.
                
                I need a comprehensive metadata description for a column named "{column_name}" that might appear in a dataset.
                
                Please create a detailed metadata description that covers ALL of the following aspects:
                
                1. SEMANTIC MEANING:
                   - What does this column name typically represent? (Be thorough about all possible meanings)
                   - What information would this column likely contain?
                   - Are there different domains or contexts where this column might have different meanings?
                
                2. DATA CHARACTERISTICS:
                   - What data type(s) would you expect for this column? (int, float, string, date, categorical, etc.)
                   - What would be the likely range, format, or units of measurement?
                   - Are there common patterns, constraints, or special values (e.g., NULL interpretations)?
                   - What distribution shape might this data follow? (normal, skewed, multimodal, etc.)
                
                3. CONTEXTUAL RELATIONSHIPS:
                   - What other columns would likely co-occur with this column in datasets?
                   - Which domains, industries, or fields commonly use data with this column?
                   - What analytical questions is this column typically used to answer?
                
                4. DATA QUALITY CONSIDERATIONS:
                   - What are common data quality issues for this type of column?
                   - How might missing values be interpreted?
                   - What validation rules would typically apply?
                
                5. ANALYTICAL VALUE:
                   - How is this column typically used in analysis or modeling?
                   - Would this column likely be a dependent or independent variable?
                   - What transformations are commonly applied to this type of data?
                
                Format your response as a cohesive, detailed paragraph that integrates all these aspects.
                Be comprehensive yet concise. Ensure your description is factually accurate and considers multiple interpretations where applicable.
                
                Aim for 200-300 words that would give a data scientist a thorough understanding of what this column name represents.
                """
}

class EnhancedColumnSemanticTokenizer(ColumnSemanticTokenizer):
    """
    Enhanced version of ColumnSemanticTokenizer that supports different curation strategies.
    """
    
    def __init__(self, *args, **kwargs):
        # Extract the curation strategy from kwargs if present, otherwise use standard
        self.curation_strategy = kwargs.pop('curation_strategy', 'standard')
        self.prompt_template = PROMPT_TEMPLATES.get(self.curation_strategy, PROMPT_TEMPLATES['standard'])
        
        # Dictionary to store raw structured data for special curation strategies
        self.structured_data = {}
        
        # Path to save conceptual clusters data
        self.clusters_save_path = kwargs.pop('clusters_save_path', None)
        
        # Initialize the statistical column mapper for numeric features
        self.column_mapper = StatisticalColumnMapper()
        
        # Dictionary to store causal relationships for statistical features
        self.statistical_causal_relationships = {}
        
        # Path to save statistical causal relationship data
        self.stat_causal_path = kwargs.pop('stat_causal_path', 'statistical_causal_relationships.json')
        
        # Call the parent constructor with the remaining arguments
        super().__init__(*args, **kwargs)
    
    def _parse_generated_list(self, text: str, column_name: str = None) -> List[str]:
        """
        Parse the generated text to extract a list of semantic values.
        Enhanced to handle different output formats based on curation strategy.
        
        Args:
            text: The generated text containing a Python list
            column_name: Optional name of the column being processed
            
        Returns:
            List of extracted semantic values
        """
        if self.curation_strategy == 'metadata_description':
            # For metadata descriptions, we're getting a paragraph, not a list
            try:
                # Clean the text to prepare for extraction
                clean_text = text.replace('```', '').strip()
                
                # Store the raw metadata description if column_name is provided
                if column_name is not None:
                    self.structured_data[column_name] = clean_text
                    # Save the description to a file if path is provided
                    self._save_clusters_data()
                
                # For tokenization, split the description into meaningful chunks
                # We'll use sentences or phrases as our "terms"
                import re
                
                # Split text into sentences
                sentences = re.split(r'(?<=[.!?])\s+', clean_text)
                
                # Further split long sentences on commas, semicolons, etc.
                terms = []
                for sentence in sentences:
                    if len(sentence) > 80:  # If sentence is long, split it further
                        fragments = re.split(r'(?<=[,;:])\s+', sentence)
                        terms.extend(fragments)
                    else:
                        terms.append(sentence)
                
                # Add the column name itself and some key phrases
                key_terms = []
                if column_name:
                    key_terms = [
                        column_name,
                        f"{column_name} data",
                        f"{column_name} feature",
                        f"{column_name} column",
                        f"{column_name} field"
                    ]
                
                # Combine all pieces
                return key_terms + terms
                
            except Exception:
                # Fall back to a very simple approach if parsing fails
                return [text[:1000]] if text else []
        
        elif self.curation_strategy == 'statistical_causal_relationships':
            # Try to extract a dictionary with causal relationships
            try:
                # Clean the text to prepare for extraction
                clean_text = text.replace('```python', '').replace('```', '')
                
                # Find the dictionary portion of the text
                start_idx = clean_text.find("{")
                end_idx = clean_text.rfind("}") + 1
                
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    dict_str = clean_text[start_idx:end_idx]
                    # Parse the JSON-like structure
                    causal_dict = ast.literal_eval(dict_str)
                    
                    # Store the raw causal relationships if column_name is provided
                    if column_name is not None:
                        self.statistical_causal_relationships[column_name] = causal_dict
                        # Save to file
                        self._save_statistical_causal_data()
                    
                    # Extract all causal terms and flatten them for tokenization
                    all_terms = []
                    
                    # Add the column name and value range terms
                    all_terms.append(column_name)
                    all_terms.extend(causal_dict.keys())
                    
                    # Add class names and causal explanations
                    for value_range, class_info_list in causal_dict.items():
                        for class_info in class_info_list:
                            if isinstance(class_info, dict):
                                if 'class' in class_info:
                                    all_terms.append(class_info['class'])
                                if 'causal_explanation' in class_info:
                                    # Split explanation into phrases
                                    explanation = class_info['causal_explanation']
                                    phrases = re.split(r'(?<=[.!?])\s+|(?<=,)\s+', explanation)
                                    all_terms.extend(phrases)
                    
                    return all_terms
            except Exception as e:
                # Fall back to a simpler approach with warning
                print(f"Error parsing statistical causal relationships: {e}")
                if column_name:
                    return [column_name, "causal relationship", "statistical feature", "value range", "target class"]
                return ["causal relationship", "statistical feature", "value range", "target class"]
            
        elif self.curation_strategy == 'conceptual_clusters':
            # Try to extract a list of dictionaries with clusters
            try:
                # Clean the text to prepare for extraction
                clean_text = text.replace('```python', '').replace('```', '')
                
                # Find the list portion of the text
                start_idx = clean_text.find("[")
                end_idx = clean_text.rfind("]") + 1
                
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    list_str = clean_text[start_idx:end_idx]
                    # Parse the JSON-like structure
                    clusters = ast.literal_eval(list_str)
                    
                    # Store the raw cluster data if column_name is provided
                    if column_name is not None:
                        self.structured_data[column_name] = clusters
                        # Save the clusters to a file if path is provided
                        self._save_clusters_data()
                    
                    # Extract all terms from all clusters and flatten them
                    all_terms = []
                    for cluster in clusters:
                        if isinstance(cluster, dict) and 'terms' in cluster:
                            all_terms.extend(cluster['terms'])
                        elif isinstance(cluster, dict) and 'cluster' in cluster and 'terms' in cluster:
                            # Add the cluster name as a term too
                            all_terms.append(cluster['cluster'])
                            all_terms.extend(cluster['terms'])
                    
                    return all_terms
            except Exception:
                # Fall back to standard parsing without verbose error message
                pass
        
        elif self.curation_strategy == 'contrastive_pairs':
            # Try to extract a list of pairs
            try:
                # Clean the text to prepare for extraction
                clean_text = text.replace('```python', '').replace('```', '')
                
                # Find the list portion of the text
                start_idx = clean_text.find("[")
                end_idx = clean_text.rfind("]") + 1
                
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    list_str = clean_text[start_idx:end_idx]
                    # Parse the list of pairs
                    pairs = ast.literal_eval(list_str)
                    
                    # Store the raw pairs data if column_name is provided
                    if column_name is not None:
                        self.structured_data[column_name] = pairs
                        # Save the structured data to a file if path is provided
                        self._save_clusters_data()
                    
                    # Flatten the pairs into a single list
                    all_terms = []
                    for pair in pairs:
                        if isinstance(pair, list) and len(pair) == 2:
                            all_terms.extend(pair)
                    
                    return all_terms
            except Exception:
                # Fall back to standard parsing without verbose error message
                pass
        
        # For other strategies or if specialized parsing failed, use the standard parser
        return super()._parse_generated_list(text)
    
    def _save_clusters_data(self):
        """
        Save the structured data to a JSON file.
        Only saves if clusters_save_path is set.
        """
        if self.clusters_save_path and self.structured_data:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(self.clusters_save_path)), exist_ok=True)
            
            # Check if the file already exists
            existing_data = {}
            if os.path.exists(self.clusters_save_path):
                try:
                    with open(self.clusters_save_path, 'r') as f:
                        existing_data = json.load(f)
                except Exception:
                    # If loading fails, we'll start with an empty dict
                    pass
            
            # Merge existing data with new data (update will overwrite duplicates)
            existing_data.update(self.structured_data)
            
            # Save the merged data to a JSON file
            with open(self.clusters_save_path, 'w') as f:
                json.dump(existing_data, f, indent=2)
                
            # Only clear the dictionary if we're in sequential mode
            # In parallel mode, we need to keep the data for batch processing
            if not hasattr(self, 'keep_structured_data') or not self.keep_structured_data:
                # Clear the structured_data dictionary to prevent memory build-up
                # We've already saved it to the file and merged with existing data
                self.structured_data = {}
    
    def _save_statistical_causal_data(self):
        """
        Save the statistical causal relationships to a JSON file.
        Only saves if stat_causal_path is set.
        """
        if self.stat_causal_path and self.statistical_causal_relationships:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(self.stat_causal_path)), exist_ok=True)
            
            # Check if the file already exists
            existing_data = {}
            if os.path.exists(self.stat_causal_path):
                try:
                    with open(self.stat_causal_path, 'r') as f:
                        existing_data = json.load(f)
                except Exception:
                    # If loading fails, we'll start with an empty dict
                    pass
            
            # Merge existing data with new data (update will overwrite duplicates)
            existing_data.update(self.statistical_causal_relationships)
            
            # Save the merged data to a JSON file
            with open(self.stat_causal_path, 'w') as f:
                json.dump(existing_data, f, indent=2)
                
            # Only clear if we're in sequential mode
            if not hasattr(self, 'keep_structured_data') or not self.keep_structured_data:
                # Clear the dictionary to prevent memory build-up
                self.statistical_causal_relationships = {}
    
    def generate_statistical_causal_relationships(self, column_name: str, feature_stats: Dict[int, Dict[str, Any]] = None) -> Dict:
        """
        Generate causal relationships between statistical feature values and target classes
        using LLM prompting.
        
        Parameters:
        -----------
        column_name : str
            The name of the column to generate causal relationships for
        feature_stats : Dict, optional
            Statistics for numeric features. If None, will use random stats.
            
        Returns:
        --------
        Dict
            Causal relationships between value ranges and target classes
        """
        # Get value range descriptions for the column
        value_range_descriptions = self.column_mapper.get_column_value_descriptions(column_name)
        
        # Join the descriptions for the prompt
        value_ranges_text = "\n".join([f"- {desc}" for desc in value_range_descriptions])
        
        # Generate the prompt with the column name and value ranges
        prompt = self.prompt_template.format(
            column_name=column_name,
            value_ranges=value_ranges_text
        )
        
        # Generate causal relationships using the text client
        try:
            if self.provider == "local":
                generated_text = self.text_client.generate(
                    prompt,
                    max_length=1024,
                    temperature=0.7,
                )
            elif self.provider == "gemini":
                # For Gemini, we pass parameters that will be mapped correctly by the client
                generated_text = self.text_client.generate(
                    prompt,
                    temperature=0.7,
                    max_tokens=1024,  # Will be mapped to max_output_tokens
                )
            else:
                # For other providers (Together, Anthropic)
                generated_text = self.text_client.generate(
                    prompt,
                    max_tokens=1024,
                    temperature=0.7,
                )
            
            # Parse the generated output
            causal_relationships = self._parse_generated_list(generated_text, column_name=column_name)
            
            # Return the causal relationships if they've been processed
            if column_name in self.statistical_causal_relationships:
                return self.statistical_causal_relationships[column_name]
            else:
                # If parsing failed, return a default structure
                default_relationships = {}
                for i, value_range in enumerate(value_range_descriptions):
                    default_relationships[value_range] = [
                        {
                            "class": f"class_{i}_1", 
                            "causal_explanation": f"Default causal relationship for {value_range}"
                        },
                        {
                            "class": f"class_{i}_2",
                            "causal_explanation": f"Secondary causal relationship for {value_range}"
                        }
                    ]
                return default_relationships
                
        except Exception as e:
            print(f"Error generating causal relationships for {column_name}: {e}")
            # Return empty dict if generation fails
            return {}
    
    def _generate_numeric_property_descriptors(self, column_name: str) -> List[str]:
        """
        Generate statistical property descriptors for numeric data.
        This is a fallback if the LLM can't generate appropriate descriptions.
        
        Args:
            column_name: The name of the column
            
        Returns:
            List of statistical property descriptors
        """
        # Basic statistical property descriptors
        magnitude_terms = [
            "high", "low", "medium", "large", "small", "tiny", "huge", "maximum", "minimum",
            "peak", "valley", "average", "median", "extreme", "moderate", "significant",
            "insignificant", "negligible", "considerable", "substantial"
        ]
        
        distribution_terms = [
            "normal", "skewed", "uniform", "bimodal", "multimodal", "outlier", "variance",
            "standard deviation", "percentile", "quartile", "distribution", "range", "spread",
            "concentration", "dispersion", "density", "sparse", "dense", "clustered", "scattered"
        ]
        
        trend_terms = [
            "increasing", "decreasing", "rising", "falling", "growing", "shrinking", "stable",
            "unstable", "volatile", "steady", "fluctuating", "spiking", "dipping", "surging",
            "plummeting", "accelerating", "decelerating", "seasonal", "cyclical", "trending"
        ]
        
        correlation_terms = [
            "correlated", "uncorrelated", "positively correlated", "negatively correlated",
            "inversely related", "directly related", "causal relationship", "correlation coefficient",
            "strong correlation", "weak correlation", "no correlation", "perfect correlation",
            "spurious correlation", "partial correlation", "conditional correlation"
        ]
        
        relative_terms = [
            "above average", "below average", "higher than expected", "lower than expected",
            "within normal range", "outside normal range", "above threshold", "below threshold",
            "top percentile", "bottom percentile", "leading indicator", "lagging indicator",
            "relative", "absolute", "comparative", "normalized", "raw", "adjusted", "standardized"
        ]
        
        # Combine and shuffle to create a varied list
        all_terms = magnitude_terms + distribution_terms + trend_terms + correlation_terms + relative_terms
        random.shuffle(all_terms)
        
        # Add column-name specific terms if possible
        column_specific = [
            f"{column_name} trend", 
            f"high {column_name}", 
            f"low {column_name}",
            f"{column_name} spike", 
            f"{column_name} dip",
            f"{column_name} average", 
            f"{column_name} outlier"
        ]
        
        all_terms = column_specific + all_terms
        return all_terms[:100]  # Return up to 100 terms
    
    def process_column(self, column_name: str, max_non_zero_tokens: Optional[int] = None, seq_length: int = 1152) -> torch.Tensor:
        """
        Process a column name to generate tokenized semantic values using the selected curation strategy.
        
        Args:
            column_name: Name of the column to process
            max_non_zero_tokens: Maximum number of non-zero tokens to return
            seq_length: Maximum sequence length
            
        Returns:
            Tensor of tokenized values
        """
        if max_non_zero_tokens is None:
            max_non_zero_tokens = self.max_tokens
            
        # Clean the column name
        column_name = column_name.replace("_", " ").replace("-", " ")
        
        # Handle statistical causal relationships differently
        if self.curation_strategy == 'statistical_causal_relationships':
            # Generate causal relationships for this column
            self.generate_statistical_causal_relationships(column_name)
            
            # Get the semantic values from the causal relationships if available
            if column_name in self.statistical_causal_relationships:
                causal_data = self.statistical_causal_relationships[column_name]
                
                # Extract terms from the causal relationships
                semantic_values = [column_name]
                
                # Add value ranges
                semantic_values.extend(causal_data.keys())
                
                # Add class names and causal explanations
                for value_range, class_info_list in causal_data.items():
                    for class_info in class_info_list:
                        if isinstance(class_info, dict):
                            if 'class' in class_info:
                                semantic_values.append(class_info['class'])
                            if 'causal_explanation' in class_info:
                                # Add explanation as individual terms
                                explanation = class_info['causal_explanation']
                                semantic_values.append(explanation)
            else:
                # Fallback if generation failed
                semantic_values = [column_name, "causal relationship", "statistical feature", "value range", "target class"]
        else:
            # Generate the appropriate prompt based on the curation strategy
            prompt = self.prompt_template.format(column_name=column_name)
            
            # Generate text using the selected client
            try:
                if self.provider == "local":
                    generated_text = self.text_client.generate(
                        prompt,
                        max_length=1024,
                        temperature=0.7,
                    )
                elif self.provider == "gemini":
                    # For Gemini, we pass parameters that will be mapped correctly by the client
                    generated_text = self.text_client.generate(
                        prompt,
                        temperature=0.7,
                        max_tokens=1024,  # Will be mapped to max_output_tokens
                    )
                else:
                    # For other providers (Together, Anthropic)
                    generated_text = self.text_client.generate(
                        prompt,
                        max_tokens=1024,
                        temperature=0.7,
                    )
                    
                # Parse the generated list, passing the column name for structured data storage
                semantic_values = self._parse_generated_list(generated_text, column_name=column_name)
                
                # If we're using the numeric_properties strategy and didn't get good results, use the fallback
                if self.curation_strategy == 'numeric_properties' and (not semantic_values or len(semantic_values) < 20):
                    semantic_values = self._generate_numeric_property_descriptors(column_name)
                    
            except Exception:
                # If generation fails, create a fallback based on the column name without printing error
                if self.curation_strategy == 'numeric_properties':
                    semantic_values = self._generate_numeric_property_descriptors(column_name)
                else:
                    semantic_values = [column_name, "data", "value", "column", "field"]
        
        # Continue with tokenization as in the original method
        column_name_tokens = self.clip_tokenizer(
            column_name,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_tokens,
            truncation=True
        ).input_ids[0]
        
        # Drop special CLIP tokens
        column_name_tokens = truncate_tensor(column_name_tokens)
        
        if not isinstance(semantic_values, list):
            return torch.cat([column_name_tokens, torch.zeros(self.max_tokens - len(column_name_tokens))])
        
        # Join the values into a string
        joined_values = ", ".join(
            [str(value) for value in semantic_values]
        )
        
        # Tokenize with CLIP
        tokens = self.clip_tokenizer(
            joined_values,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_tokens,
            truncation=True
        ).input_ids[0]  # Get the first (and only) sequence
        
        tokens = truncate_tensor(tokens)
        
        # Ensure we don't exceed max_non_zero_tokens for non-zero tokens
        if max_non_zero_tokens < self.max_tokens:
            non_zero_mask = tokens != 0
            non_zero_count = torch.sum(non_zero_mask).item()
            
            if non_zero_count > max_non_zero_tokens:
                # Find indices of non-zero tokens
                non_zero_indices = torch.nonzero(tokens).squeeze()
                
                # Create a mask to keep only the first max_non_zero_tokens non-zero tokens
                keep_mask = torch.zeros_like(tokens, dtype=torch.bool)
                keep_mask[non_zero_indices[:max_non_zero_tokens]] = True
                
                # Zero out tokens beyond the limit
                tokens = tokens * keep_mask
        
        return torch.cat([column_name_tokens, tokens])


# Example usage
if __name__ == "__main__":

    load_secrets()
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate tokenized semantic values for table column names with enhanced curation strategies")
    
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
        help="Model name for the specified provider. If not provided, uses a default model."
    )
    
    parser.add_argument(
        "--device", 
        type=str, 
        default=None,
        help="Computing device (only relevant for local provider)"
    )
    
    parser.add_argument(
        "--max-tokens", 
        type=int, 
        default=200,
        help="Maximum number of tokens to keep (lt/eq)"
    )
    
    parser.add_argument(
        "--max-non-zero-tokens", 
        type=int, 
        default=200,
        help="Maximum number of non-zero tokens to return (lt/eq)"
    )
    
    parser.add_argument(
        "--output-file", 
        type=str, 
        default=None,
        help="Path to save the tokenized features. If not provided, a timestamped filename will be used."
    )
    
    parser.add_argument(
        "--save-interval", 
        type=int, 
        default=100,
        help="Number of columns to process before saving intermediate progress (default: 100)"
    )
    
    parser.add_argument(
        "--log-file", 
        type=str, 
        default=None,
        help="Path to the log file tracking completed columns (default: strategy-specific JSON file)"
    )
    
    parser.add_argument(
        "--parallel",
        action="store_true",
        default=True,
        help="Use parallel processing for Gemini API requests (default: True)"
    )
    
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Maximum number of concurrent API requests (default: 10)"
    )
    
    parser.add_argument(
        "--curation-strategy",
        type=str,
        choices=list(PROMPT_TEMPLATES.keys()),
        default="standard",
        help=f"Curation strategy to use for semantic values (default: standard)"
    )
    
    parser.add_argument(
        "--clusters-save-path",
        type=str,
        default="structured_data.json",
        help="Path to save structured data (clusters/pairs/metadata descriptions) in JSON format. For conceptual_clusters, contrastive_pairs, and metadata_description strategies."
    )
    
    parser.add_argument(
        "--stat-causal-path",
        type=str,
        default="statistical_causal_relationships.json",
        help="Path to save statistical causal relationship data in JSON format. For statistical_causal_relationships strategy."
    )
    
    args = parser.parse_args()
    
    # Get some example column names from Schema.org
    column_names = SCHEMA_TYPES
    
    # For strategies that produce structured data, use strategy-specific filename if default is used
    if args.curation_strategy in ['conceptual_clusters', 'contrastive_pairs', 'metadata_description'] and args.clusters_save_path == "structured_data.json":
        # Create a more descriptive filename
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        args.clusters_save_path = os.path.join(os.getcwd(), f"{args.curation_strategy}_data_{timestamp}.json")
    
    # Set the log file based on curation strategy if not provided
    if args.log_file is None:
        # Use current working directory for the log file
        args.log_file = os.path.join(os.getcwd(), f"completed_columns_{args.curation_strategy}.json")
    
    # Initialize the enhanced tokenizer with curation strategy
    tokenizer = EnhancedColumnSemanticTokenizer(
        provider=args.provider,
        model=args.model,
        device=args.device,
        max_tokens=args.max_tokens,
        max_concurrent=args.max_concurrent,
        curation_strategy=args.curation_strategy,
        clusters_save_path=args.clusters_save_path,
        stat_causal_path=args.stat_causal_path
    )
    
    # Determine the output path
    if args.output_file:
        output_path = args.output_file
    else:
        now = datetime.now()
        date_time = now.strftime("%Y-%m-%d_%H-%M-%S")
        output_path = os.path.join(os.getcwd(), f"tokenized_semantic_features_{args.provider}_{args.curation_strategy}_{date_time}.pt")
    
    # Process columns - with parallel processing if requested and using Gemini
    if args.parallel and args.provider == "gemini":
        # Need to create and run the event loop
        import asyncio
        
        # Create a wrapper around the parent class's process_columns_parallel
        # This is needed because we need to modify that method for use with Enhanced tokenizer
        async def enhanced_process_columns_parallel(tokenizer, column_names, **kwargs):
            # Set flag to keep structured data between batch processing
            tokenizer.keep_structured_data = True
            """Modified version of process_columns_parallel for EnhancedColumnSemanticTokenizer"""
            
            batch_size = kwargs.get('batch_size', 10)
            max_non_zero_tokens = kwargs.get('max_non_zero_tokens', None)
            log_file = kwargs.get('log_file', 'completed_columns.json')
            save_path = kwargs.get('save_path', None)
            save_interval = kwargs.get('save_interval', 100)
            
            # We need to use the parent class's _get_expected_tensor_size method
            max_tensor_size = tokenizer._get_expected_tensor_size(column_names[0], max_non_zero_tokens)
            
            # Load the completion log if it exists
            log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), log_file)
            completed_columns = {}
            
            if os.path.exists(log_file_path):
                try:
                    with open(log_file_path, 'r') as f:
                        completed_columns = json.load(f)
                    print(f"Loaded completion log with {len(completed_columns)} entries")
                except json.JSONDecodeError:
                    print(f"Error parsing log file {log_file_path}, starting fresh")
                    completed_columns = {}
                    
            # Process first column if needed with the enhanced method
            sample_col = next((col for col in column_names if col not in completed_columns), column_names[0])
            if sample_col not in completed_columns:
                sample_tokens = tokenizer.process_column(sample_col, max_non_zero_tokens)
                # Save this result in the completed columns
                completed_columns[sample_col] = sample_tokens.tolist()
                with open(log_file_path, 'w') as f:
                    json.dump(completed_columns, f)
            
            # Filter out columns that have already been processed
            columns_to_process = [col for col in column_names if col not in completed_columns]
            print(f"Need to process {len(columns_to_process)} columns out of {len(column_names)} total")
            
            results = {}
            # Add already processed columns to results
            for col_name in column_names:
                if col_name in completed_columns:
                    tensor_data = torch.tensor(completed_columns[col_name])
                    # Ensure the tensor has the right size
                    if tensor_data.size(0) < max_tensor_size:
                        padding = torch.zeros(max_tensor_size - tensor_data.size(0), dtype=tensor_data.dtype)
                        tensor_data = torch.cat([tensor_data, padding])
                    elif tensor_data.size(0) > max_tensor_size:
                        tensor_data = tensor_data[:max_tensor_size]
                    results[col_name] = tensor_data
                    
            # Process remaining columns in batches
            for batch_start in range(0, len(columns_to_process), batch_size):
                batch_end = min(batch_start + batch_size, len(columns_to_process))
                batch_columns = columns_to_process[batch_start:batch_end]
                
                # Generate prompts for all columns in the batch using the enhanced prompt template
                prompts = []
                for col_name in batch_columns:
                    col_name_clean = col_name.replace("_", " ").replace("-", " ")
                    # Use the tokenizer's prompt template based on curation strategy
                    prompt = tokenizer.prompt_template.format(column_name=col_name_clean)
                    prompts.append(prompt)
                
                # Process the batch in parallel
                print(f"Processing batch of {len(batch_columns)} columns...")
                batch_responses = await tokenizer.text_client.generate_batch_async(
                    prompts,
                    temperature=0.7,
                    max_tokens=1024
                )
                
                # Process each response using enhanced tokenizer approach
                for i, (col_name, response_text) in enumerate(zip(batch_columns, batch_responses)):
                    try:
                        if isinstance(response_text, str) and response_text.startswith("Error:"):
                            print(f"Error processing column {col_name}: {response_text}")
                            # Even with errors, try to store something in structured_data
                            tokenizer.structured_data[col_name] = {"error": response_text}
                            col_tokens = torch.zeros(max_tensor_size, dtype=torch.long)
                        else:
                            # Parse the generated list with the column name for structured data storage
                            semantic_values = tokenizer._parse_generated_list(response_text, column_name=col_name)
                            
                            # Process the column name and semantic values as in process_column
                            col_name_clean = col_name.replace("_", " ").replace("-", " ")
                            column_name_tokens = tokenizer.clip_tokenizer(
                                col_name_clean,
                                return_tensors="pt",
                                padding="max_length",
                                max_length=tokenizer.max_tokens,
                                truncation=True
                            ).input_ids[0]
                            
                            column_name_tokens = truncate_tensor(column_name_tokens)
                            
                            if not isinstance(semantic_values, list):
                                col_tokens = torch.cat([column_name_tokens, torch.zeros(tokenizer.max_tokens - len(column_name_tokens))])
                            else:
                                joined_values = ", ".join([str(value) for value in semantic_values])
                                
                                tokens = tokenizer.clip_tokenizer(
                                    joined_values,
                                    return_tensors="pt",
                                    padding="max_length",
                                    max_length=tokenizer.max_tokens,
                                    truncation=True
                                ).input_ids[0]
                                
                                tokens = truncate_tensor(tokens)
                                
                                if max_non_zero_tokens and max_non_zero_tokens < tokenizer.max_tokens:
                                    non_zero_mask = tokens != 0
                                    non_zero_count = torch.sum(non_zero_mask).item()
                                    
                                    if non_zero_count > max_non_zero_tokens:
                                        non_zero_indices = torch.nonzero(tokens).squeeze()
                                        keep_mask = torch.zeros_like(tokens, dtype=torch.bool)
                                        keep_mask[non_zero_indices[:max_non_zero_tokens]] = True
                                        tokens = tokens * keep_mask
                                
                                col_tokens = torch.cat([column_name_tokens, tokens])
                                
                            # Ensure the tensor has the right size
                            if col_tokens.size(0) < max_tensor_size:
                                padding = torch.zeros(max_tensor_size - col_tokens.size(0), dtype=col_tokens.dtype)
                                col_tokens = torch.cat([col_tokens, padding])
                            elif col_tokens.size(0) > max_tensor_size:
                                col_tokens = col_tokens[:max_tensor_size]
                            
                            # Save in the completion log
                            completed_columns[col_name] = col_tokens.tolist()
                        
                        results[col_name] = col_tokens
                        
                    except Exception as e:
                        print(f"Error processing column {col_name}: {e}")
                        import traceback
                        traceback.print_exc()
                        col_tokens = torch.zeros(max_tensor_size, dtype=torch.long)
                        results[col_name] = col_tokens
                
                # Update the log file after each batch
                with open(log_file_path, 'w') as f:
                    json.dump(completed_columns, f)
                    
                # Explicitly save the structured data for this batch to ensure we don't lose anything
                # This calls _save_clusters_data with the current structured_data dictionary
                if hasattr(tokenizer, 'clusters_save_path') and tokenizer.clusters_save_path and hasattr(tokenizer, 'structured_data'):
                    tokenizer._save_clusters_data()
                
                # Save intermediate results if we've processed enough columns
                total_processed = batch_end + len([c for c in column_names if c in completed_columns and c not in columns_to_process])
                if save_path and total_processed % save_interval == 0:
                    try:
                        ordered_results = [results[col] for col in column_names if col in results]
                        
                        if ordered_results:
                            intermediate_results = torch.stack(ordered_results)
                            intermediate_save_path = f"{save_path}.partial_{total_processed}"
                            torch.save(intermediate_results, intermediate_save_path)
                            print(f"Saved intermediate progress ({total_processed}/{len(column_names)} columns) to {intermediate_save_path}")
                    except Exception as e:
                        print(f"Error saving intermediate results: {e}")
            
            # Build final results in the correct order
            final_results = []
            for col_name in column_names:
                if col_name in results:
                    final_results.append(results[col_name])
                else:
                    col_tokens = torch.zeros(max_tensor_size, dtype=torch.long)
                    final_results.append(col_tokens)
                    
            return torch.stack(final_results)
        
        if __name__ == "__main__":  # Protect against multiple process spawning
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            tokens_batch = loop.run_until_complete(
                enhanced_process_columns_parallel(
                    tokenizer,
                    column_names, 
                    max_non_zero_tokens=args.max_non_zero_tokens,
                    save_path=output_path,
                    save_interval=args.save_interval,
                    log_file=args.log_file,
                    batch_size=args.max_concurrent
                )
            )
    else:
        # Use the standard sequential processing
        tokens_batch = tokenizer.process_columns(
            column_names, 
            max_non_zero_tokens=args.max_non_zero_tokens,
            save_path=output_path,
            save_interval=args.save_interval,
            log_file=args.log_file
        )
    
    # Save the final results
    torch.save(tokens_batch, output_path)