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
from tqdm import tqdm
from datetime import datetime

from ticl.datasets.processed_schema_types import SCHEMA_TYPES
from ticl.utils import load_secrets

def truncate_tensor(T, values_to_drop=[0, 49406, 49407]):
    """
    Takes a torch tensor T and returns a truncated tensor T_short where
    all indices with specified values have been dropped, retaining original order.

    Args:
        T (torch.Tensor): Input tensor
        values_to_drop (list): List of values to remove from tensor (default: [0, 49406, 49407])

    Returns:
        torch.Tensor: Truncated tensor with specified values removed
    """
    # Create a mask for values we want to keep
    mask = torch.ones_like(T, dtype=torch.bool)

    # Update mask for each value we want to drop
    for value in values_to_drop:
        mask &= (T != value)

    # Apply the mask to get the truncated tensor
    T_short = T[mask]

    return T_short


class TextGenerationClient:
    """Base class for text generation clients."""
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt."""
        raise NotImplementedError("Subclasses must implement this method")


class LocalGenerationClient(TextGenerationClient):
    """Client for local text generation using HuggingFace models."""
    
    def __init__(self, model_name: str, device: str):
        """
        Initialize the local generation client.
        
        Args:
            model_name: HuggingFace model ID for text generation
            device: Computing device ('cuda', 'cpu', etc.)
        """
        self.text_pipeline = pipeline(
            "text-generation",
            model=model_name,
            device_map=device,
            torch_dtype=torch.bfloat16,
        )
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using local HuggingFace model.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            The generated text
        """
        generation_kwargs = {
            "max_length": 1024,
            "num_return_sequences": 1,
            "do_sample": True,
            "temperature": 0.7,
        }
        generation_kwargs.update(kwargs)
        
        response = self.text_pipeline(
            prompt,
            **generation_kwargs
        )[0]['generated_text']
        
        return response


class GeminiClient(TextGenerationClient):
    """Client for Google's Gemini API with rate limiting and retry support."""
    
    def __init__(self, model_name: str, max_retries: int = 5, max_concurrent: int = 5):
        """
        Initialize the Gemini client.
        
        Args:
            model_name: Gemini model name
            max_retries: Maximum number of retries on failure
            max_concurrent: Maximum number of concurrent requests
        """
        try:
            import google.generativeai as genai
            
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable is not set")
            
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
            self.model_name = model_name
            self.api_key = api_key
            self.max_retries = max_retries
            self.max_concurrent = max_concurrent
            
            # Quota and rate limiting tracking
            self.request_count = 0
            self.error_count = 0
            self.retry_count = 0
            self.last_request_time = 0
            self.semaphore = None  # Will be initialized when needed
            
            # Initialize asyncio event loop access
            self.loop = None
        except ImportError:
            raise ImportError("Please install required packages: pip install google-generativeai aiohttp")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using Gemini API (synchronous version).
        
        Args:
            prompt: The input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            The generated text
        """
        # Add jitter to avoid exactly simultaneous requests
        jitter = 0.1 * random.random()
        time.sleep(jitter)
        
        for attempt in range(self.max_retries):
            try:
                # Track requests
                self.request_count += 1
                current_time = time.time()
                time_since_last = current_time - self.last_request_time
                self.last_request_time = current_time
                
                # If we're making requests too quickly, add a small delay
                if time_since_last < 0.05:  # 50ms
                    time.sleep(0.05 - time_since_last + jitter)
                
                # Gemini has specific parameter naming
                generation_config = {}
                
                # Map common parameters to Gemini-specific ones
                if 'temperature' in kwargs:
                    generation_config['temperature'] = kwargs.pop('temperature')
                
                # Handle max_tokens by mapping to Gemini's max_output_tokens
                if 'max_tokens' in kwargs:
                    generation_config['max_output_tokens'] = kwargs.pop('max_tokens')
                
                # Pass only supported parameters to generate_content
                response = self.model.generate_content(prompt, generation_config=generation_config)
                return response.text
                
            except Exception as e:
                self.error_count += 1
                error_message = str(e)
                
                # Check for rate limiting or quota errors
                if "429" in error_message or "quota" in error_message.lower():
                    # Exponential backoff with jitter
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    print(f"Rate limit or quota exceeded. Retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                    self.retry_count += 1
                    continue
                    
                # For other errors, retry with backoff
                if attempt < self.max_retries - 1:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    print(f"API error: {e}. Retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                    self.retry_count += 1
                else:
                    print(f"Failed after {self.max_retries} attempts: {e}")
                    raise
        
        raise Exception(f"Failed to generate content after {self.max_retries} attempts")

    async def _generate_async(self, prompt: str, **kwargs) -> str:
        """
        Generate text using Gemini API asynchronously.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            The generated text
        """
        import google.generativeai as genai
        
        # Add jitter to avoid exactly simultaneous requests
        jitter = 0.1 * random.random()
        await asyncio.sleep(jitter)
        
        # Initialize semaphore if needed
        if self.semaphore is None:
            self.semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async with self.semaphore:
            for attempt in range(self.max_retries):
                try:
                    # Track requests
                    self.request_count += 1
                    current_time = time.time()
                    time_since_last = current_time - self.last_request_time
                    self.last_request_time = current_time
                    
                    # If we're making requests too quickly, add a small delay
                    if time_since_last < 0.05:  # 50ms
                        await asyncio.sleep(0.05 - time_since_last + jitter)
                    
                    # Prepare Gemini-specific parameters
                    generation_config = {}
                    
                    # Map common parameters to Gemini-specific ones
                    if 'temperature' in kwargs:
                        generation_config['temperature'] = kwargs.pop('temperature')
                    
                    # Handle max_tokens by mapping to Gemini's max_output_tokens
                    if 'max_tokens' in kwargs:
                        generation_config['max_output_tokens'] = kwargs.pop('max_tokens')
                    
                    # Run the API call in a thread to avoid blocking the event loop
                    response = await asyncio.to_thread(
                        self.model.generate_content, 
                        prompt, 
                        generation_config=generation_config
                    )
                    return response.text
                    
                except Exception as e:
                    self.error_count += 1
                    error_message = str(e)
                    
                    # Check for rate limiting or quota errors
                    if "429" in error_message or "quota" in error_message.lower():
                        # Exponential backoff with jitter
                        wait_time = (2 ** attempt) + random.uniform(0, 1)
                        print(f"Rate limit or quota exceeded. Retrying in {wait_time:.2f} seconds...")
                        await asyncio.sleep(wait_time)
                        self.retry_count += 1
                        continue
                        
                    # For other errors, retry with backoff
                    if attempt < self.max_retries - 1:
                        wait_time = (2 ** attempt) + random.uniform(0, 1)
                        print(f"API error: {e}. Retrying in {wait_time:.2f} seconds...")
                        await asyncio.sleep(wait_time)
                        self.retry_count += 1
                    else:
                        print(f"Failed after {self.max_retries} attempts: {e}")
                        raise
            
            raise Exception(f"Failed to generate content after {self.max_retries} attempts")
    
    async def generate_batch_async(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Generate text for multiple prompts in parallel.
        
        Args:
            prompts: List of prompts to process
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated texts
        """
        # Create tasks for all prompts
        tasks = []
        for prompt in prompts:
            # Add some jitter to the task creation to avoid thundering herd
            await asyncio.sleep(random.uniform(0, 0.1))
            tasks.append(self._generate_async(prompt, **kwargs))
        
        # Run all tasks concurrently and collect results
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results, retrying failed tasks if needed
        final_results = []
        for result in results:
            if isinstance(result, Exception):
                # If a task failed, return an error message
                final_results.append(f"Error: {str(result)}")
            else:
                final_results.append(result)
        
        # Print stats
        print(f"Completed batch of {len(prompts)} requests.")
        print(f"Total requests: {self.request_count}, Errors: {self.error_count}, Retries: {self.retry_count}")
        
        return final_results


class TogetherClient(TextGenerationClient):
    """Client for Together AI API."""
    
    def __init__(self, model_name: str):
        """
        Initialize the Together AI client.
        
        Args:
            model_name: Together AI model name
        """
        try:
            import together
            
            api_key = os.getenv("TOGETHER_API_KEY")
            if not api_key:
                raise ValueError("TOGETHER_API_KEY environment variable is not set")
            
            together.api_key = api_key
            self.model_name = model_name
        except ImportError:
            raise ImportError("Please install Together AI Python SDK: pip install together")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using Together AI API.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            The generated text
        """
        import together
        
        generation_kwargs = {
            "max_tokens": 1024,
            "temperature": 0.7,
        }
        generation_kwargs.update(kwargs)
        
        response = together.Complete.create(
            prompt=prompt,
            model=self.model_name,
            **generation_kwargs
        )
        
        return response["output"]["choices"][0]["text"]


class AnthropicClient(TextGenerationClient):
    """Client for Anthropic's Claude API."""
    
    def __init__(self, model_name: str):
        """
        Initialize the Anthropic client.
        
        Args:
            model_name: Anthropic model name
        """
        try:
            import anthropic
            
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
            
            self.client = anthropic.Anthropic(api_key=api_key)
            self.model_name = model_name
        except ImportError:
            raise ImportError("Please install Anthropic Python SDK: pip install anthropic")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using Anthropic's Claude API.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            The generated text
        """
        import anthropic
        
        generation_kwargs = {
            "max_tokens": 1024,
            "temperature": 0.7,
        }
        generation_kwargs.update(kwargs)
        
        message = self.client.messages.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            **generation_kwargs
        )
        
        return message.content[0].text

class ColumnSemanticTokenizer:
    """
    A class that generates tokenized semantic values for table column names.
    """

    def __init__(
        self,
        provider: str = "local",
        model: Optional[str] = None,
        clip_tokenizer: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        max_tokens: int = 50,
        max_concurrent: int = 5
    ):
        """
        Initialize the tokenizer with specified models and parameters.

        Args:
            provider: Model provider ('local', 'gemini', 'together', or 'anthropic')
            model: Model name for the specified provider. If None, uses a default model.
            clip_tokenizer: HuggingFace model ID for CLIP tokenizer
            device: Computing device ('cuda', 'rocm', 'mps', or 'cpu')
            max_tokens: Maximum number of tokens to keep
        """
        # Determine the device to use for local models
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch, 'xpu') and torch.xpu.is_available():
                self.device = "rocm"  # For ROCm (AMD GPUs)
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"  # For Apple Silicon
            else:
                self.device = "cpu"
        else:
            self.device = device

        # Set default models based on provider
        provider_defaults = {
            "local": "Qwen/Qwen2.5-7B-Instruct",
            "gemini": "gemini-2.0-flash",
            #google/gemma-2-27b-it
            #mistralai/Mixtral-8x7B-Instruct-v0.1
            "together": "google/gemma-2-27b-it",
            "anthropic": "claude-3-sonnet-20240229"
        }
        
        if model is None:
            model = provider_defaults.get(provider, provider_defaults["local"])

        # Store the max_concurrent parameter
        self.max_concurrent = max_concurrent
        
        # Set up the text generation client based on provider
        if provider == "local":
            self.text_client = LocalGenerationClient(model, self.device)
        elif provider == "gemini":
            self.text_client = GeminiClient(model, max_concurrent=self.max_concurrent)
        elif provider == "together":
            self.text_client = TogetherClient(model)
        elif provider == "anthropic":
            self.text_client = AnthropicClient(model)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        # Set up the CLIP tokenizer
        self.clip_tokenizer = CLIPTokenizerFast.from_pretrained(clip_tokenizer)

        # Store the max tokens parameter
        self.max_tokens = max_tokens
        
        # Store provider and model for reference
        self.provider = provider
        self.model = model

    def _parse_generated_list(self, text: str) -> List[str]:
        """
        Parse the generated text to extract a list of semantic values.

        Args:
            text: The generated text containing a Python list

        Returns:
            List of extracted semantic values
        """
        # First try: the "proper" way - find a Python list and parse it
        try:
            # Clean the text
            clean_text = text.replace('```python', '').replace('```', '')
            
            # Look for list brackets
            start_idx = clean_text.find("[")
            end_idx = clean_text.rfind("]") + 1
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                list_str = clean_text[start_idx:end_idx]
                # Try to parse as literal Python list
                return ast.literal_eval(list_str)
        except Exception as e:
            # Log but continue to next method if this fails
            pass
            
        # Second try: Look for quoted strings
        try:
            items = re.findall(r'["\'](.*?)["\']', text)
            if items and len(items) > 5:  # Only return if we found a substantial number
                return items
        except Exception:
            pass
            
        # Third try: Find bullet points or numbered items
        try:
            bullet_items = []
            for line in text.split('\n'):
                line = line.strip()
                if line.startswith('-') or line.startswith('*') or re.match(r'^\d+\.', line):
                    # Extract content after the bullet/number
                    content = re.sub(r'^[-*]|\d+\.\s*', '', line).strip()
                    if content and len(content) > 1:
                        bullet_items.append(content)
            
            if bullet_items and len(bullet_items) > 5:
                return bullet_items
        except Exception:
            pass
            
        # Fourth try: Just extract comma-separated chunks from entire text
        try:
            # If we see commas, try splitting by them
            if ',' in text:
                items = [item.strip() for item in text.split(',')]
                # Filter out items that are too short or look like noise
                filtered_items = [item for item in items if len(item) > 1 and not item.startswith(('```', '#', '='))]
                if filtered_items and len(filtered_items) > 5:
                    return filtered_items
        except Exception:
            pass
        
        # Last resort: return a default list with column name variations
        print(f"Warning: Could not parse list format from response, using default values")
        return ["value1", "value2", "value3", "example", "sample"]

    def process_column(
        self,
        column_name: str,
        max_non_zero_tokens: Optional[int] = None,
        seq_length: int = 1152,
    ) -> torch.Tensor:
        """
        Process a column name to generate tokenized semantic values.

        Args:
            column_name: Name of the column to process
            max_non_zero_tokens: Maximum number of non-zero tokens to return
                                (defaults to self.max_tokens)
            seq_length: Maximum sequence length

        Returns:
            Tensor of tokenized values
        """
        if max_non_zero_tokens is None:
            max_non_zero_tokens = self.max_tokens
        column_name = column_name.replace("_", " ").replace("-", " ")
        # Generate prompt for text generation
        prompt = f"""
                  {column_name} is the name of column in a table found on the web. 
                  Please give me a Python formatted list of around 100 unique values (words, integers, floating point numbers,
                  phrases, dates, stringifed numbers like zipcodes, etc) that would be likely to appear in this column with this name.
                  Where appropriate, also include synonyms, antonyms, hierarchically related concepts, and non-English translations. Be creative!
                  If {column_name} doesn't mean anything to you, return an empty list.
                  """

        # Generate text using the selected client
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

        # Parse the generated list
        semantic_values = self._parse_generated_list(generated_text)

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

    def _get_expected_tensor_size(self, column_name: str, max_non_zero_tokens: Optional[int] = None) -> int:
        """
        Calculate the expected tensor size based on the column name and max_non_zero_tokens.
        
        Args:
            column_name: The name of the column
            max_non_zero_tokens: Maximum number of non-zero tokens
            
        Returns:
            Expected tensor size
        """
        if max_non_zero_tokens is None:
            max_non_zero_tokens = self.max_tokens
            
        # Clean the column name for tokenization
        column_name_clean = column_name.replace("_", " ").replace("-", " ")
        
        # Get the tokenized column name length
        column_name_tokens = self.clip_tokenizer(
            column_name_clean,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_tokens,
            truncation=True
        ).input_ids[0]
        column_name_tokens = truncate_tensor(column_name_tokens)
        
        # Expected tensor size is name tokens + max_non_zero_tokens
        return len(column_name_tokens) + max_non_zero_tokens
    
    def process_columns(
        self,
        column_names: List[str],
        max_non_zero_tokens: Optional[int] = None,
        save_path: Optional[str] = None,
        save_interval: int = 100,
        log_file: str = "completed_columns.json"
    ) -> torch.Tensor:
        """
        Process multiple column names and stack the results.

        Args:
            column_names: List of column names to process
            max_non_zero_tokens: Maximum number of non-zero tokens to return per column
            save_path: Path to save intermediate results
            save_interval: Number of columns to process before saving intermediate progress
            log_file: Path to the log file tracking completed columns

        Returns:
            Tensor of stacked tokenized values for all columns
        """
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
        
        # Determine tensor size based on max_non_zero_tokens parameter
        sample_col = column_names[0]
        max_tensor_size = self._get_expected_tensor_size(sample_col, max_non_zero_tokens)
        print(f"Using tensor size of {max_tensor_size} based on max_non_zero_tokens={max_non_zero_tokens}")
        
        # Process first column if needed
        sample_col = next((col for col in column_names if col not in completed_columns), column_names[0])
        if sample_col not in completed_columns:
            sample_tokens = self.process_column(sample_col, max_non_zero_tokens)
            # Save this result in the completed columns
            completed_columns[sample_col] = sample_tokens.tolist()
            with open(log_file_path, 'w') as f:
                json.dump(completed_columns, f)
        
        results = []
        for i, col_name in enumerate(tqdm(column_names, desc="Processing columns", total=len(column_names))):
            # Check if column has already been processed
            if col_name in completed_columns:
                print(f"Using cached result for column: {col_name}")
                col_tokens = torch.tensor(completed_columns[col_name])
            else:
                # Process the column
                try:
                    col_tokens = self.process_column(col_name, max_non_zero_tokens)
                    # Save in the completion log
                    completed_columns[col_name] = col_tokens.tolist()
                    # Update the log file after each successful column
                    with open(log_file_path, 'w') as f:
                        json.dump(completed_columns, f)
                except Exception as e:
                    print(f"Error processing column {col_name}: {e}")
                    # Create an empty tensor as a placeholder
                    col_tokens = torch.zeros(max_tensor_size, dtype=torch.long)
            
            # Ensure the tensor has the right size (pad or truncate as needed)
            if col_tokens.size(0) < max_tensor_size:
                padding = torch.zeros(max_tensor_size - col_tokens.size(0), dtype=col_tokens.dtype)
                col_tokens = torch.cat([col_tokens, padding])
            elif col_tokens.size(0) > max_tensor_size:
                col_tokens = col_tokens[:max_tensor_size]
            
            results.append(col_tokens)
            
            # Save intermediate results every save_interval columns
            if save_path and (i + 1) % save_interval == 0:
                try:
                    intermediate_results = torch.stack(results)
                    intermediate_save_path = f"{save_path}.partial_{i+1}"
                    torch.save(intermediate_results, intermediate_save_path)
                    print(f"Saved intermediate progress ({i+1}/{len(column_names)} columns) to {intermediate_save_path}")
                except Exception as e:
                    print(f"Error saving intermediate results: {e}")

        return torch.stack(results)
    
    async def process_columns_parallel(
        self,
        column_names: List[str],
        max_non_zero_tokens: Optional[int] = None,
        save_path: Optional[str] = None,
        save_interval: int = 100,
        log_file: str = "completed_columns.json",
        batch_size: int = 10
    ) -> torch.Tensor:
        """
        Process multiple column names in parallel using async API calls.
        
        Args:
            column_names: List of column names to process
            max_non_zero_tokens: Maximum number of non-zero tokens to return per column
            save_path: Path to save intermediate results
            save_interval: Number of columns to process before saving intermediate progress
            log_file: Path to the log file tracking completed columns
            batch_size: Number of columns to process in each batch
            
        Returns:
            Tensor of stacked tokenized values for all columns
        """
        if not isinstance(self.text_client, GeminiClient):
            raise ValueError("Parallel processing is only supported for the Gemini provider")
            
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
        
                # Determine tensor size based on max_non_zero_tokens parameter
        sample_col = column_names[0]
        max_tensor_size = self._get_expected_tensor_size(sample_col, max_non_zero_tokens)
        print(f"Using tensor size of {max_tensor_size} based on max_non_zero_tokens={max_non_zero_tokens}")
        
        # Process first column if needed
        sample_col = next((col for col in column_names if col not in completed_columns), column_names[0])
        if sample_col not in completed_columns:
            sample_tokens = self.process_column(sample_col, max_non_zero_tokens)
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
            
            # Generate prompts for all columns in the batch
            prompts = []
            for col_name in batch_columns:
                col_name_clean = col_name.replace("_", " ").replace("-", " ")
                prompt = f"""
                          {col_name_clean} is the name of column in a table found on the web. 
                          Please give me a Python formatted list of around 100 unique values (words, integers, floating point numbers,
                          phrases, dates, stringifed numbers like zipcodes, etc) that would be likely to appear in this column with this name.
                          Where appropriate, also include synonyms, antonyms, hierarchically related concepts, and non-English translations. Be creative!
                          If {col_name_clean} doesn't mean anything to you, return an empty list.
                          """
                prompts.append(prompt)
            
            # Process the batch in parallel
            print(f"Processing batch of {len(batch_columns)} columns...")
            # Pass both temperature and max_tokens to be correctly mapped
            batch_responses = await self.text_client.generate_batch_async(
                prompts,
                temperature=0.7,
                max_tokens=1024  # Will be mapped to max_output_tokens
            )
            
            # Process each response
            for i, (col_name, response_text) in enumerate(zip(batch_columns, batch_responses)):
                try:
                    # Check if response contains an error
                    if isinstance(response_text, str) and response_text.startswith("Error:"):
                        print(f"Error processing column {col_name}: {response_text}")
                        # Create an empty tensor as a placeholder with the expected size
                        col_tokens = torch.zeros(max_tensor_size, dtype=torch.long)
                    else:
                        # Parse the generated list
                        semantic_values = self._parse_generated_list(response_text)
                        
                        # Process the column name
                        col_name_clean = col_name.replace("_", " ").replace("-", " ")
                        column_name_tokens = self.clip_tokenizer(
                            col_name_clean,
                            return_tensors="pt",
                            padding="max_length",
                            max_length=self.max_tokens,
                            truncation=True
                        ).input_ids[0]
                        
                        # Drop special CLIP tokens
                        column_name_tokens = truncate_tensor(column_name_tokens)
                        
                        if not isinstance(semantic_values, list):
                            col_tokens = torch.cat([column_name_tokens, torch.zeros(self.max_tokens - len(column_name_tokens))])
                        else:
                            # Join the values into a string
                            joined_values = ", ".join([str(value) for value in semantic_values])
                            
                            # Tokenize with CLIP
                            tokens = self.clip_tokenizer(
                                joined_values,
                                return_tensors="pt",
                                padding="max_length",
                                max_length=self.max_tokens,
                                truncation=True
                            ).input_ids[0]
                            
                            tokens = truncate_tensor(tokens)
                            
                            # Ensure we don't exceed max_non_zero_tokens for non-zero tokens
                            if max_non_zero_tokens and max_non_zero_tokens < self.max_tokens:
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
                    # Create an empty tensor as a placeholder
                    col_tokens = torch.zeros(max_tensor_size, dtype=torch.long)
                    results[col_name] = col_tokens
            
            # Update the log file after each batch
            with open(log_file_path, 'w') as f:
                json.dump(completed_columns, f)
            
            # Save intermediate results if we've processed enough columns
            total_processed = batch_end + len([c for c in column_names if c in completed_columns and c not in columns_to_process])
            if save_path and total_processed % save_interval == 0:
                try:
                    # Build a list of tensors in the same order as column_names
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
                # This should not happen, but just in case
                col_tokens = torch.zeros(max_tensor_size, dtype=torch.long)
                final_results.append(col_tokens)
        
        return torch.stack(final_results)


# Example usage
if __name__ == "__main__":

    load_secrets()
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate tokenized semantic values for table column names")
    
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
        default=201,
        help="Maximum number of tokens to keep (lt)"
    )
    
    parser.add_argument(
        "--max-non-zero-tokens", 
        type=int, 
        default=201,
        help="Maximum number of non-zero tokens to return (lt)"
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
        default="completed_columns.json",
        help="Path to the log file tracking completed columns (default: completed_columns.json)"
    )
    
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Use parallel processing for Gemini API requests"
    )
    
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Maximum number of concurrent API requests (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Get some example column names from Schema.org
    column_names = SCHEMA_TYPES
    
    print(f"Using provider: {args.provider}")
    print(f"Model: {args.model if args.model else 'default for provider'}")
    
    # Initialize the tokenizer
    tokenizer = ColumnSemanticTokenizer(
        provider=args.provider,
        model=args.model,
        device=args.device,
        max_tokens=args.max_tokens,
        max_concurrent=args.max_concurrent
    )
    
    # Determine the output path
    if args.output_file:
        output_path = args.output_file
    else:
        now = datetime.now()
        date_time = now.strftime("%Y-%m-%d_%H-%M-%S")
        output_path = f"tokenized_semantic_features_{args.provider}_{date_time}.pt"
    
    # Process columns - with parallel processing if requested and using Gemini
    if args.parallel and args.provider == "gemini":
        print(f"Using parallel processing with max {args.max_concurrent} concurrent requests")
        # Need to create and run the event loop
        import asyncio
        
        if __name__ == "__main__":  # Protect against multiple process spawning
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            tokens_batch = loop.run_until_complete(
                tokenizer.process_columns_parallel(
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
    print(f"Final tokenized features saved to {output_path}")