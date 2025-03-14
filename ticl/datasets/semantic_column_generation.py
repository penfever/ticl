import torch
from transformers import pipeline, CLIPTokenizerFast
from typing import List, Optional, Dict, Any
import ast
import argparse
import os
import json
from tqdm import tqdm

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
    """Client for Google's Gemini API."""
    
    def __init__(self, model_name: str):
        """
        Initialize the Gemini client.
        
        Args:
            model_name: Gemini model name
        """
        try:
            import google.generativeai as genai
            
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable is not set")
            
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
        except ImportError:
            raise ImportError("Please install Google Generative AI Python SDK: pip install google-generativeai")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using Gemini API.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            The generated text
        """
        response = self.model.generate_content(prompt)
        return response.text


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
        max_tokens: int = 50
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

        # Set up the text generation client based on provider
        if provider == "local":
            self.text_client = LocalGenerationClient(model, self.device)
        elif provider == "gemini":
            self.text_client = GeminiClient(model)
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
        try:
            # Try to find and parse the Python list in the text
            start_idx = text.find("[")
            end_idx = text.rfind("]") + 1

            if start_idx != -1 and end_idx != -1:
                list_str = text[start_idx:end_idx]
                return ast.literal_eval(list_str)

            # Fallback: extract items one by one if list syntax is malformed
            import re
            items = re.findall(r'"([^"]*)"', text)
            if items:
                return items

            return []
        except (SyntaxError, ValueError) as e:
            print(f"Error parsing generated list: {e}")
            # Try a more lenient approach
            items = []
            for line in text.split('\n'):
                line = line.strip()
                if line.startswith('"') or line.startswith("'"):
                    # Extract quoted items
                    import re
                    match = re.search(r'[\'"]([^\'"]+)[\'"]', line)
                    if match:
                        items.append(match.group(1))
                elif line.startswith("-"):
                    # Extract items from bullet points
                    items.append(line[1:].strip())

            return items

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
        generated_text = self.text_client.generate(
            prompt,
            max_length=1024 if self.provider == "local" else None,
            max_tokens=1024 if self.provider != "local" else None,
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
        
        # Process a sample column to determine expected tensor size
        # First check if we have any completed columns to use as a reference
        max_tensor_size = 0
        if completed_columns:
            for tensor_data in completed_columns.values():
                max_tensor_size = max(max_tensor_size, len(tensor_data))
            print(f"Using max tensor size from completed columns: {max_tensor_size}")
        
        # If no completed columns or we need to set initial size, process first column
        if max_tensor_size == 0:
            sample_col = next((col for col in column_names if col not in completed_columns), column_names[0])
            sample_tokens = self.process_column(sample_col, max_non_zero_tokens)
            max_tensor_size = sample_tokens.size(0)
            print(f"Setting initial tensor size from sample column: {max_tensor_size}")
            # Save this result in the completed columns
            if sample_col not in completed_columns:
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


# Example usage
if __name__ == "__main__":
    from datetime import datetime
    from processed_schema_types import SCHEMA_TYPES

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
        max_tokens=args.max_tokens
    )
    
    # Determine the output path
    if args.output_file:
        output_path = args.output_file
    else:
        now = datetime.now()
        date_time = now.strftime("%Y-%m-%d_%H-%M-%S")
        output_path = f"tokenized_semantic_features_{args.provider}_{date_time}.pt"
    
    # Process multiple columns with periodic saving and logging
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