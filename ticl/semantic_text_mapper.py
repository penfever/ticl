"""
Semantic Text Mapper for bridging text descriptions and learned semantic classes.

This module provides functionality to:
1. Map text descriptions to semantic representations
2. Find closest matching classes for text descriptions
3. Generate new class boundaries from text descriptions
"""

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Tuple, Union, Optional
import random

class SemanticTextMapper:
    """
    Maps between text descriptions and semantic class representations.
    
    This class provides methods to:
    - Encode text descriptions into semantic space
    - Map text to existing class representations
    - Generate new class boundaries from text descriptions
    """
    
    def __init__(
        self, 
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None
    ):
        """
        Initialize the text mapper with a sentence embedding model.
        
        Parameters:
        -----------
        model_name : str
            HuggingFace model name for the text embedding model
        device : str, optional
            Device to run the model on (auto-detected if None)
        """
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = device
            
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
        # Cache of text embeddings
        self.embedding_cache = {}
        
    def encode_text(self, text: Union[str, List[str]]) -> torch.Tensor:
        """
        Encode text into the semantic embedding space.
        
        Parameters:
        -----------
        text : str or List[str]
            Text or list of texts to encode
            
        Returns:
        --------
        torch.Tensor
            Embedded representation(s) of the text
        """
        # Handle single text or list
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text
            
        # Check cache for existing embeddings
        uncached_texts = [t for t in texts if t not in self.embedding_cache]
        
        # If we have new texts to encode
        if uncached_texts:
            # Tokenize
            inputs = self.tokenizer(
                uncached_texts, 
                padding=True, 
                truncation=True, 
                return_tensors="pt"
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Use mean pooling for sentence embeddings
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            
            # Multiply token embeddings by attention mask to ignore padding
            masked_embeddings = token_embeddings * attention_mask.unsqueeze(-1)
            
            # Sum tokens and divide by actual sequence length
            summed = torch.sum(masked_embeddings, dim=1)
            seq_lengths = torch.sum(attention_mask, dim=1, keepdim=True)
            embeddings = summed / seq_lengths
            
            # Normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            # Update cache
            for i, text in enumerate(uncached_texts):
                self.embedding_cache[text] = embeddings[i].cpu()
                
        # Retrieve all embeddings from cache
        result_embeddings = torch.stack([self.embedding_cache[t] for t in texts])
        
        # Return single embedding or batch
        if len(texts) == 1:
            return result_embeddings.squeeze(0)
        return result_embeddings
    
    def map_text_to_class(
        self, 
        text: str, 
        class_token_patterns: Dict, 
        semantic_data: torch.Tensor,
        similarity_threshold: float = 0.5
    ) -> Tuple[int, float]:
        """
        Map a text description to the closest matching class.
        
        Parameters:
        -----------
        text : str
            Text description to map to a class
        class_token_patterns : dict
            Dictionary mapping class indices to token patterns
        semantic_data : torch.Tensor
            Tensor of semantic token data
        similarity_threshold : float
            Minimum similarity threshold to consider a match
            
        Returns:
        --------
        Tuple[int, float]
            Tuple of (best_class_index, similarity_score)
            Returns (-1, 0.0) if no match above threshold
        """
        # Encode the input text
        text_embedding = self.encode_text(text)
        
        best_class = -1
        best_similarity = 0.0
        
        # For each class, compute similarity to its semantic pattern
        for class_idx, pattern_info in class_token_patterns.items():
            semantic_class = pattern_info['semantic_class']
            token_indices = pattern_info['token_indices']
            
            # Get tokens for this class from semantic data
            semantic_tokens = semantic_data[semantic_class, token_indices]
            
            # Convert tokens to text embedding space (simplified mapping)
            # In a real implementation, you'd want a more sophisticated mapping
            # between token values and the text embedding space
            token_mean = semantic_tokens.float().mean()
            # Create a tensor of the same shape as text_embedding filled with the mean value
            token_embedding = torch.full_like(text_embedding, token_mean)
            
            # Compute similarity (no need to unsqueeze for vectors)
            similarity = F.cosine_similarity(text_embedding, token_embedding, dim=0).item()
            
            # Update best match if better
            if similarity > best_similarity:
                best_similarity = similarity
                best_class = class_idx
        
        # Check if similarity exceeds threshold
        if best_similarity < similarity_threshold:
            return -1, 0.0
            
        return best_class, best_similarity
    
    def generate_class_boundaries(
        self, 
        class_descriptions: Dict[str, str], 
        semantic_data: torch.Tensor,
        num_features_per_class: int = 5,
        feature_significance: float = 0.7
    ) -> Dict[str, Dict]:
        """
        Generate new class boundaries from text descriptions.
        
        Parameters:
        -----------
        class_descriptions : Dict[str, str]
            Dictionary mapping class names to text descriptions
        semantic_data : torch.Tensor
            Tensor of semantic token data
        num_features_per_class : int
            Number of semantic features to assign per class
        feature_significance : float
            Significance level for semantic features (0-1)
            
        Returns:
        --------
        Dict[str, Dict]
            Dictionary with class tokens and boundaries
        """
        # Encode all class descriptions
        class_embeddings = {}
        for class_name, description in class_descriptions.items():
            class_embeddings[class_name] = self.encode_text(description)
        
        # Number of semantic classes
        num_semantic_classes = semantic_data.shape[0]
        
        # Initialize class token patterns
        class_token_patterns = {}
        
        # For each class, select appropriate tokens
        for i, (class_name, embedding) in enumerate(class_embeddings.items()):
            # Select a semantic class 
            semantic_class = i % num_semantic_classes
            
            # Select a subset of tokens for this class
            num_tokens = min(10, semantic_data.shape[1])
            token_indices = random.sample(range(semantic_data.shape[1]), num_tokens)
            
            # Extract tokens for this class
            tokens = semantic_data[semantic_class, token_indices]
            
            # Store the class token pattern
            class_token_patterns[class_name] = {
                'semantic_class': semantic_class,
                'token_indices': token_indices,
                'tokens': tokens,
                'embedding': embedding,
                'features': random.sample(range(50), num_features_per_class),  # Sample from 50 semantic features
                'significance': feature_significance
            }
        
        return class_token_patterns
    
    def apply_text_boundaries(
        self, 
        x: torch.Tensor,
        class_token_patterns: Dict[str, Dict]
    ) -> torch.Tensor:
        """
        Apply generated class boundaries to feature tensor.
        
        Parameters:
        -----------
        x : torch.Tensor
            Feature tensor with shape [samples, batch, features]
        class_token_patterns : Dict[str, Dict]
            Dictionary with class tokens and boundaries
            
        Returns:
        --------
        torch.Tensor
            Class predictions with shape [samples, batch]
        """
        # Get the device of the input tensor
        device = x.device
        
        # Initialize predictions with -1 (unknown class)
        predictions = torch.full((x.shape[0], x.shape[1]), -1, dtype=torch.float, device=device)
        
        # For each sample and batch
        for s in range(x.shape[0]):
            for b in range(x.shape[1]):
                sample_features = x[s, b]
                
                # Score for each class
                class_scores = {}
                
                # Calculate score for each class
                for class_name, pattern in class_token_patterns.items():
                    features = pattern['features']
                    tokens = pattern['tokens'].to(device)
                    
                    # Extract semantic features for this sample
                    semantic_values = sample_features[features]
                    
                    # Count matches with class tokens
                    matches = 0
                    for token in tokens:
                        if any((semantic_values - token).abs() < 1e-5):
                            matches += 1
                    
                    # Calculate score
                    score = matches / len(tokens) * pattern['significance']
                    class_scores[class_name] = score
                
                # Assign the highest scoring class
                if class_scores:
                    best_class = max(class_scores.items(), key=lambda x: x[1])
                    if best_class[1] > 0.2:  # Minimum score threshold
                        # Convert class name to index
                        class_idx = list(class_token_patterns.keys()).index(best_class[0])
                        predictions[s, b] = class_idx
        
        return predictions