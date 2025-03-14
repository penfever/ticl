"""
Text-based classifier interface for TiCL models.

This module provides a simplified interface for classifying tabular data
using text descriptions instead of learning from labeled examples.
"""

import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Tuple, Optional

from ticl.models.semantic_aware_model import SemanticAwareClassifier
from ticl.semantic_text_mapper import SemanticTextMapper


class TextualClassifier:
    """
    Interface for classifying tabular data using text descriptions.
    
    This class wraps a semantic-aware model to provide:
    1. Classification using text descriptions
    2. Generation of class boundaries from text descriptions
    """
    
    def __init__(
        self, 
        model: SemanticAwareClassifier,
        semantic_data: torch.Tensor,
        device: Optional[str] = None,
        use_mixed_precision: bool = True
    ):
        """
        Initialize the textual classifier.
        
        Parameters:
        -----------
        model : SemanticAwareClassifier
            The semantic-aware model to use for classification
        semantic_data : torch.Tensor
            Semantic token data used during training
        device : str, optional
            Device to run inference on (auto-detected if None)
        use_mixed_precision : bool
            Whether to use mixed precision (FP16) for inference to save memory
        """
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = device
        
        # Enable mixed precision for memory efficiency (CUDA only)
        self.use_mixed_precision = use_mixed_precision and torch.cuda.is_available() and self.device == "cuda"
            
        # Store model and semantic data (keep semantic data on CPU to save GPU memory)
        self.model = model.to(self.device)
        self.model.eval()
        
        # Ensure semantic data is int32 to save memory and stays on CPU
        if semantic_data.dtype != torch.int32 and semantic_data.dtype != torch.int64:
            semantic_data = semantic_data.to(dtype=torch.int32)
        self.semantic_data = semantic_data.to("cpu")  # Always store on CPU to save GPU memory
        
        # Initialize text mapper - keep on CPU for text processing
        self.text_mapper = SemanticTextMapper(device="cpu")
        
        # Clean up GPU memory after initialization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    def preprocess_data(
        self, 
        data: Union[pd.DataFrame, np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """
        Preprocess input data into model format.
        
        Parameters:
        -----------
        data : Union[pd.DataFrame, np.ndarray, torch.Tensor]
            Input data to preprocess
            
        Returns:
        --------
        torch.Tensor
            Preprocessed data in model format
        """
        # Convert to torch tensor if needed
        if isinstance(data, pd.DataFrame):
            x = torch.tensor(data.values, dtype=torch.float32)
        elif isinstance(data, np.ndarray):
            x = torch.tensor(data, dtype=torch.float32)
        elif isinstance(data, torch.Tensor):
            x = data
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        # Add batch dimension if needed
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        # Add samples dimension if needed (unsupervised tabular format)
        if x.dim() == 2:
            x = x.unsqueeze(0).transpose(0, 1)
            
        # Move to device
        x = x.to(self.device)
        
        return x
    
    def classify_with_text(
        self, 
        data: Union[pd.DataFrame, np.ndarray, torch.Tensor],
        text_description: str
    ) -> Tuple[torch.Tensor, float]:
        """
        Classify data using a text description.
        Memory-efficient implementation that uses mixed precision and CPU offloading.
        
        Parameters:
        -----------
        data : Union[pd.DataFrame, np.ndarray, torch.Tensor]
            Input data to classify
        text_description : str
            Text description of the class
            
        Returns:
        --------
        Tuple[torch.Tensor, float]
            Tuple of (class_predictions, similarity_score)
        """
        # Preprocess data
        x = self.preprocess_data(data)
        
        # IMPORTANT: Don't load all semantic data to GPU at once - the model actually 
        # doesn't use it directly. For backward compatibility, we still pass it,
        # but keep it on CPU where it's actually accessed.
        
        # Get predictions based on text - use mixed precision if enabled
        try:
            with torch.no_grad():
                if self.use_mixed_precision and torch.cuda.is_available() and x.device.type == 'cuda':
                    # CUDA with mixed precision
                    with torch.cuda.amp.autocast():
                        results = self.model.predict_from_text(
                            x, 
                            text_description, 
                            self.semantic_data,  # Keep on CPU
                            self.text_mapper
                        )
                else:
                    # Standard precision for MPS/CPU or when mixed precision is disabled
                    results = self.model.predict_from_text(
                        x, 
                        text_description, 
                        self.semantic_data,  # Keep on CPU
                        self.text_mapper
                    )
                
        finally:
            # Explicit cleanup after prediction
            if x.device.type != 'cpu':
                x = x.cpu()
                del x
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return results['class_preds'], results['similarity']
    
    def classify_with_descriptions(
        self, 
        data: Union[pd.DataFrame, np.ndarray, torch.Tensor],
        class_descriptions: Dict[str, str]
    ) -> Tuple[torch.Tensor, Dict[int, str]]:
        """
        Classify data using multiple class descriptions.
        Memory-efficient implementation that uses mixed precision and CPU offloading.
        
        Parameters:
        -----------
        data : Union[pd.DataFrame, np.ndarray, torch.Tensor]
            Input data to classify
        class_descriptions : Dict[str, str]
            Dictionary mapping class names to text descriptions
            
        Returns:
        --------
        Tuple[torch.Tensor, Dict[int, str]]
            Tuple of (class_predictions, class_mapping)
        """
        # Preprocess data
        x = self.preprocess_data(data)
        
        # IMPORTANT: Don't load all semantic data to GPU at once - the model actually 
        # doesn't use it directly. For backward compatibility, we still pass it,
        # but keep it on CPU where it's actually accessed.
        
        # Generate boundaries and classify - use mixed precision if enabled
        try:
            with torch.no_grad():
                if self.use_mixed_precision and torch.cuda.is_available() and x.device.type == 'cuda':
                    # CUDA with mixed precision
                    with torch.cuda.amp.autocast():
                        results = self.model.generate_boundaries_from_text(
                            x,
                            class_descriptions,
                            self.semantic_data,  # Keep on CPU
                            self.text_mapper
                        )
                else:
                    # Standard precision for MPS/CPU or when mixed precision is disabled
                    results = self.model.generate_boundaries_from_text(
                        x,
                        class_descriptions,
                        self.semantic_data,  # Keep on CPU
                        self.text_mapper
                    )
        finally:
            # Explicit cleanup after prediction
            if x.device.type != 'cpu':
                x = x.cpu()
                del x
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return results['class_preds'], results['class_mapping']
    
    def predict_proba(
        self,
        data: Union[pd.DataFrame, np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """
        Get class probabilities using the base model.
        Memory-efficient implementation using mixed precision.
        
        Parameters:
        -----------
        data : Union[pd.DataFrame, np.ndarray, torch.Tensor]
            Input data to classify
            
        Returns:
        --------
        torch.Tensor
            Class probabilities on CPU
        """
        # Preprocess data
        x = self.preprocess_data(data)
        
        # Get base model predictions with mixed precision if enabled
        try:
            with torch.no_grad():
                if self.use_mixed_precision and torch.cuda.is_available() and x.device.type == 'cuda':
                    # CUDA with mixed precision
                    with torch.cuda.amp.autocast():
                        outputs = self.model(x)
                else:
                    # Standard precision for MPS/CPU or when mixed precision is disabled
                    outputs = self.model(x)
                
                # Get class logits and compute probabilities
                class_logits = outputs['class_logits']
                
                # Do softmax on CPU to save memory
                class_logits_cpu = class_logits.cpu()
                probs = torch.softmax(class_logits_cpu, dim=-1)
                
                # Clean up
                del outputs
                del class_logits
                del class_logits_cpu
        finally:
            # Explicit cleanup
            if x.device.type != 'cpu':
                x = x.cpu()
                del x
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return probs


# Example usage
def example_usage():
    """Example of how to use the TextualClassifier interface."""
    import torch
    from ticl.model_builder import get_model
    
    # Load a pretrained model
    config = {
        'model_type': 'tabpfn',
        'semantic_prediction': True,
        'num_semantic_classes': 3,
        'prior': {
            'classification': {
                'semantic_feature_p': 0.3,
                'max_num_classes': 2
            },
            'num_features': 100
        }
    }
    
    # This would be your actual trained model
    _, model, _, _ = get_model(config, device='cpu', should_train=False)
    
    # Get semantic data
    from ticl.datasets.semantic_prior_data_loader import get_random_semantic_data
    semantic_data = get_random_semantic_data()
    
    # Create TextualClassifier
    classifier = TextualClassifier(model, semantic_data)
    
    # Sample data
    data = torch.rand(10, 150)  # 10 samples, 150 features (including 50 semantic)
    
    # Classify with single text description
    preds, similarity = classifier.classify_with_text(
        data, 
        "High income customers with good credit score"
    )
    print(f"Predictions: {preds}")
    print(f"Similarity: {similarity}")
    
    # Classify with multiple class descriptions
    class_descriptions = {
        "high_income": "Customers with income over $100K per year",
        "medium_income": "Customers with income between $50K and $100K per year",
        "low_income": "Customers with income below $50K per year"
    }
    
    preds, class_mapping = classifier.classify_with_descriptions(data, class_descriptions)
    print(f"Multi-class predictions: {preds}")
    print(f"Class mapping: {class_mapping}")


if __name__ == "__main__":
    example_usage()