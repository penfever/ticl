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
        device: Optional[str] = None
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
        """
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = device
            
        # Store model and semantic data (keep semantic data on CPU to save GPU memory)
        self.model = model.to(self.device)
        self.model.eval()
        self.semantic_data = semantic_data.to("cpu")  # Store on CPU, only move to GPU when needed
        
        # Initialize text mapper - also keep on CPU for text processing
        self.text_mapper = SemanticTextMapper(device="cpu")
        
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
        
        # Move semantic data to GPU only for this operation
        semantic_data_gpu = self.semantic_data.to(self.device)
        
        # Get predictions based on text
        try:
            with torch.no_grad():
                results = self.model.predict_from_text(
                    x, 
                    text_description, 
                    semantic_data_gpu,
                    self.text_mapper
                )
        finally:
            # Make sure we clean up GPU memory even if there's an error
            semantic_data_gpu = semantic_data_gpu.cpu()
            del semantic_data_gpu
            torch.cuda.empty_cache()
        
        return results['class_preds'], results['similarity']
    
    def classify_with_descriptions(
        self, 
        data: Union[pd.DataFrame, np.ndarray, torch.Tensor],
        class_descriptions: Dict[str, str]
    ) -> Tuple[torch.Tensor, Dict[int, str]]:
        """
        Classify data using multiple class descriptions.
        
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
        
        # Move semantic data to GPU only for this operation
        semantic_data_gpu = self.semantic_data.to(self.device)
        
        # Generate boundaries and classify
        try:
            with torch.no_grad():
                results = self.model.generate_boundaries_from_text(
                    x,
                    class_descriptions,
                    semantic_data_gpu,
                    self.text_mapper
                )
        finally:
            # Make sure we clean up GPU memory even if there's an error
            semantic_data_gpu = semantic_data_gpu.cpu()
            del semantic_data_gpu
            torch.cuda.empty_cache()
        
        return results['class_preds'], results['class_mapping']
    
    def predict_proba(
        self,
        data: Union[pd.DataFrame, np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """
        Get class probabilities using the base model.
        
        Parameters:
        -----------
        data : Union[pd.DataFrame, np.ndarray, torch.Tensor]
            Input data to classify
            
        Returns:
        --------
        torch.Tensor
            Class probabilities
        """
        # Preprocess data
        x = self.preprocess_data(data)
        
        # Get base model predictions
        with torch.no_grad():
            outputs = self.model(x)
            probs = torch.softmax(outputs['class_logits'], dim=-1)
        
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