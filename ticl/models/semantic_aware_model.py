import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticAwareClassifier(nn.Module):
    """
    Extension of base TabPFN or similar models with an additional semantic prediction head.
    This model supports self-supervised learning of semantic class relationships.
    """
    
    def __init__(self, base_model, num_semantic_classes):
        """
        Initialize a semantic-aware classifier that extends a base model.
        
        Parameters:
        -----------
        base_model : nn.Module
            The base tabular model (TabPFN, MotherNet, etc.)
        num_semantic_classes : int
            Number of semantic classes to predict
        """
        super().__init__()
        
        # Store the base model
        self.base_model = base_model
        
        # Embedding size from the base model
        self.emsize = base_model.emsize
        
        # Add a semantic prediction head
        self.semantic_head = nn.Linear(self.emsize, num_semantic_classes)
        
    def forward(self, x):
        """
        Forward pass with both class and semantic predictions.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor with shape [samples, batch, features]
            
        Returns:
        --------
        dict
            Dictionary containing class_logits and semantic_logits
        """
        # Get encoder features from the base model
        features = self.base_model.encoder(x)
        
        # Get class predictions from the base model's decoder
        class_logits = self.base_model.decoder(features)
        
        # Get semantic class predictions from our additional head
        semantic_logits = self.semantic_head(features)
        
        return {
            'class_logits': class_logits,
            'semantic_logits': semantic_logits
        }
    
    def predict(self, x):
        """
        Make predictions using the trained model.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor
            
        Returns:
        --------
        dict
            Dictionary containing class predictions and semantic predictions
        """
        outputs = self.forward(x)
        
        class_preds = outputs['class_logits'].argmax(dim=-1)
        semantic_preds = outputs['semantic_logits'].argmax(dim=-1)
        
        return {
            'class_preds': class_preds,
            'semantic_preds': semantic_preds
        }
    
    def predict_from_text(self, x, text_description, semantic_data, text_mapper=None):
        """
        Make predictions using text description to map to classes.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor
        text_description : str
            Text description of the class
        semantic_data : torch.Tensor
            Semantic token data used for training
        text_mapper : SemanticTextMapper, optional
            Text mapper to use (created if None)
            
        Returns:
        --------
        dict
            Dictionary containing class predictions based on text mapping
        """
        # Import the text mapper only when needed
        from ticl.semantic_text_mapper import SemanticTextMapper
        
        # Create text mapper if not provided
        if text_mapper is None:
            text_mapper = SemanticTextMapper(device=x.device)
        
        # Get model's outputs
        outputs = self.forward(x)
        semantic_preds = outputs['semantic_logits']
        
        # Convert semantic predictions to class info
        semantic_classes = semantic_preds.argmax(dim=-1)
        
        # Create a mapping from semantic classes to the model's internal class tokens
        class_mapping = {}
        for c in range(semantic_classes.max().item() + 1):
            # Find samples classified as this semantic class
            mask = semantic_classes == c
            if mask.any():
                # Use most common predicted class for this semantic class
                class_mapping[c] = {
                    'semantic_class': c,
                    'token_indices': torch.arange(10).tolist(),  # Simplification
                    'tokens': torch.ones(10)  # Placeholder
                }
        
        # Map text to closest class
        best_class, similarity = text_mapper.map_text_to_class(
            text_description, 
            class_mapping, 
            semantic_data
        )
        
        # Get the class predictions
        if best_class >= 0:
            # Find samples with the matched semantic class
            mask = semantic_classes == best_class
            class_preds = torch.full_like(semantic_classes, -1)
            class_preds[mask] = outputs['class_logits'][mask].argmax(dim=-1)
        else:
            # Fallback to regular class predictions if no match
            class_preds = outputs['class_logits'].argmax(dim=-1)
        
        return {
            'class_preds': class_preds,
            'mapped_class': best_class,
            'similarity': similarity
        }
    
    def generate_boundaries_from_text(self, x, class_descriptions, semantic_data, text_mapper=None):
        """
        Generate new class boundaries from text descriptions.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor
        class_descriptions : Dict[str, str]
            Dictionary mapping class names to text descriptions
        semantic_data : torch.Tensor
            Semantic token data used for training
        text_mapper : SemanticTextMapper, optional
            Text mapper to use (created if None)
            
        Returns:
        --------
        dict
            Dictionary containing new class predictions based on text-generated boundaries
        """
        # Import the text mapper only when needed
        from ticl.semantic_text_mapper import SemanticTextMapper
        
        # Create text mapper if not provided
        if text_mapper is None:
            text_mapper = SemanticTextMapper(device=x.device)
        
        # Generate class boundaries from text descriptions
        class_token_patterns = text_mapper.generate_class_boundaries(
            class_descriptions,
            semantic_data
        )
        
        # Apply these boundaries to classify the input
        predictions = text_mapper.apply_text_boundaries(x, class_token_patterns)
        
        return {
            'class_preds': predictions,
            'class_mapping': {i: name for i, name in enumerate(class_descriptions.keys())}
        }


class SemanticConsistencyLoss(nn.Module):
    """
    Combined loss function for classification and semantic prediction.
    """
    
    def __init__(self, semantic_weight=0.5):
        """
        Initialize the loss function.
        
        Parameters:
        -----------
        semantic_weight : float
            Weight for the semantic prediction loss
        """
        super().__init__()
        self.semantic_weight = semantic_weight
        self.class_loss = nn.CrossEntropyLoss()
        self.semantic_loss = nn.CrossEntropyLoss(ignore_index=-100)
        
    def forward(self, outputs, targets):
        """
        Compute the combined loss.
        
        Parameters:
        -----------
        outputs : dict
            Model outputs containing 'class_logits' and 'semantic_logits'
        targets : dict
            Target values containing 'class_targets' and 'semantic_targets'
            
        Returns:
        --------
        torch.Tensor
            Combined loss value
        """
        # Get the outputs and targets
        class_logits = outputs['class_logits']
        semantic_logits = outputs['semantic_logits']
        class_targets = targets['class_targets']
        semantic_targets = targets['semantic_targets']
        
        # Compute the class loss
        class_loss = self.class_loss(class_logits, class_targets)
        
        # Compute the semantic loss (only if we have semantic targets)
        if semantic_targets is not None:
            # Reshape semantic_logits from [samples, batch, num_semantic_classes] to [samples*batch, num_semantic_classes]
            batch_size = semantic_logits.size(1)
            semantic_logits_flat = semantic_logits.reshape(-1, semantic_logits.size(-1))
            semantic_targets_flat = semantic_targets.reshape(-1)
            
            semantic_loss = self.semantic_loss(semantic_logits_flat, semantic_targets_flat)
            # Combined loss
            return class_loss + self.semantic_weight * semantic_loss
        else:
            # Just class loss if no semantic targets
            return class_loss


def create_semantic_aware_model(base_model, num_semantic_classes=3):
    """
    Factory function to create a semantic-aware model.
    
    Parameters:
    -----------
    base_model : nn.Module
        The base model to extend
    num_semantic_classes : int
        Number of semantic classes to predict
        
    Returns:
    --------
    SemanticAwareClassifier
        The extended model
    """
    return SemanticAwareClassifier(base_model, num_semantic_classes)