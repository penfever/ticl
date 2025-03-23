import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from ticl.utils import log_gpu_memory, log_tensor_info, memory_logger, track_tensors_memory

class SemanticAwareClassifier(nn.Module):
    """
    Extension of base TabPFN or similar models with a CLIP-style semantic head.
    
    Rather than projecting numeric features to CLIP space, this model:
    1. Takes dedicated semantic token features (which are actual text tokens)
    2. Processes them directly with CLIP's text encoder
    3. Uses contrastive learning to align semantic features with class distributions
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
        
        # Embed sizes
        self.emsize = base_model.emsize  # For base model 
        
        # Inherit attributes from base model for compatibility with train.py
        self.n_out = base_model.n_out
        
        # Initialize attributes needed for training
        self.learning_rates = []
        self.losses = []
        self.wallclock_times = []
        self.start_time = 0
        
        # Import CLIP-related modules
        from transformers import CLIPTokenizerFast, CLIPTextModel
        
        # Store semantic class count
        self.num_semantic_classes = num_semantic_classes
        self.num_tokens_per_class = 5  # Top-k tokens per semantic class for interpretability
        
        # Initialize the CLIP tokenizer and text encoder
        self.model_name = "openai/clip-vit-base-patch32"
        self.tokenizer = CLIPTokenizerFast.from_pretrained(self.model_name)
        
        # Load pretrained CLIP text model
        self.clip_text_model = CLIPTextModel.from_pretrained(self.model_name)
        
        # Freeze CLIP text encoder parameters by default
        for param in self.clip_text_model.parameters():
            param.requires_grad = False
        
        # Get transformer dimensions
        self.transformer_dim = self.clip_text_model.config.hidden_size  # Usually 512 for base model
        
        # Temperature parameter for similarity scaling (learnable but bounded)
        initial_temp = np.log(1 / 0.1)
        self.logit_scale = nn.Parameter(torch.ones([]) * initial_temp)
        
        # Token statistics for interpretation
        self.token_stats = {}
    
    def _process_semantic_tokens(self, semantic_tokens):
        """
        Process semantic tokens using the CLIP text encoder.
        
        Parameters:
        -----------
        semantic_tokens : torch.Tensor
            Semantic token IDs [batch_size, seq_len]
            
        Returns:
        --------
        torch.Tensor
            CLIP text embeddings [batch_size, transformer_dim]
        """
        # Ensure tokens are on the right device
        device = semantic_tokens.device
        
        # Format tokens for CLIP
        attention_mask = (semantic_tokens != -100).long()
        input_ids = torch.where(semantic_tokens == -100, 
                                torch.tensor(self.tokenizer.pad_token_id, device=device), 
                                semantic_tokens)
        
        # Create input dict for CLIP
        token_dict = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        
        # Process with CLIP text encoder
        with torch.set_grad_enabled(self.clip_text_model.parameters()[0].requires_grad):
            outputs = self.clip_text_model(**token_dict)
        
        # Return pooled embeddings
        return outputs.pooler_output
    
    def forward(self, x, single_eval_pos=None, class_texts=None):
        """
        Forward pass that directly uses CLIP to encode semantic features.
        
        Parameters:
        -----------
        x : torch.Tensor or tuple
            Input tensor with shape [samples, batch, features] or
            tuple of (x_data, semantic_tokens) with semantic_tokens of shape [batch, seq_len]
        single_eval_pos : int, optional
            Position to split training and evaluation data
        class_texts : list of str, optional
            Text descriptions for semantic classes
            
        Returns:
        --------
        dict
            Dictionary containing class_logits and semantic_logits
        """
        # Extract semantic tokens if provided in tuple format
        semantic_tokens = None
        if isinstance(x, tuple) and len(x) >= 2:
            if len(x) == 3:
                # Format with style: (style, x_data, semantic_tokens)
                style, x_data, semantic_tokens = x
                # Pass style and data to base model
                base_input = (style, x_data)
            else:
                # Format: (x_data, semantic_tokens)
                x_data, semantic_tokens = x
                base_input = x_data
        else:
            # Standard format
            x_data = x
            base_input = x
        
        # Forward pass on the base model
        base_output = self.base_model(base_input, single_eval_pos=single_eval_pos)
        
        # Update semantic classes count if needed
        if isinstance(base_output, torch.Tensor) and len(base_output.shape) >= 3:
            num_output_classes = base_output.shape[-1]
            if num_output_classes != self.num_semantic_classes:
                self.num_semantic_classes = num_output_classes
        
        # Process semantic tokens directly with CLIP if provided
        semantic_embeddings = None
        if semantic_tokens is not None:
            semantic_embeddings = self._process_semantic_tokens(semantic_tokens)
        
        # Process class texts with CLIP if provided
        text_features = None
        all_token_texts = []
        
        # Process provided class texts with CLIP
        if class_texts and len(class_texts) > 0:
            with torch.no_grad():
                # Process text descriptions in batches to manage memory
                batch_size = min(16, len(class_texts))
                text_features_list = []
                
                for i in range(0, len(class_texts), batch_size):
                    batch_texts = class_texts[i:i+batch_size]
                    
                    # Tokenize texts
                    text_tokens = self.tokenizer(
                        batch_texts,
                        padding="max_length",
                        truncation=True,
                        max_length=77,
                        return_tensors="pt"
                    ).to(semantic_tokens.device if semantic_tokens is not None else base_output.device)
                    
                    # Get text embeddings
                    batch_outputs = self.clip_text_model(**text_tokens)
                    batch_text_features = batch_outputs.pooler_output
                    text_features_list.append(batch_text_features)
                    
                    # Store tokenization info for interpretability
                    for j, text in enumerate(batch_texts):
                        token_ids = text_tokens['input_ids'][j].tolist()
                        token_texts = self.tokenizer.convert_ids_to_tokens(token_ids)
                        token_texts = [t for t in token_texts if t not in 
                                     ['<pad>', '<|startoftext|>', '<|endoftext|>']]
                        
                        all_token_texts.append({
                            'text': text,
                            'tokens': token_texts[:self.num_tokens_per_class],
                            'class_idx': i + j
                        })
                
                # Concatenate all text features
                if text_features_list:
                    text_features = torch.cat(text_features_list, dim=0)
                    
                    # Ensure text features are on the same device
                    device = semantic_embeddings.device if semantic_embeddings is not None else base_output.device
                    text_features = text_features.to(device)
                    
                    # Normalize for similarity computation
                    text_features = F.normalize(text_features, dim=1)
        
        # Calculate semantic similarity if we have both semantic embeddings and text features
        semantic_logits = None
        if semantic_embeddings is not None and text_features is not None:
            # Normalize semantic embeddings
            semantic_embeddings_norm = F.normalize(semantic_embeddings, dim=1)
            
            # Calculate similarity with temperature scaling
            scaled_logit = self.logit_scale.exp()
            semantic_logits = scaled_logit * torch.matmul(semantic_embeddings_norm, text_features.t())
        
        # Create return dictionary
        result = {
            'class_logits': base_output,
            'semantic_logits': semantic_logits,
            'semantic_embeddings': semantic_embeddings,
            'text_features': text_features,
            'token_texts': all_token_texts
        }
        
        return result
    
    def predict(self, x):
        """
        Make predictions using the trained model.
        
        Parameters:
        -----------
        x : torch.Tensor or tuple
            Input tensor or tuple of (x_data, semantic_tokens)
            
        Returns:
        --------
        dict
            Dictionary containing class predictions and semantic predictions
        """
        # Get model outputs
        outputs = self.forward(x)
        
        # Get class predictions
        class_preds = outputs['class_logits'].argmax(dim=-1).cpu()
        
        # Get semantic predictions if available
        semantic_preds = None
        if outputs['semantic_logits'] is not None:
            semantic_preds = outputs['semantic_logits'].argmax(dim=-1).cpu()
        
        # Get token texts for interpretability
        token_texts = outputs['token_texts']
        
        # Return predictions
        return {
            'class_preds': class_preds,
            'semantic_preds': semantic_preds,
            'token_texts': token_texts
        }
    
    def predict_from_text(self, x, text_description, semantic_data=None, text_mapper=None):
        """
        Make predictions using text description to map to classes.
        
        Parameters:
        -----------
        x : torch.Tensor or tuple
            Input tensor or tuple of (x_data, semantic_tokens)
        text_description : str
            Text description of the class
        semantic_data : torch.Tensor, optional
            Not used in this implementation as we directly use CLIP text encoder
        text_mapper : object, optional
            Not used in this implementation
            
        Returns:
        --------
        dict
            Dictionary containing class predictions based on text mapping
        """
        # Process input data
        outputs = self.forward(x)
        
        # Process query text with CLIP
        with torch.no_grad():
            # Tokenize the input text
            text_tokens = self.tokenizer(
                text_description,
                padding="max_length",
                truncation=True,
                max_length=77,
                return_tensors="pt"
            ).to(outputs['semantic_embeddings'].device if outputs['semantic_embeddings'] is not None else 'cpu')
            
            # Get text embeddings
            query_features = self.clip_text_model(**text_tokens).pooler_output
            
            # Normalize for similarity computation
            query_features = F.normalize(query_features, dim=1)
        
        # Calculate similarity with semantic features if available
        best_class = torch.tensor(-1)
        max_similarity = torch.tensor(0.0)
        
        if outputs['semantic_embeddings'] is not None:
            # Normalize semantic embeddings
            semantic_norm = F.normalize(outputs['semantic_embeddings'], dim=1)
            
            # Calculate similarity
            similarities = torch.matmul(query_features, semantic_norm.t())
            
            # Find best match
            max_similarity, best_idx = similarities.max(dim=1)
            best_class = best_idx.cpu()
            max_similarity = max_similarity.cpu()
        
        # Get base model predictions
        class_preds = outputs['class_logits'].argmax(dim=-1).cpu()
        
        # Extract tokenization info
        token_texts = []
        if len(outputs['token_texts']) > 0:
            token_texts = outputs['token_texts']
        
        # Get query tokens for interpretability
        query_token_ids = text_tokens['input_ids'][0].tolist()
        query_token_texts = self.tokenizer.convert_ids_to_tokens(query_token_ids)
        query_token_texts = [t for t in query_token_texts if t not in 
                           ['<pad>', '<|startoftext|>', '<|endoftext|>']]
        
        # Return results
        return {
            'class_preds': class_preds,
            'mapped_class': best_class.item(),
            'similarity': max_similarity.item(),
            'query_tokens': query_token_texts[:self.num_tokens_per_class],
            'token_texts': token_texts
        }


class SemanticConsistencyLoss(nn.Module):
    """
    CLIP-style contrastive loss for aligning semantic tokens with class distributions.
    This implements the InfoNCE/NT-Xent contrastive loss from the CLIP paper.
    """
    
    def __init__(self, semantic_weight=0.2):
        """
        Initialize the loss function.
        
        Parameters:
        -----------
        semantic_weight : float
            Weight for the semantic contrastive loss component
        """
        super().__init__()
        self.semantic_weight = semantic_weight
        
        # Main classification loss
        self.class_loss = nn.CrossEntropyLoss()
        
        # Component loss values for logging
        self.last_class_loss = 0.0
        self.last_semantic_loss = 0.0
    
    def forward(self, outputs, targets):
        """
        Compute the combined loss with CLIP-style contrastive learning.
        
        Parameters:
        -----------
        outputs : dict
            Model outputs containing 'class_logits', 'semantic_logits',
            'semantic_embeddings', and 'text_features'
        targets : dict
            Target values containing 'class_targets' and 'semantic_targets'
            
        Returns:
        --------
        torch.Tensor
            Combined loss value
        """
        # Get class prediction outputs and targets
        class_logits = outputs['class_logits'].permute(1, 2, 0)
        class_targets = targets['class_targets'].permute(1, 0)
        
        # Calculate standard classification loss
        class_loss = self.class_loss(class_logits, class_targets)
        
        # Initialize semantic loss
        semantic_loss = torch.tensor(0.0, device=class_loss.device, requires_grad=True)
        
        # Calculate semantic contrastive loss if we have necessary components
        if (outputs['semantic_logits'] is not None and 
            'semantic_targets' in targets and 
            targets['semantic_targets'] is not None):
            
            # Get logits and targets
            semantic_logits = outputs['semantic_logits']
            semantic_targets = targets['semantic_targets']
            
            # Check for valid semantic targets (not -100)
            valid_mask = semantic_targets != -100
            valid_count = valid_mask.sum().item()
            
            if valid_count > 0:
                # Transpose targets for batch processing
                targets_transposed = semantic_targets.permute(1, 0)
                batch_size = targets_transposed.shape[0]
                
                # Initialize loss accumulation
                total_loss = 0
                total_rows = 0
                
                # Process each batch item
                for i in range(batch_size):
                    # Get logits and targets for this batch item
                    batch_logits = semantic_logits[i]
                    batch_targets = targets_transposed[i]
                    
                    # Filter out ignore indices (-100)
                    valid_indices = batch_targets != -100
                    valid_target_count = valid_indices.sum().item()
                    
                    if valid_target_count > 0:
                        # Get valid targets
                        valid_targets = batch_targets[valid_indices]
                        
                        # Ensure targets are in valid range
                        num_classes = batch_logits.size(0)
                        if valid_targets.max() >= num_classes:
                            valid_targets = valid_targets.clamp(max=num_classes-1)
                        
                        # Compute cross-entropy loss
                        batch_loss = F.cross_entropy(
                            batch_logits.unsqueeze(0).expand(valid_target_count, -1),
                            valid_targets
                        )
                        
                        # Add to total loss
                        total_loss += batch_loss * valid_target_count
                        total_rows += valid_target_count
                
                # Normalize by total valid targets
                if total_rows > 0:
                    semantic_loss = total_loss / total_rows
        
        # Store component losses for logging
        self.last_class_loss = class_loss.item()
        self.last_semantic_loss = semantic_loss.item()
        
        # Combine losses with weighting
        total_loss = class_loss + (self.semantic_weight * semantic_loss)
        
        return total_loss


def create_semantic_aware_model(base_model, num_semantic_classes=None, freeze_clip=True):
    """
    Factory function to create a semantic-aware model.
    
    Parameters:
    -----------
    base_model : nn.Module
        The base model to extend
    num_semantic_classes : int, optional
        Number of semantic classes to predict
    freeze_clip : bool
        Whether to freeze the CLIP text encoder parameters
        
    Returns:
    --------
    SemanticAwareClassifier
        The semantic-aware model
    """
    # Determine number of semantic classes if not provided
    if num_semantic_classes is None:
        try:
            # Try to load from semantic data
            from ticl.datasets.semantic_prior_data_loader import load_semantic_prior_data
            column_names, _ = load_semantic_prior_data()
            num_semantic_classes = len(column_names)
            num_semantic_classes = max(num_semantic_classes, 3)  # Ensure at least 3 classes
        except:
            # Default fallback
            num_semantic_classes = 3
    
    # Create model
    model = SemanticAwareClassifier(base_model, num_semantic_classes)
    
    # Configure CLIP encoder freezing
    if freeze_clip:
        print("Freezing CLIP text encoder parameters")
        for param in model.clip_text_model.parameters():
            param.requires_grad = False
    else:
        print("Fine-tuning CLIP text encoder (this will increase GPU memory usage)")
        for param in model.clip_text_model.parameters():
            param.requires_grad = True
    
    # Return configured model
    return model