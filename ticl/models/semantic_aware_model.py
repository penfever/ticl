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
            Semantic token IDs [batch_size, seq_len] or [seq_len]
            
        Returns:
        --------
        torch.Tensor
            CLIP text embeddings [batch_size, transformer_dim]
        """
        # Ensure tokens are on the right device
        device = semantic_tokens.device
        
        # Make sure we have a batch dimension
        if semantic_tokens.dim() == 1:
            semantic_tokens = semantic_tokens.unsqueeze(0)  # Add batch dimension
        
        # CLIP has a maximum context length of 77 tokens, so we need to truncate if longer
        max_length = 77
        if semantic_tokens.shape[-1] > max_length:
            semantic_tokens = semantic_tokens[..., :max_length]
            
        # Format tokens for CLIP
        attention_mask = (semantic_tokens != -100).long()
        input_ids = torch.where(semantic_tokens == -100, 
                                torch.tensor(self.tokenizer.pad_token_id, device=device), 
                                semantic_tokens)
        
        # Check that attention_mask has the right shape (batch_size, seq_len)
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
            
        # Ensure input_ids have the right shape (batch_size, seq_len)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
                    
        # Sanity check for sequence length
        if input_ids.shape[-1] > max_length:
            memory_logger.warning(f"Input IDs still too long ({input_ids.shape[-1]}), truncating to {max_length}")
            input_ids = input_ids[..., :max_length]
            attention_mask = attention_mask[..., :max_length]
            
        # Create input dict for CLIP
        token_dict = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        
        # Process with CLIP text encoder
        # Check if any parameters require gradients
        requires_grad = any(p.requires_grad for p in self.clip_text_model.parameters())
        
        # Use that to determine whether to track gradients
        with torch.set_grad_enabled(requires_grad):
            outputs = self.clip_text_model(**token_dict)
        
        # Return pooled embeddings
        return outputs.pooler_output
    
    def forward(self, x, single_eval_pos=None, class_texts=None, batch_info=None):
        """
        Forward pass that directly uses CLIP to encode semantic features.
        
        Parameters:
        -----------
        x : torch.Tensor or tuple
            Input tensor with shape [samples, batch, features] or
            tuple of (x_data, y_data) or (style, x_data, y_data) following TabPFN convention
        single_eval_pos : int, optional
            Position to split training and evaluation data
        class_texts : list of str, optional
            Text descriptions for semantic classes
        batch_info : dict, optional
            Dictionary with additional batch information, including semantic targets
            
        Returns:
        --------
        dict
            Dictionary containing class_logits and semantic_logits
        """
        # Pass data to base model first to get base predictions
        base_output = self.base_model(x, single_eval_pos=single_eval_pos)
        
        # Extract semantic features and tokens
        semantic_tokens = None
        semantic_targets = None
        class_token_patterns = None
        
        # Try to extract semantic information from batch_info
        if batch_info is not None:
            # Get semantic targets if available
            if 'semantic_targets' in batch_info:
                semantic_targets = batch_info['semantic_targets']
                
            # Get class token patterns if available
            if 'class_token_patterns' in batch_info:
                class_token_patterns = batch_info['class_token_patterns']
                
            # Get direct semantic tokens if available (fallback)
            if 'semantic_tokens' in batch_info:
                semantic_tokens = batch_info['semantic_tokens']
        
        # As a backup, try to extract semantic info from the input tuple structure
        if (semantic_tokens is None or class_token_patterns is None) and isinstance(x, tuple):
            info_dict = None
            
            # Extract info dictionary from the input structure
            if len(x) == 3:  # (style, x_src, y_src)
                style_src, x_src, y_src = x
                if isinstance(x_src, tuple) and len(x_src) >= 2 and isinstance(x_src[0], dict):
                    info_dict = x_src[0]
            elif len(x) == 2:  # (x_src, y_src)
                x_src, y_src = x
                if isinstance(x_src, tuple) and len(x_src) >= 2 and isinstance(x_src[0], dict):
                    info_dict = x_src[0]
            
            # Extract semantic information from the info dictionary
            if info_dict is not None:
                if 'semantic_tokens' in info_dict and semantic_tokens is None:
                    semantic_tokens = info_dict['semantic_tokens']
                    
                if 'class_token_patterns' in info_dict and class_token_patterns is None:
                    class_token_patterns = info_dict['class_token_patterns']
                    
                if 'semantic_targets' in info_dict and semantic_targets is None:
                    semantic_targets = info_dict['semantic_targets']
        
        # Extract features from the base model for record keeping, but we don't use these
        extracted_features = None
        if hasattr(self.base_model, 'features') and self.base_model.features is not None:
            extracted_features = self.base_model.features
        
        # Update semantic classes count if needed
        if isinstance(base_output, torch.Tensor) and len(base_output.shape) >= 3:
            num_output_classes = base_output.shape[-1]
            if num_output_classes != self.num_semantic_classes:
                self.num_semantic_classes = num_output_classes
        
        # ===== STEP 1: Embed semantic tokens for each class token pattern =====
        pattern_embeddings = {}
        column_embeddings = {}
        device = base_output.device
        
        # Process if we have class token patterns available
        if class_token_patterns is not None and len(class_token_patterns) > 0:
            
            # For each class token pattern, get CLIP embeddings of:
            # 1. The token sequence itself
            # 2. The column name (if available)
            for class_idx, pattern in class_token_patterns.items():
                # 1. Process tokens sequence
                tokens = pattern['tokens']
                token_embedding = self._process_semantic_tokens(tokens)
                pattern_embeddings[class_idx] = token_embedding
                
                # 2. Process column name if available
                if 'column_name' in pattern and pattern['column_name'] is not None:
                    column_name = pattern['column_name']
                    with torch.no_grad():
                        column_tokens = self.tokenizer(
                            column_name,
                            padding="max_length",
                            truncation=True,
                            max_length=77,
                            return_tensors="pt"
                        ).to(device)
                        
                        column_output = self.clip_text_model(**column_tokens)
                        column_embedding = column_output.pooler_output
                        column_embeddings[class_idx] = column_embedding
        elif semantic_tokens is not None:
            # Fallback: use direct semantic tokens if available
            fallback_embedding = self._process_semantic_tokens(semantic_tokens)
            pattern_embeddings[0] = fallback_embedding
        
        # Check if we managed to get any embeddings
        if not pattern_embeddings:
            raise ValueError("No semantic tokens found in input data. This model requires semantic tokens for training.")
                
        # ===== STEP 2: Process semantic class texts to get class embeddings =====
        class_embeddings = {}
        
        # If class_texts are provided, use them directly
        if class_texts and len(class_texts) > 0:
            with torch.no_grad():
                for i, text in enumerate(class_texts):
                    text_tokens = self.tokenizer(
                        text,
                        padding="max_length",
                        truncation=True,
                        max_length=77,
                        return_tensors="pt"
                    ).to(device)
                    
                    text_output = self.clip_text_model(**text_tokens)
                    class_embeddings[i] = text_output.pooler_output
        # Otherwise, try to use class_name from token patterns
        elif class_token_patterns is not None:
            with torch.no_grad():
                for class_idx, pattern in class_token_patterns.items():
                    if 'class_name' in pattern and pattern['class_name'] is not None:
                        text = pattern['class_name']
                        semantic_class = pattern['semantic_class']
                        
                        text_tokens = self.tokenizer(
                            text,
                            padding="max_length",
                            truncation=True,
                            max_length=77,
                            return_tensors="pt"
                        ).to(device)
                        
                        text_output = self.clip_text_model(**text_tokens)
                        class_embeddings[semantic_class] = text_output.pooler_output
        
        # Make sure we have some class embeddings
        if not class_embeddings:
            memory_logger.debug("No class texts found. Creating generic class embeddings.")
            with torch.no_grad():
                # Generate some synthetic class names
                num_classes = self.num_semantic_classes
                for i in range(num_classes):
                    text = f"Class {i}"
                    text_tokens = self.tokenizer(
                        text,
                        padding="max_length",
                        truncation=True,
                        max_length=77,
                        return_tensors="pt"
                    ).to(device)
                    
                    text_output = self.clip_text_model(**text_tokens)
                    class_embeddings[i] = text_output.pooler_output
        
        # ===== STEP 3: Compute similarity between embeddings and class descriptions =====
        # Get batch size for proper tensor shapes
        batch_size = 1
        if isinstance(base_output, torch.Tensor) and base_output.dim() >= 3:
            batch_size = base_output.shape[1]
                
        # For tracking tokenization info
        all_token_texts = []
        
        # Prepare embeddings for similarity calculation
        # 1. Combine pattern and column embeddings (if available)
        combined_pattern_embeddings = []
        pattern_to_class_mapping = []  # Maps pattern index to original class index
        
        for pattern_idx in sorted(pattern_embeddings.keys()):
            pattern_emb = pattern_embeddings[pattern_idx]
            
            # If we have column embedding for this pattern, average with pattern embedding
            if pattern_idx in column_embeddings:
                col_emb = column_embeddings[pattern_idx]
                # Average the embeddings (could also concatenate or use other combination)
                combined_emb = (pattern_emb + col_emb) / 2.0
            else:
                combined_emb = pattern_emb
                
            combined_pattern_embeddings.append(combined_emb)
            pattern_to_class_mapping.append(pattern_idx)
            
            # Add token info for interpretability
            if class_token_patterns is not None and pattern_idx in class_token_patterns:
                pattern = class_token_patterns[pattern_idx]
                class_name = pattern.get('class_name', f"Class_{pattern_idx}")
                column_name = pattern.get('column_name', None)
                
                description = class_name
                if column_name:
                    description += f" ({column_name})"
                    
                all_token_texts.append({
                    'text': description,
                    'class_idx': pattern_idx,
                    'semantic_class': pattern.get('semantic_class', pattern_idx)
                })
        
        # Stack embeddings if we have multiple
        if combined_pattern_embeddings:
            pattern_embeddings_tensor = torch.cat(combined_pattern_embeddings, dim=0)
            
            # Normalize for cosine similarity
            pattern_embeddings_norm = F.normalize(pattern_embeddings_tensor, dim=1)
        else:
            # Create dummy tensor as placeholder
            pattern_embeddings_norm = torch.zeros((1, self.transformer_dim), device=device)
        
        # 2. Stack and normalize class embeddings
        class_embeddings_list = []
        class_indices = []
        
        for class_idx in sorted(class_embeddings.keys()):
            class_emb = class_embeddings[class_idx]
            class_embeddings_list.append(class_emb)
            class_indices.append(class_idx)
            
        if class_embeddings_list:
            class_embeddings_tensor = torch.cat(class_embeddings_list, dim=0)
            
            # Normalize for cosine similarity
            class_embeddings_norm = F.normalize(class_embeddings_tensor, dim=1)
        else:
            # Create dummy tensor as placeholder
            class_embeddings_norm = torch.zeros((1, self.transformer_dim), device=device)
            
        # 3. Calculate temperature-scaled similarities
        semantic_logits = None
        scaled_logit = self.logit_scale.exp()
        
        # Calculate similarities:
        # - Each pattern embedding should have similarity with each class
        # This will give us a matrix [num_patterns, num_classes]
        raw_similarities = scaled_logit * torch.matmul(
            pattern_embeddings_norm, 
            class_embeddings_norm.t()
        )
        
        # Now reshape for loss calculation - we need [batch_size, num_embeddings, num_classes]
        # Different cases depending on number of embeddings vs batch size:
        num_patterns = len(pattern_to_class_mapping)
        
        if num_patterns == batch_size:
            # Ideal case - one embedding per batch item
            semantic_logits = raw_similarities.unsqueeze(1)  # [batch, 1, num_classes]
        elif num_patterns < batch_size:
            # Fewer patterns than batch size - repeat to match
            repeats = (batch_size + num_patterns - 1) // num_patterns  # Ceiling division
            repeated = raw_similarities.repeat(repeats, 1)[:batch_size]
            semantic_logits = repeated.unsqueeze(1)  # [batch, 1, num_classes]
        else:
            # More patterns than batch size - use the first batch_size
            semantic_logits = raw_similarities[:batch_size].unsqueeze(1)  # [batch, 1, num_classes]
                
        # Create return dictionary
        result = {
            'class_logits': base_output,
            'semantic_logits': semantic_logits,
            'semantic_embeddings': pattern_embeddings_tensor,
            'text_features': class_embeddings_tensor,
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
        
        # Skip semantic loss if no semantic components
        if ('semantic_logits' not in outputs or 
            outputs['semantic_logits'] is None or 
            'semantic_targets' not in targets):
            memory_logger.debug("Skipping semantic loss - missing components")
            return class_loss
            
        # Get logits and targets
        semantic_logits = outputs['semantic_logits']  # [batch_size, num_embeddings, num_classes]
        semantic_targets = targets['semantic_targets']  # [samples, batch_size]
        
        # Check for valid semantic targets (not -100)
        valid_mask = semantic_targets != -100
        valid_count = valid_mask.sum().item()
        
        if valid_count > 0:
            # To simplify computation, transpose targets to [batch_size, samples]
            targets_transposed = semantic_targets.permute(1, 0)
            batch_size = targets_transposed.shape[0]
            
            # Verify the batch_size matches with logits
            if batch_size != semantic_logits.shape[0]:
                memory_logger.warning(
                    f"Batch size mismatch: targets ({batch_size}) != logits ({semantic_logits.shape[0]})"
                )
                # Adjust to the smaller of the two
                batch_size = min(batch_size, semantic_logits.shape[0])
            
            # Initialize loss accumulation
            total_loss = 0
            total_valid_samples = 0
            
            # Process each batch item
            for batch_idx in range(batch_size):
                # Get logits for this batch item [num_embeddings, num_classes]
                # Usually num_embeddings=1, so we'll squeeze it later
                batch_logits = semantic_logits[batch_idx]
                
                # Get targets for this batch [samples]
                batch_targets = targets_transposed[batch_idx]
                
                # Filter out ignore indices (-100)
                valid_indices = batch_targets != -100
                valid_target_count = valid_indices.sum().item()
                
                if valid_target_count > 0:
                    # Get valid targets [valid_count]
                    valid_targets = batch_targets[valid_indices]
                    
                    # Ensure targets are in valid range by clamping to num_classes-1
                    num_classes = batch_logits.size(-1)
                    if valid_targets.max() >= num_classes:
                        valid_targets = valid_targets.clamp(max=num_classes-1)
                    
                    # Reshape logits to make them compatible with the cross-entropy loss
                    # If we have multiple embeddings, we'll need to handle each separately
                    num_embeddings = batch_logits.size(0)
                    
                    if num_embeddings == 1:
                        # If we have just one embedding (the common case),
                        # expand it to match valid_target_count
                        reshaped_logits = batch_logits.squeeze(0).unsqueeze(0).expand(valid_target_count, -1)
                        
                        # Compute cross-entropy loss for this embedding
                        batch_loss = F.cross_entropy(reshaped_logits, valid_targets)
                        
                    else:
                        # If we have multiple embeddings, compute loss for each and average
                        embedding_losses = []
                        
                        for emb_idx in range(num_embeddings):
                            emb_logits = batch_logits[emb_idx].unsqueeze(0).expand(valid_target_count, -1)
                            emb_loss = F.cross_entropy(emb_logits, valid_targets)
                            embedding_losses.append(emb_loss)
                        
                        # Average across embeddings
                        batch_loss = torch.stack(embedding_losses).mean()
                    
                    # Weighted contribution based on number of valid targets
                    total_loss += batch_loss * valid_target_count
                    total_valid_samples += valid_target_count
            
            # Normalize by total valid targets
            if total_valid_samples > 0:
                semantic_loss = total_loss / total_valid_samples
                memory_logger.debug(f"Semantic loss: {semantic_loss.item():.4f} from {total_valid_samples} valid samples")
            else:
                memory_logger.debug("No valid semantic targets found after filtering")
        else:
            memory_logger.debug("No valid semantic targets found in batch")
        
        # Store component losses for logging
        self.last_class_loss = class_loss.item()
        self.last_semantic_loss = semantic_loss.item()
        
        # Combine losses with weighting
        total_loss = class_loss + (self.semantic_weight * semantic_loss)
        memory_logger.debug(f"Total loss: {total_loss.item():.4f} = {class_loss.item():.4f} + " 
                           f"{self.semantic_weight} * {semantic_loss.item():.4f}")
        
        return total_loss


def get_semantic_class_count():
    """
    Get the current count of semantic classes from the semantic data loader.
    
    Returns:
    --------
    int
        Number of semantic classes available
    """
    try:
        from ticl.datasets.semantic_prior_data_loader import load_semantic_prior_data
        
        # Try to load real data to get column count
        column_names, _ = load_semantic_prior_data()
        num_classes = len(column_names)
        
        # Ensure we have at least 3 classes (minimum reasonable number)
        return max(num_classes, 3)
    except (ImportError, FileNotFoundError, Exception) as e:
        # Fallback to default if loading fails
        return 3


def create_semantic_aware_model(base_model, num_semantic_classes=None, freeze_clip=False):
    """
    Factory function to create a semantic-aware model.
    
    Parameters:
    -----------
    base_model : nn.Module
        The base model to extend
    num_semantic_classes : int, optional
        Number of semantic classes to predict
    freeze_clip : bool
        Whether to freeze the CLIP text encoder parameters (default: False)
        
    Returns:
    --------
    SemanticAwareClassifier
        The semantic-aware model
    """
    # Determine number of semantic classes if not provided
    if num_semantic_classes is None:
        num_semantic_classes = get_semantic_class_count()
    
    # Create model
    model = SemanticAwareClassifier(base_model, num_semantic_classes)
    
    # Configure CLIP encoder freezing
    if freeze_clip:
        print("Freezing CLIP text encoder parameters")
        for param in model.clip_text_model.parameters():
            param.requires_grad = False
    else:
        print("Fine-tuning CLIP text encoder (training the transformer)")
        for param in model.clip_text_model.parameters():
            param.requires_grad = True
    
    # Return configured model
    return model