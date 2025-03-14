import torch
import torch.nn as nn
import torch.nn.functional as F
from ticl.utils import log_gpu_memory, log_tensor_info, memory_logger, track_tensors_memory

class SemanticAwareClassifier(nn.Module):
    """
    Extension of base TabPFN or similar models with an additional semantic prediction head.
    This model uses a pretrained CLIP text encoder for handling semantic class relationships
    for more efficient text encoding and class prediction.
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
        
        # Inherit attributes from base model for compatibility with train.py
        self.n_out = base_model.n_out
        
        # Initialize attributes needed for training
        self.learning_rates = []
        self.losses = []
        self.wallclock_times = []
        self.start_time = 0
        
        # Import CLIP-related modules
        from transformers import CLIPTokenizerFast, CLIPTextModel
        
        # Constants for the semantic prediction head
        self.num_semantic_classes = num_semantic_classes
        self.clip_vocab_size = 49408  # CLIP tokenizer vocab size
        self.num_tokens_per_class = 5  # Top-k tokens per semantic class for interpretability
        
        # Initialize the CLIP tokenizer and text encoder
        self.model_name = "openai/clip-vit-base-patch32"
        self.tokenizer = CLIPTokenizerFast.from_pretrained(self.model_name)
        
        # Load pretrained CLIP text model (freeze its parameters by default)
        self.clip_text_model = CLIPTextModel.from_pretrained(self.model_name)
        
        # Freeze CLIP text encoder parameters for better stability
        for param in self.clip_text_model.parameters():
            param.requires_grad = False
            
        # Get transformer dimensions from the CLIP model
        transformer_dim = self.clip_text_model.config.hidden_size  # Usually 512 for base model
        
        # Project input features to transformer dimension
        self.semantic_projection = nn.Linear(self.emsize, transformer_dim)
        
        # Create a lookup table for semantic class embeddings
        self.semantic_class_embeddings = nn.Parameter(
            torch.randn(num_semantic_classes, transformer_dim),
            requires_grad=True
        )
        
        # Intermediate projection for feature enhancement before class matching
        self.feature_enhancer = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim * 2),
            nn.LayerNorm(transformer_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(transformer_dim * 2, transformer_dim)
        )
    
    def forward(self, x, single_eval_pos=None, class_texts=None):
        """
        Forward pass with both class and semantic predictions using a pretrained CLIP text encoder.
        This method processes tabular features and can compare them with text embeddings.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor with shape [samples, batch, features]
        single_eval_pos : int, optional
            Position to split training and evaluation data
        class_texts : list of str, optional
            Text descriptions for semantic classes (for embedding during training)
            
        Returns:
        --------
        dict
            Dictionary containing class_logits and semantic_logits
        """
        # Profiling: Log initial memory state
        log_gpu_memory("Start of SemanticAwareClassifier.forward")
        memory_logger.debug("Running CLIP-based semantic matching")
        
        # Pass the input through the base model's forward method 
        # which properly handles single_eval_pos
        base_output = self.base_model(x, single_eval_pos=single_eval_pos)
        
        # Extract the transformer's encoded features for semantic processing
        # The semantic head expects features with shape [batch_size, emsize]
        if hasattr(self.base_model, 'transformer_encoder') and hasattr(self.base_model, 'encoder'):
            # For TabPFN, we need to reconstruct the encoder features
            if len(x) == 3:  # style is given
                style_src, x_src, y_src = x
            else:
                x_src, y_src = x
            
            # Encode input features
            x_encoded = self.base_model.encoder(x_src)
            
            # Apply the transformer encoder
            # For semantic features, we'll use features from the test set
            if single_eval_pos is not None and single_eval_pos < x_encoded.shape[0]:
                # Use only evaluation samples for semantic features
                features = x_encoded[single_eval_pos:]
                # Average over samples to get a [batch_size, emsize] tensor
                features = features.mean(dim=0)
            else:
                # Fallback: use all samples and average them
                features = x_encoded.mean(dim=0)
        elif hasattr(self.base_model, 'features') and self.base_model.features is not None:
            # If the base model directly exposes features
            features = self.base_model.features
        else:
            # Last resort fallback - try to reshape the output
            if len(base_output.shape) == 3:
                # Average over samples to get [batch_size, output_dim]
                features = base_output.mean(dim=0)
            else:
                # Otherwise use as is
                features = base_output
        
        # Project features to CLIP text model dimension
        projected_features = self.semantic_projection(features)
        
        # Add batch dimension if needed (for single sample case)
        if len(projected_features.shape) == 1:
            projected_features = projected_features.unsqueeze(0)
        
        batch_size = projected_features.shape[0]
        
        # Apply feature enhancement to match CLIP embedding space
        enhanced_features = self.feature_enhancer(projected_features)
        
        # Average pooling over batch if needed
        if enhanced_features.dim() > 2:
            enhanced_features = enhanced_features.mean(dim=1)
        
        # ===== CLIP-like semantic similarity computation =====
        # Use the stored class embeddings to compute similarities
        semantic_similarities = torch.matmul(
            enhanced_features,                         # [batch_size, dim]
            self.semantic_class_embeddings.transpose(0, 1)  # [dim, num_classes]
        )
        
        # ===== Process class texts if provided (used during training) =====
        all_token_texts = []
        
        # If class texts are provided, use CLIP to compute their embeddings
        if class_texts and len(class_texts) > 0:
            memory_logger.debug(f"Processing {len(class_texts)} class texts with CLIP")
            
            # Create CLIP embeddings for class texts (execute on CPU to save GPU memory)
            with torch.no_grad():
                # Tokenize on CPU - process in small batches to conserve memory
                clip_device = "cpu"  # Always process text on CPU first
                text_features_list = []
                
                # Process in batches of up to 10 texts to control memory usage
                batch_size = min(10, len(class_texts))
                for i in range(0, len(class_texts), batch_size):
                    batch_texts = class_texts[i:i+batch_size]
                    
                    # Tokenize text batch with CLIP tokenizer
                    text_tokens = self.tokenizer(
                        batch_texts,
                        padding="max_length",
                        truncation=True,
                        max_length=77,
                        return_tensors="pt"
                    ).to(clip_device)
                    
                    # Extract text features with CLIP text encoder
                    batch_text_features = self.clip_text_model(**text_tokens).pooler_output
                    text_features_list.append(batch_text_features)
                
                # Concatenate all batches of features
                all_text_features = torch.cat(text_features_list, dim=0)
                
                # Create a simple mapping between texts and their CLIP embeddings
                for i, text in enumerate(class_texts):
                    # Store text-feature mapping
                    top_tokens = self.tokenizer.tokenize(text)[:self.num_tokens_per_class]
                    
                    # Store token information
                    class_tokens = {
                        'text': text,
                        'tokens': top_tokens,
                        'class_idx': i,
                    }
                    all_token_texts.append(class_tokens)
        
        # Create the return dictionary
        result = {
            'class_logits': base_output,
            'semantic_logits': semantic_similarities,  # [batch_size, num_classes]
            'token_texts': all_token_texts,  # For interpretability
        }
        
        # Clean up intermediate tensors
        del projected_features
        del enhanced_features
        
        # Memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        log_gpu_memory("End of SemanticAwareClassifier.forward")
        return result
    
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
            Dictionary containing class predictions, semantic predictions, and token sets
        """
        # Get model outputs and immediately move tensors to CPU to free GPU memory
        outputs = self.forward(x)
        
        # Get class predictions from class logits and move to CPU
        class_preds = outputs['class_logits'].argmax(dim=-1).cpu()
        
        # Get semantic class predictions and move to CPU
        semantic_preds = outputs['semantic_logits'].argmax(dim=-1).cpu()
        
        # Process token predictions for interpretability
        token_texts = outputs['token_texts']
        
        # Clean up GPU memory from forward pass
        if 'token_logits' in outputs:
            del outputs['token_logits']
        
        # Decode the predicted tokens using the CLIP tokenizer
        decoded_tokens = []
        for class_tokens in token_texts:
            # Map to full CLIP vocabulary
            class_idx = class_tokens['class_idx']
            indices = class_tokens['indices']
            scores = class_tokens['scores']
            
            # Decode indices to actual words (if needed)
            try:
                # Process on CPU to save GPU memory
                with torch.no_grad():
                    # This would map the predicted indices back to full CLIP vocab indices
                    # For simplicity, we're just using the indices as-is for now
                    token_words = [self.tokenizer.decode([idx]) for idx in indices]
                
                decoded = {
                    'class_idx': class_idx,
                    'tokens': token_words,
                    'scores': scores,
                }
                decoded_tokens.append(decoded)
            except Exception as e:
                # Fallback if decoding fails
                decoded_tokens.append({
                    'class_idx': class_idx,
                    'tokens': [f"token_{idx}" for idx in indices],
                    'scores': scores,
                })
        
        # Final cleanup of any remaining tensors
        del outputs
        
        # Explicit GPU memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Return predictions with interpretable token sets
        return {
            'class_preds': class_preds,
            'semantic_preds': semantic_preds,
            'token_sets': decoded_tokens
        }
    
    def predict_from_text(self, x, text_description, semantic_data=None, text_mapper=None):
        """
        Make predictions using text description to map to classes.
        This method uses the pretrained CLIP text encoder to process the description
        and find the most semantically similar class.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor
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
        # Get model's outputs with CLIP-based approach
        memory_logger.debug(f"Running predict_from_text with description: '{text_description}'")
        
        # Process tabular data with our model
        if torch.cuda.is_available() and x.device.type == 'cuda':
            with torch.cuda.amp.autocast(enabled=True):
                outputs = self.forward(x)
        else:
            outputs = self.forward(x)
        
        # Process text description with CLIP (on CPU to save GPU memory)
        with torch.no_grad():
            # Tokenize the input text with the CLIP tokenizer
            text_tokens = self.tokenizer(
                text_description,
                padding="max_length",
                truncation=True,
                max_length=77,
                return_tensors="pt"
            ).to("cpu")
            
            # Get text embeddings from CLIP text encoder
            text_features = self.clip_text_model(**text_tokens).pooler_output  # [1, hidden_size]
            
            # Extract the tokenized text for interpretability
            token_ids = text_tokens.input_ids[0].tolist()
            # Filter out special tokens and get the actual tokens
            token_texts = self.tokenizer.convert_ids_to_tokens(token_ids)
            # Remove padding, BOS, EOS tokens
            token_texts = [t for t in token_texts if t not in ['<pad>', '<|startoftext|>', '<|endoftext|>']]
            query_token_texts = token_texts[:self.num_tokens_per_class]  # Keep just the first few tokens
            
        # Move text features to same device as model outputs if needed
        if outputs.get('semantic_logits', None) is not None:
            device = outputs['semantic_logits'].device
            text_features = text_features.to(device)
            
            # Normalize feature vectors
            text_features = F.normalize(text_features, p=2, dim=1)
            
            # Use feature-enhanced dot product to compute similarity with semantic classes
            # Get the similarity between the text embedding and our semantic class embeddings
            text_class_similarities = torch.matmul(
                text_features,  # [1, hidden_size]
                self.semantic_class_embeddings.transpose(0, 1)  # [hidden_size, num_classes]
            )  # [1, num_classes]
            
            # Find best matching semantic class
            best_class_idx = text_class_similarities.argmax(dim=1)
            max_similarity = text_class_similarities.max(dim=1)[0]
            
            # Move to CPU for return
            best_class = best_class_idx.cpu()
            max_similarity = max_similarity.cpu()
        else:
            best_class = torch.tensor(-1)
            max_similarity = torch.tensor(0.0)
        
        # Get class predictions from the base model
        class_logits = outputs.get('class_logits', None)
        if class_logits is not None:
            class_logits_cpu = class_logits.cpu()
            class_preds = class_logits_cpu.argmax(dim=-1)
        else:
            class_preds = torch.full((1,), -1, dtype=torch.long, device='cpu')
        
        # Clean up tensors to free GPU memory
        del outputs
        if 'semantic_logits' in locals():
            del semantic_logits
        if 'class_logits' in locals():
            del class_logits
        if 'text_features' in locals():
            del text_features
        if 'text_tokens' in locals():
            del text_tokens
        
        # Force GPU memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Extract token information for interpretability
        token_texts = []
        if 'token_texts' in outputs:
            token_texts = outputs['token_texts']
            
        # Get matched class information
        matched_token_info = []
        if best_class.item() >= 0 and best_class.item() < len(token_texts):
            matched_token_info = [token_texts[best_class.item()]]
        
        # Return results (all on CPU)
        return {
            'class_preds': class_preds,
            'mapped_class': best_class.item(),
            'similarity': max_similarity.item(),
            'query_tokens': query_token_texts,
            'matched_tokens': matched_token_info
        }
    
    def generate_boundaries_from_text(self, x, class_descriptions, semantic_data=None, text_mapper=None):
        """
        Generate new class boundaries directly from text descriptions using pretrained CLIP.
        This method processes all class descriptions with the CLIP text encoder and
        finds the most similar tabular features for each description.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor
        class_descriptions : Dict[str, str]
            Dictionary mapping class names to text descriptions
        semantic_data : torch.Tensor, optional
            Not used in this implementation as we use CLIP directly
        text_mapper : object, optional
            Not used in this implementation
            
        Returns:
        --------
        dict
            Dictionary containing new class predictions based on text descriptions
        """
        memory_logger.debug(f"=== STARTING generate_boundaries_from_text with pretrained CLIP ===")
        memory_logger.debug(f"Processing {len(class_descriptions)} class descriptions")
        
        # Pass class descriptions to the forward method for processing together with tabular data
        class_texts = list(class_descriptions.values())
        
        # Run model forward pass with the text descriptions
        if torch.cuda.is_available() and x.device.type == 'cuda':
            with torch.cuda.amp.autocast(enabled=True):
                model_outputs = self.forward(x, class_texts=class_texts)
        else:
            model_outputs = self.forward(x, class_texts=class_texts)
        
        # Create class name to index mapping for the output results
        class_name_to_idx = {name: i for i, name in enumerate(class_descriptions.keys())}
        
        # Extract semantic logits for tabular data
        semantic_logits = model_outputs.get('semantic_logits', None)
        class_logits = model_outputs.get('class_logits', None)
        
        # Process text descriptions with CLIP text encoder (batched processing)
        memory_logger.debug(f"Processing class descriptions with CLIP text encoder")
        class_token_mappings = {}
        text_embeddings = []
        
        # Process in smaller batches to manage memory
        batch_size = 5
        with torch.no_grad():
            for i in range(0, len(class_texts), batch_size):
                batch_texts = class_texts[i:i+batch_size]
                
                # Tokenize on CPU
                text_tokens = self.tokenizer(
                    batch_texts,
                    padding="max_length",
                    truncation=True,
                    max_length=77,
                    return_tensors="pt"
                ).to("cpu")
                
                # Process with CLIP text encoder
                batch_embeddings = self.clip_text_model(**text_tokens).pooler_output
                text_embeddings.append(batch_embeddings)
                
                # Create token mappings for interpretability
                for j, text in enumerate(batch_texts):
                    class_idx = i + j
                    class_name = list(class_descriptions.keys())[class_idx] if class_idx < len(class_descriptions) else f"unknown_{class_idx}"
                    
                    # Get tokens for this text
                    token_ids = text_tokens.input_ids[j].tolist()
                    token_texts = self.tokenizer.convert_ids_to_tokens(token_ids)
                    token_texts = [t for t in token_texts if t not in ['<pad>', '<|startoftext|>', '<|endoftext|>']]
                    
                    # Store token mapping
                    class_token_mappings[class_name] = {
                        'text': text,
                        'tokens': token_texts[:self.num_tokens_per_class],
                        'class_idx': class_idx
                    }
                
                # Clean up batch resources
                del text_tokens
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Concatenate all text embeddings
        all_text_embeddings = torch.cat(text_embeddings, dim=0)
        
        # Create predictions tensor on CPU
        if semantic_logits is not None:
            semantic_logits_cpu = semantic_logits.cpu()
            predictions_shape = semantic_logits_cpu.shape[:-1]  # Remove class dimension
            
            # For each sample, find the most similar class description
            if len(predictions_shape) > 1:  # Handle multi-dimensional case
                predictions = semantic_logits_cpu.argmax(dim=-1)
            else:  # Handle single dimension case
                predictions = semantic_logits_cpu.argmax(dim=-1).unsqueeze(0)
        elif class_logits is not None:
            class_logits_cpu = class_logits.cpu()
            predictions_shape = class_logits_cpu.shape[:-1]
            
            # Fallback to class predictions if no semantic logits
            if len(predictions_shape) > 1:
                predictions = class_logits_cpu.argmax(dim=-1)
            else:
                predictions = class_logits_cpu.argmax(dim=-1).unsqueeze(0)
        else:
            # Default case if no logits available
            predictions = torch.zeros((1, 1), dtype=torch.long)
        
        # Clean up tensors to free memory
        if 'semantic_logits' in locals():
            del semantic_logits
            del semantic_logits_cpu
        if 'class_logits' in locals():
            del class_logits
            del class_logits_cpu
        if 'all_text_embeddings' in locals():
            del all_text_embeddings
        for embedding in text_embeddings:
            del embedding
        del text_embeddings
        
        # Force memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Return results
        return {
            'class_preds': predictions,
            'class_mapping': class_name_to_idx,
            'class_token_mappings': class_token_mappings
        }


class SemanticConsistencyLoss(nn.Module):
    """
    Combined loss function for classification and CLIP-style semantic prediction.
    This loss combines standard classification loss with a contrastive loss for semantic alignment.
    """
    
    def __init__(self, semantic_weight=0.5, temperature=0.07):
        """
        Initialize the loss function.
        
        Parameters:
        -----------
        semantic_weight : float
            Weight for the semantic prediction loss component
        temperature : float
            Temperature parameter for the contrastive loss calculation
        """
        super().__init__()
        self.semantic_weight = semantic_weight
        self.temperature = temperature
        
        # Main classification loss
        self.class_loss = nn.CrossEntropyLoss()
        
        # Semantic similarity loss (CLIP-style contrastive loss)
        self.semantic_loss = nn.CrossEntropyLoss()
        
    def forward(self, outputs, targets):
        """
        Compute the combined loss with CLIP-style contrastive supervision.
        
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
        # Get outputs
        class_logits = outputs['class_logits']
        semantic_logits = outputs.get('semantic_logits', None)
        
        # Get targets
        class_targets = targets['class_targets']
        semantic_targets = targets.get('semantic_targets', None)
        
        # Compute class loss
        class_loss = self.class_loss(class_logits, class_targets)
        
        # Initialize semantic loss
        semantic_loss = torch.tensor(0.0, device=class_loss.device)
        
        # Compute semantic loss if targets are provided
        if semantic_targets is not None and semantic_logits is not None:
            # Reshape if needed
            if len(semantic_logits.shape) > 2:
                semantic_logits_flat = semantic_logits.reshape(-1, semantic_logits.size(-1))
                semantic_targets_flat = semantic_targets.reshape(-1)
            else:
                semantic_logits_flat = semantic_logits
                semantic_targets_flat = semantic_targets
                
            # Apply temperature scaling for contrastive loss
            scaled_logits = semantic_logits_flat / self.temperature
                
            # Apply semantic loss
            semantic_loss = self.semantic_loss(scaled_logits, semantic_targets_flat)
        
        # Combine losses
        total_loss = class_loss + self.semantic_weight * semantic_loss
        
        # Store components for debugging
        self.last_class_loss = class_loss.item()
        self.last_semantic_loss = semantic_loss.item()
        
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


def get_clip_text_embeddings(texts, clip_model, tokenizer, batch_size=5, device="cpu"):
    """
    Process texts using a CLIP text encoder to get their embeddings.
    Memory-optimized version that processes texts in batches on CPU.
    
    Parameters:
    -----------
    texts : list of str
        The texts to encode
    clip_model : transformers.CLIPTextModel
        The CLIP text model to use for encoding
    tokenizer : transformers.CLIPTokenizerFast
        The CLIP tokenizer to use
    batch_size : int
        Maximum batch size for processing texts
    device : str
        Device to use for processing (CPU recommended for memory efficiency)
        
    Returns:
    --------
    torch.Tensor
        Text embeddings with shape [len(texts), hidden_size]
    """
    # Ensure we're using CPU for memory efficiency
    processing_device = device
    
    # Tokenize and encode texts in batches to save memory
    all_embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            # Extract batch
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize with CLIP tokenizer
            tokens = tokenizer(
                batch_texts, 
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=77  # CLIP's standard context length
            ).to(processing_device)
            
            # Get embeddings from CLIP model
            outputs = clip_model(**tokens)
            embeddings = outputs.pooler_output
            
            # Store batch results
            all_embeddings.append(embeddings)
            
            # Clean up this batch's tensors
            del tokens
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Concatenate all batches
    if all_embeddings:
        text_embeddings = torch.cat(all_embeddings, dim=0)
        return text_embeddings
    else:
        # Return empty tensor if no texts were provided
        return torch.zeros((0, clip_model.config.hidden_size), device=processing_device)


def create_semantic_aware_model(base_model, num_semantic_classes=None, freeze_clip=True):
    """
    Factory function to create a semantic-aware model with a pretrained CLIP text encoder.
    
    Parameters:
    -----------
    base_model : nn.Module
        The base model to extend
    num_semantic_classes : int, optional
        Number of semantic classes to predict. If None, will be determined 
        automatically from available semantic data.
    freeze_clip : bool
        Whether to freeze the CLIP text encoder parameters (recommended)
        
    Returns:
    --------
    SemanticAwareClassifier
        The extended model with CLIP text encoder
    """
    if num_semantic_classes is None:
        num_semantic_classes = get_semantic_class_count()
        print(f"Creating semantic-aware model with {num_semantic_classes} semantic classes")
    
    model = SemanticAwareClassifier(base_model, num_semantic_classes)
    
    # Set CLIP text encoder parameters to frozen/trainable based on flag
    if freeze_clip:
        print("Freezing CLIP text encoder parameters")
        for param in model.clip_text_model.parameters():
            param.requires_grad = False
    else:
        print("Fine-tuning CLIP text encoder (this will increase GPU memory usage)")
        for param in model.clip_text_model.parameters():
            param.requires_grad = True
    
    # Count parameters for semantic head components
    clip_params = sum(p.numel() for p in model.clip_text_model.parameters())
    trainable_clip_params = sum(p.numel() for p in model.clip_text_model.parameters() if p.requires_grad)
    projection_params = sum(p.numel() for p in model.semantic_projection.parameters())
    class_embedding_params = sum(p.numel() for p in model.semantic_class_embeddings)
    enhancer_params = sum(p.numel() for p in model.feature_enhancer.parameters())
    
    trainable_params = projection_params + class_embedding_params + enhancer_params + trainable_clip_params
    total_params = clip_params + projection_params + class_embedding_params + enhancer_params
    
    print(f"Semantic head has {total_params:,} parameters ({trainable_params:,} trainable)")
    print(f"  - CLIP Text Encoder: {clip_params:,} ({trainable_clip_params:,} trainable)")
    print(f"  - Projection: {projection_params:,}")
    print(f"  - Class embeddings: {class_embedding_params:,}")
    print(f"  - Feature enhancer: {enhancer_params:,}")
    
    return model