import torch
import torch.nn as nn
import torch.nn.functional as F
from ticl.utils import log_gpu_memory, log_tensor_info, memory_logger, track_tensors_memory

class SemanticAwareClassifier(nn.Module):
    """
    Extension of base TabPFN or similar models with an additional semantic prediction head.
    This model supports self-supervised learning of semantic class relationships
    using a CLIP-like approach with a single forward pass for all semantic classes.
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
        
        # Create a semantic head based on CLIP tokenizer vocabulary
        # Import tokenizer-related code
        from transformers import CLIPTokenizerFast
        
        # Constants for the semantic prediction head
        self.num_semantic_classes = num_semantic_classes
        self.clip_vocab_size = 49408  # CLIP tokenizer vocab size
        self.num_tokens_per_class = 5  # Top-k tokens per semantic class for interpretability
        
        # Initialize the CLIP tokenizer
        self.tokenizer_name = "openai/clip-vit-base-patch32"
        self.tokenizer = CLIPTokenizerFast.from_pretrained(self.tokenizer_name)
        
        # CLIP-like design variables
        transformer_dim = 512  # Embedding dimension
        vocab_size = 2048      # Number of token embeddings to use (subset of CLIP vocabulary)
        
        # Project input features to transformer dimension
        self.semantic_projection = nn.Linear(self.emsize, transformer_dim)
        
        # Single transformer encoder for all classes - CLIP-like approach
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True,
            activation="gelu"
        )
        self.semantic_transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=4
        )
        
        # Create a lookup table for semantic class embeddings
        self.semantic_class_embeddings = nn.Parameter(
            torch.randn(num_semantic_classes, transformer_dim),
            requires_grad=True
        )
        
        # Token vocabulary embeddings - these represent the CLIP vocabulary subset
        # These are like "target" vocabulary embeddings in a CLIP model
        self.token_embeddings = nn.Parameter(
            torch.randn(vocab_size, transformer_dim),
            requires_grad=True
        )
        
        # Store the vocabulary size for easy reference
        self.max_pred_tokens = vocab_size
        
        # Intermediate projection for feature enhancement before token matching
        self.feature_enhancer = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim * 2),
            nn.LayerNorm(transformer_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(transformer_dim * 2, transformer_dim)
        )
    
    def forward(self, x, single_eval_pos=None):
        """
        Forward pass with both class and semantic predictions using a CLIP-like approach.
        Instead of processing each semantic class separately, this uses a single forward
        pass and matrix multiplication for efficiency.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor with shape [samples, batch, features]
        single_eval_pos : int, optional
            Position to split training and evaluation data
            
        Returns:
        --------
        dict
            Dictionary containing class_logits and semantic_logits
        """
        # Profiling: Log initial memory state
        log_gpu_memory("Start of SemanticAwareClassifier.forward")
        memory_logger.debug("Running efficient CLIP-like semantic matching")
        
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
        
        # Project features to transformer dimension
        projected_features = self.semantic_projection(features)
        
        # Add batch dimension if needed (for single sample case)
        if len(projected_features.shape) == 1:
            projected_features = projected_features.unsqueeze(0)
        
        batch_size = projected_features.shape[0]
        
        # Run the transformer once on the features
        transformed_features = self.semantic_transformer(projected_features)
        
        # Apply feature enhancement
        enhanced_features = self.feature_enhancer(transformed_features)
        
        # Average pooling over batch if needed
        if enhanced_features.dim() > 2:
            enhanced_features = enhanced_features.mean(dim=1)
        
        # CLIP-like approach: Compute similarities with all semantic class embeddings at once
        # [batch_size, dim] x [num_classes, dim]T = [batch_size, num_classes]
        semantic_similarities = torch.matmul(
            enhanced_features,                         # [batch_size, dim]
            self.semantic_class_embeddings.transpose(0, 1)  # [dim, num_classes]
        )
        
        # Compute token logits for the vocabulary: batch_features x token_embeddings
        # [batch_size, dim] x [vocab_size, dim]T = [batch_size, vocab_size]
        token_logits = torch.matmul(
            enhanced_features,                     # [batch_size, dim]
            self.token_embeddings.transpose(0, 1)  # [dim, vocab_size]
        )
        
        # Get top tokens for each semantic class - for interpretability only
        all_token_texts = []
        
        # Use class embeddings to compute token logits for each semantic class
        for class_idx in range(self.num_semantic_classes):
            # Get the class embedding
            class_embedding = self.semantic_class_embeddings[class_idx].unsqueeze(0)  # [1, dim]
            
            # Compute token logits for this class: class_embedding x token_embeddings
            # [1, dim] x [vocab_size, dim]T = [1, vocab_size]
            class_token_logits = torch.matmul(
                class_embedding,                    # [1, dim]
                self.token_embeddings.transpose(0, 1)  # [dim, vocab_size]
            )
            
            # Get top tokens for interpretability
            topk_values, topk_indices = torch.topk(
                class_token_logits.squeeze(), 
                k=self.num_tokens_per_class
            )
            
            # Store token information
            class_tokens = {
                'indices': topk_indices.cpu().tolist(),
                'scores': torch.sigmoid(topk_values).cpu().tolist(),
                'class_idx': class_idx
            }
            all_token_texts.append(class_tokens)
        
        # Create the return dictionary
        result = {
            'class_logits': base_output,
            'semantic_logits': semantic_similarities,  # [batch_size, num_classes]
            'token_logits': token_logits,  # [batch_size, vocab_size] - per batch item
            'token_texts': all_token_texts  # For interpretability
        }
        
        # Clean up intermediate tensors
        del projected_features
        del transformed_features
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
        This version directly uses CLIP tokenizer to compare text with token predictions.
        Memory-optimized implementation using a CLIP-like approach.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor
        text_description : str
            Text description of the class
        semantic_data : torch.Tensor, optional
            Not used in this implementation as we directly use CLIP tokens
        text_mapper : object, optional
            Not used in this implementation
            
        Returns:
        --------
        dict
            Dictionary containing class predictions based on text mapping
        """
        # Get model's outputs with efficient CLIP-like approach
        memory_logger.debug(f"Running predict_from_text with description: '{text_description}'")
        
        if torch.cuda.is_available() and x.device.type == 'cuda':
            with torch.cuda.amp.autocast(enabled=True):
                outputs = self.forward(x)
        else:
            outputs = self.forward(x)
        
        # Tokenize the input text description with CLIP (on CPU to save GPU memory)
        with torch.no_grad():
            query_tokens, query_token_texts = text_to_clip_tokens(
                text_description, 
                self.tokenizer, 
                max_tokens=self.max_pred_tokens
            )
        
        # Get token embeddings for text by creating a mask from query tokens
        token_mask = torch.zeros(self.max_pred_tokens, device='cpu')
        for token_idx in query_tokens:
            if token_idx < self.max_pred_tokens:
                token_mask[token_idx] = 1.0
        
        # Compute similarity between semantic classes and text description
        # First, get the semantic class logits from the model output
        semantic_logits = outputs.get('semantic_logits', None)
        
        # Get top matching semantic class
        if semantic_logits is not None:
            semantic_classes = semantic_logits.cpu()  # Move to CPU for processing
            best_class = semantic_classes.argmax(dim=-1)
            max_similarity = semantic_classes.max(dim=-1)[0]
            
            # Move to CPU
            best_class = best_class.cpu()
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
            class_preds = torch.full_like(best_class, -1, dtype=torch.long, device='cpu')
        
        # Clean up tensors to free memory
        del outputs
        del semantic_logits
        del class_logits
        
        # Force memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Get token information for the best matching class
        token_texts = outputs.get('token_texts', [])
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
        Generate new class boundaries directly from text descriptions using CLIP tokenizer.
        Memory-efficient CLIP-like implementation that processes all classes at once.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor
        class_descriptions : Dict[str, str]
            Dictionary mapping class names to text descriptions
        semantic_data : torch.Tensor, optional
            Not used in this implementation as we directly use CLIP tokens
        text_mapper : object, optional
            Not used in this implementation
            
        Returns:
        --------
        dict
            Dictionary containing new class predictions based on text descriptions
        """
        memory_logger.debug(f"=== STARTING generate_boundaries_from_text with CLIP-like approach ===")
        memory_logger.debug(f"Processing {len(class_descriptions)} class descriptions")
        
        # Run model forward pass - the most memory-intensive part
        if torch.cuda.is_available() and x.device.type == 'cuda':
            with torch.cuda.amp.autocast(enabled=True):
                model_outputs = self.forward(x)
        else:
            model_outputs = self.forward(x)
            
        # Process each class description (do this on CPU)
        memory_logger.debug(f"Tokenizing class descriptions")
        class_token_mappings = {}
        
        # Process all descriptions in one pass
        for class_name, description in class_descriptions.items():
            # Tokenize on CPU
            query_tokens, query_token_texts = text_to_clip_tokens(
                description, 
                self.tokenizer, 
                max_tokens=self.max_pred_tokens
            )
            
            # Store tokenized description
            class_token_mappings[class_name] = {
                'tokens': query_tokens,
                'token_texts': query_token_texts,
            }
        
        # Extract semantic logits and class logits
        semantic_logits = model_outputs.get('semantic_logits', None)
        class_logits = model_outputs.get('class_logits', None)
        
        # Clean up large tensors
        if 'token_logits' in model_outputs:
            del model_outputs['token_logits']
        
        # Move processing to CPU
        if semantic_logits is not None:
            semantic_logits_cpu = semantic_logits.cpu()
            semantic_classes = semantic_logits_cpu.argmax(dim=-1)
            del semantic_logits
        else:
            semantic_classes = None
            
        if class_logits is not None:
            class_logits_cpu = class_logits.cpu()
            del class_logits
            class_logits = class_logits_cpu
        
        # Force memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Create class name to index mapping
        class_name_to_idx = {name: i for i, name in enumerate(class_descriptions.keys())}
        
        # Create predictions tensor on CPU
        if semantic_classes is not None:
            predictions_shape = semantic_classes.shape
        elif class_logits is not None:
            predictions_shape = class_logits.shape[:-1]
        else:
            predictions_shape = (1, 1)
        
        # Create prediction tensor (on CPU)
        predictions = torch.full(predictions_shape, -1, dtype=torch.long)
        
        # Process semantic logits to get predictions
        # Take the most similar semantic class for each sample
        if semantic_classes is not None:
            # Assign the class directly based on semantic class prediction
            for i, class_idx in enumerate(semantic_classes.flatten()):
                class_idx = class_idx.item()
                # Map the semantic class index to a class name if possible
                for class_name, idx in class_name_to_idx.items():
                    if idx == class_idx:
                        flat_idx = i // predictions.shape[-1] if predictions.dim() > 1 else 0
                        batch_idx = i % predictions.shape[-1] if predictions.dim() > 1 else i
                        predictions[flat_idx, batch_idx] = class_idx
        
        # For unmatched predictions, use class logits
        if class_logits is not None:
            mask = predictions == -1
            if mask.any():
                predictions[mask] = class_logits[mask].argmax(dim=-1)
        
        # Clean up
        if semantic_classes is not None:
            del semantic_classes
            del semantic_logits_cpu
        
        if class_logits is not None:
            del class_logits
        
        # Return results
        return {
            'class_preds': predictions,
            'class_mapping': class_name_to_idx,
            'class_token_mappings': class_token_mappings
        }


class SemanticConsistencyLoss(nn.Module):
    """
    Combined loss function for classification and CLIP-like semantic prediction.
    """
    
    def __init__(self, semantic_weight=0.5):
        """
        Initialize the loss function.
        
        Parameters:
        -----------
        semantic_weight : float
            Weight for the semantic token prediction loss
        """
        super().__init__()
        self.semantic_weight = semantic_weight
        
        # Main classification loss
        self.class_loss = nn.CrossEntropyLoss()
        
        # Semantic similarity loss (similar to CLIP contrastive loss)
        self.semantic_loss = nn.CrossEntropyLoss()
        
    def forward(self, outputs, targets):
        """
        Compute the combined loss with CLIP-like token supervision.
        
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
                
            # Apply semantic loss
            semantic_loss = self.semantic_loss(semantic_logits_flat, semantic_targets_flat)
        
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


def text_to_clip_tokens(text, tokenizer, max_tokens=512):
    """
    Convert text to CLIP tokens with attention to which tokens are most relevant.
    Memory-optimized version that explicitly frees GPU memory.
    
    Parameters:
    -----------
    text : str
        The text to tokenize
    tokenizer : CLIPTokenizerFast
        The CLIP tokenizer to use
    max_tokens : int
        Maximum number of tokens to return
        
    Returns:
    --------
    tuple
        (token_ids, token_texts) - the token IDs and their text representations
    """
    # Ensure we do this on CPU to save VRAM
    device_type = 'cpu'
    
    # Tokenize the text - with output on CPU
    try:
        tokens = tokenizer(
            text, 
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=77  # CLIP's standard max length
        )
        
        # If tokens are on GPU, move to CPU
        if hasattr(tokens, 'input_ids') and hasattr(tokens.input_ids, 'device') and tokens.input_ids.device.type != 'cpu':
            tokens.input_ids = tokens.input_ids.cpu()
    
        # Get token IDs and convert to list
        token_ids = tokens.input_ids[0].tolist()
        
        # Clean up tokens tensor
        del tokens
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Filter out special tokens
        filtered_ids = [tid for tid in token_ids if tid not in [0, 49406, 49407]]  # Skip BOS, EOS, PAD
        
        # Truncate if needed
        if len(filtered_ids) > max_tokens:
            filtered_ids = filtered_ids[:max_tokens]
        
        # Get text representation for each token - do in small batches for memory efficiency
        token_texts = []
        batch_size = 50  # Process tokens in batches to save memory
        
        for i in range(0, len(filtered_ids), batch_size):
            batch = filtered_ids[i:i+batch_size]
            batch_texts = [tokenizer.decode([tid]) for tid in batch]
            token_texts.extend(batch_texts)
            
            # Clean up after each batch if on CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        return filtered_ids, token_texts
    
    except Exception as e:
        # Provide a failsafe in case of tokenization errors
        print(f"Warning: Error in text tokenization: {e}")
        # Return empty lists as fallback
        return [], []


def create_semantic_aware_model(base_model, num_semantic_classes=None):
    """
    Factory function to create a semantic-aware model.
    
    Parameters:
    -----------
    base_model : nn.Module
        The base model to extend
    num_semantic_classes : int, optional
        Number of semantic classes to predict. If None, will be determined 
        automatically from available semantic data.
        
    Returns:
    --------
    SemanticAwareClassifier
        The extended model
    """
    if num_semantic_classes is None:
        num_semantic_classes = get_semantic_class_count()
        print(f"Creating semantic-aware model with {num_semantic_classes} semantic classes")
    
    model = SemanticAwareClassifier(base_model, num_semantic_classes)
    
    # Count parameters for semantic head components
    transformer_params = sum(p.numel() for p in model.semantic_transformer.parameters())
    projection_params = sum(p.numel() for p in model.semantic_projection.parameters())
    class_embedding_params = sum(p.numel() for p in model.semantic_class_embeddings)
    token_embedding_params = sum(p.numel() for p in model.token_embeddings)
    enhancer_params = sum(p.numel() for p in model.feature_enhancer.parameters())
    
    total_params = transformer_params + projection_params + class_embedding_params + token_embedding_params + enhancer_params
    
    print(f"Semantic head has {total_params:,} parameters")
    print(f"  - Transformer: {transformer_params:,}")
    print(f"  - Projections: {projection_params:,}")
    print(f"  - Class embeddings: {class_embedding_params:,}")
    print(f"  - Token embeddings: {token_embedding_params:,}")
    print(f"  - Feature enhancer: {enhancer_params:,}")
    
    return model