import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from ticl.utils import log_gpu_memory, log_tensor_info, memory_logger, track_tensors_memory

class SemanticAwareClassifier(nn.Module):
    """
    Extension of base TabPFN or similar models with a CLIP-style semantic head.
    This model uses CLIP's text encoder directly to compare tabular features with
    text descriptions using a contrastive approach similar to the original CLIP paper.
    """
    
    def __init__(self, base_model, num_semantic_classes):
        """
        Initialize a semantic-aware classifier that extends a base model.
        
        Parameters:
        -----------
        base_model : nn.Module
            The base tabular model (TabPFN, MotherNet, etc.)
        num_semantic_classes : int
            Number of semantic classes to predict (only used for compatibility)
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
        
        # Store semantic class count for compatibility
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
            
        # Get transformer dimensions from the CLIP model
        transformer_dim = self.clip_text_model.config.hidden_size  # Usually 512 for base model
        
        # Project tabular features to CLIP embedding space with normalization
        self.semantic_projection = nn.Sequential(
            nn.Linear(self.emsize, transformer_dim),
            nn.LayerNorm(transformer_dim),  # Add normalization to stabilize projection
            nn.Dropout(0.1)  # Add dropout to prevent overfitting
        )
        
        # Feature enhancement network to better align tabular features with text space
        # Enhanced with more normalization and activation layers
        self.feature_enhancer = nn.Sequential(
            nn.LayerNorm(transformer_dim),  # Normalize inputs first
            nn.Linear(transformer_dim, transformer_dim * 2),
            nn.LayerNorm(transformer_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),  # Increased dropout for more regularization
            nn.Linear(transformer_dim * 2, transformer_dim),
            nn.LayerNorm(transformer_dim)  # Final normalization layer
        )
        
        # Temperature parameter for similarity scaling (learnable but bounded)
        # Start with a more conservative temperature (0.1 instead of 0.07)
        # This is especially important for training stability
        initial_temp = np.log(1 / 0.1)
        self.logit_scale = nn.Parameter(torch.ones([]) * initial_temp)
        
        # Register a hook to clamp the logit_scale parameter to prevent it from exploding
        def clamp_logit_scale(grad):
            # This will keep logit_scale.exp() between exp(-3) and exp(3)
            max_val = 3.0
            min_val = -3.0
            current_val = self.logit_scale.data
            if current_val > max_val or current_val < min_val:
                self.logit_scale.data.clamp_(min=min_val, max=max_val)
            return grad
        
        self.logit_scale.register_hook(clamp_logit_scale)
    
    def forward(self, x, single_eval_pos=None, class_texts=None):
        """
        Forward pass with CLIP-style text-tabular contrastive learning.
        This implementation directly encodes class text descriptions with CLIP
        and computes similarities with tabular features in a shared embedding space.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor with shape [samples, batch, features]
        single_eval_pos : int, optional
            Position to split training and evaluation data
        class_texts : list of str, optional
            Text descriptions for semantic classes
            
        Returns:
        --------
        dict
            Dictionary containing class_logits and semantic_logits
        """
        log_gpu_memory("Start of SemanticAwareClassifier.forward")
        # Pass the input through the base model's forward method 
        # which properly handles single_eval_pos
        base_output = self.base_model(x, single_eval_pos=single_eval_pos)
        
        # Update semantic classes count if needed
        if isinstance(base_output, torch.Tensor) and len(base_output.shape) >= 3:
            num_output_classes = base_output.shape[-1]
            if num_output_classes != self.num_semantic_classes:
                self.num_semantic_classes = num_output_classes
        
        # Extract features based on model type
        if hasattr(self.base_model, 'features') and self.base_model.features is not None:
            # If the base model directly exposes features (preferred method)
            features = self.base_model.features
        
        # Check for TabFlex model
        elif hasattr(self.base_model, 'get_cls_embedding'):
            # TabFlex models have a get_cls_embedding method for classification token
            features = self.base_model.get_cls_embedding()
            
            # Handle test set specific features for TabFlex
            if single_eval_pos is not None and hasattr(features, 'shape') and len(features.shape) > 1:
                if single_eval_pos < features.shape[0]:
                    # Use only evaluation samples for semantic features
                    features = features[single_eval_pos:]
                    # Average over samples if needed
                    if len(features.shape) > 2:
                        features = features.mean(dim=0)
        
        # Check for TabPFN model with encoder + transformer structure
        elif hasattr(self.base_model, 'transformer_encoder') and hasattr(self.base_model, 'encoder'):
            if len(x) == 3:  # style is given
                style_src, x_src, y_src = x
            else:
                x_src, y_src = x
            
            # Encode input features
            x_encoded = self.base_model.encoder(x_src)
            
            # For semantic features, we'll use features from the test set
            if single_eval_pos is not None and single_eval_pos < x_encoded.shape[0]:
                # Use only evaluation samples for semantic features
                features = x_encoded[single_eval_pos:]
                # Average over samples to get a [batch_size, emsize] tensor
                features = features.mean(dim=0)
            else:
                # Fallback: use all samples and average them
                features = x_encoded.mean(dim=0)
                
        # Check for MotherNet or other models with hidden states
        elif hasattr(self.base_model, 'hidden_states') and self.base_model.hidden_states is not None:
            features = self.base_model.hidden_states[-1]  # Use last layer's hidden states
            
            # Extract test set features if applicable
            if single_eval_pos is not None and single_eval_pos < features.shape[0]:
                features = features[single_eval_pos:]
                
            # Average if needed
            if len(features.shape) > 2:
                features = features.mean(dim=0)
                
        else:
            # Last resort fallback - try to extract from the output
            if len(base_output.shape) == 3:
                # Average over samples to get [batch_size, output_dim]
                features = base_output.mean(dim=0)
            else:
                # Otherwise use as is
                features = base_output
        
        # Handle feature dimension mismatch
        if features.shape[-1] != self.emsize:
            # Reshape features if possible to match emsize
            if features.numel() > 0 and features.numel() % self.emsize == 0:
                features = features.reshape(-1, self.emsize)
            else:
                # Create default features as fallback for training to continue
                features = torch.zeros((features.shape[0] if len(features.shape) > 1 else 1, self.emsize), 
                                       device=features.device, dtype=features.dtype)
        
        # Check features for extreme or NaN values before projection
        if torch.isnan(features).any() or torch.isinf(features).any():
            features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Detect if values are very large, which could indicate instability
            max_val = features.abs().max().item()
            if max_val > 100.0:
                memory_logger.warning(f"Very large feature values detected (max={max_val:.2f}), applying normalization")
                # Apply feature-wise normalization to bring values to reasonable range
                features = features / (features.norm(dim=-1, keepdim=True) + 1e-4)
        
        # Project features to CLIP text model dimension with enhanced error handling
        try:
            # Now using sequential model with built-in normalization and dropout
            projected_features = self.semantic_projection(features)
            
            # Extra check for NaNs after projection
            if torch.isnan(projected_features).any() or torch.isinf(projected_features).any():
                memory_logger.warning("NaN or Inf values detected after projection, using fallback")
                transformer_dim = self.clip_text_model.config.hidden_size
                projected_features = torch.zeros((features.shape[0] if len(features.shape) > 1 else 1, transformer_dim), 
                                                device=features.device)
                # Add small random noise for gradient flow
                projected_features = projected_features + torch.randn_like(projected_features) * 0.01
        except Exception as e:
            # Create fallback projected features with detailed error logging
            transformer_dim = self.clip_text_model.config.hidden_size
            projected_features = torch.zeros((features.shape[0] if len(features.shape) > 1 else 1, transformer_dim), 
                                            device=features.device)
            # Add small random noise for gradient flow
            projected_features = projected_features + torch.randn_like(projected_features) * 0.01
        
        # Add batch dimension if needed (for single sample case)
        if len(projected_features.shape) == 1:
            projected_features = projected_features.unsqueeze(0)
        
        batch_size = projected_features.shape[0]
        
        # Apply feature enhancement with extra safety checks
        try:
            # Feature enhancer now has built-in normalization layers
            tabular_features = self.feature_enhancer(projected_features)
            
            # Check for NaNs after enhancement
            if torch.isnan(tabular_features).any() or torch.isinf(tabular_features).any():
                memory_logger.warning("NaN or Inf values detected after feature enhancement, using fallback")
                tabular_features = projected_features  # Fallback to just the projected features
                # Apply simple normalization instead of the enhancer output
                tabular_features = F.normalize(tabular_features, dim=1)
        except Exception as e:
            memory_logger.error(f"Feature enhancement failed: {e}")
            # Fallback to just normalized projected features
            tabular_features = F.normalize(projected_features, dim=1)
        
        # Average pooling over batch if needed
        if tabular_features.dim() > 2:
            tabular_features = tabular_features.mean(dim=1)
        
        # Final normalization with safety bounds
        # Clip any extreme values before normalization
        tabular_features = torch.clamp(tabular_features, min=-10.0, max=10.0)
        
        # Normalize feature vectors to enable proper cosine similarity
        # Use a small epsilon to prevent division by zero
        tabular_features = F.normalize(tabular_features, dim=1, eps=1e-3)
        
        # ===== Process class texts with CLIP text encoder =====
        text_features = None
        all_token_texts = []
        text_tokens = None  # Track for memory management
        
        # Process class texts if available
        if class_texts and len(class_texts) > 0:
            
            # Create CLIP embeddings for class texts
            # Using a balanced approach for different devices
            with torch.no_grad():
                # Determine the best device for running CLIP
                # For MPS (Apple Silicon) and CUDA, use the same device as model
                # This prevents "Expected all tensors to be on the same device" errors
                if tabular_features.device.type in ['mps', 'cuda']:
                    # For MPS and CUDA, it's safer to run on the same device
                    clip_device = tabular_features.device
                else:
                    # For CPU, continue using CPU
                    clip_device = "cpu"
                
                text_features_list = []
                
                # Process in batches to control memory usage
                batch_size = min(16, len(class_texts))
                
                for i in range(0, len(class_texts), batch_size):
                    batch_texts = class_texts[i:i+batch_size]
                    
                    # Tokenize text batch with CLIP tokenizer
                    text_tokens = self.tokenizer(
                        batch_texts,
                        padding="max_length",
                        truncation=True,
                        max_length=77,
                        return_tensors="pt"
                    )
                    
                    # Move tokens to the appropriate device
                    text_tokens = {k: v.to(clip_device) for k, v in text_tokens.items()}
                    
                    try:
                        # Extract text features with CLIP text encoder
                        batch_outputs = self.clip_text_model(**text_tokens)
                        batch_text_features = batch_outputs.pooler_output
                        text_features_list.append(batch_text_features)
                    except Exception as e:
                        memory_logger.error(f"Error processing text batch: {e}")
                        # Try again with CPU as fallback for any device-specific error
                        try:
                            check = clip_device.type != 'cpu'
                        except:
                            check = clip_device != 'cpu'
                        if check:
                            # Move tokens to CPU
                            cpu_tokens = {k: v.to('cpu') for k, v in text_tokens.items()}
                            try:
                                # Process on CPU
                                batch_outputs = self.clip_text_model.to('cpu')(**cpu_tokens)
                                batch_text_features = batch_outputs.pooler_output
                                text_features_list.append(batch_text_features)
                                # Move model back to original device
                                self.clip_text_model.to(clip_device)
                            except Exception as e2:
                                memory_logger.error(f"CPU fallback also failed: {e2}")
                                # Return empty features as ultimate fallback
                                continue
                    
                    # Store token information for interpretability
                    for j, text in enumerate(batch_texts):
                        # Access tokens through the dictionary (we converted text_tokens to a dict earlier)
                        token_ids = text_tokens['input_ids'][j].tolist()
                        
                        try:
                            # Convert token IDs to actual token texts
                            token_texts = self.tokenizer.convert_ids_to_tokens(token_ids)
                            # Filter out special tokens
                            token_texts = [t for t in token_texts if t not in ['<pad>', '<|startoftext|>', '<|endoftext|>']]
                            
                            # Create token info dictionary
                            class_tokens = {
                                'text': text,
                                'tokens': token_texts[:self.num_tokens_per_class],
                                'class_idx': i + j,
                            }
                            all_token_texts.append(class_tokens)
                        except Exception as e:
                            # Add a simpler version without tokenization
                            class_tokens = {
                                'text': text,
                                'tokens': ['<token_error>'],
                                'class_idx': i + j,
                            }
                            all_token_texts.append(class_tokens)
                    
                    # Clean up batch tensors
                    del text_tokens
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # If we have any features, concatenate them
                if len(text_features_list) > 0:
                    text_features = torch.cat(text_features_list, dim=0)
                    
                    # Move to same device as tabular features if needed for computation
                    if tabular_features.device.type != text_features.device.type:
                        try:
                            # Try to move text features to match tabular features
                            text_features = text_features.to(tabular_features.device)
                        except RuntimeError as e:
                            memory_logger.error(f"Failed to move text features to {tabular_features.device}: {e}")
                            # Fall back to moving tabular features to text device as a last resort
                            tabular_features = tabular_features.to(text_features.device)
                    
                    # Normalize text features for cosine similarity
                    text_features = F.normalize(text_features, dim=1)
                else:
                    # Handle the case where all batches failed
                    memory_logger.warning(f"No text features were successfully processed")
                    text_features = None
                
                # Clean up intermediates
                del text_features_list
        
        # ===== Compute CLIP-style contrastive similarities =====
        # If we have text features, compute cosine similarity scaled by temperature
        if text_features is not None:
            # Check features for NaN values first and fix if needed
            if torch.isnan(tabular_features).any() or torch.isinf(tabular_features).any():
                memory_logger.warning("NaN or inf values detected in tabular features, applying stabilization")
                tabular_features = torch.nan_to_num(tabular_features, nan=0.0, posinf=1.0, neginf=-1.0)
                # Re-normalize after fixing NaNs
                tabular_features = F.normalize(tabular_features, dim=1)
                
            if torch.isnan(text_features).any() or torch.isinf(text_features).any():
                memory_logger.warning("NaN or inf values detected in text features, applying stabilization")
                text_features = torch.nan_to_num(text_features, nan=0.0, posinf=1.0, neginf=-1.0)
                # Re-normalize after fixing NaNs
                text_features = F.normalize(text_features, dim=1)
            
            try:
                # Get temperature-scaled logits (similar to CLIP) with safety bounds
                # Clamp logit_scale to avoid extreme values which can cause overflow
                logit_scale_raw = self.logit_scale.clamp(min=-5, max=5)  # Much tighter bounds
                
                # Use a fixed small scale value for the first few epochs to stabilize training
                # After training stabilizes, we can switch to using the learned parameter
                epoch_count = getattr(self, '_epoch_count', 0)
                warmup_epochs = 10  # Use fixed scale for this many epochs
                
                if epoch_count < warmup_epochs:
                    # Use fixed small scale during warmup 
                    logit_scale = torch.tensor(5.0, device=tabular_features.device)
                else:
                    # Exponential moving average to smooth scale changes
                    logit_scale = torch.exp(logit_scale_raw)
                
                # Additional safety check on the scale
                if torch.isnan(logit_scale) or torch.isinf(logit_scale) or logit_scale > 100:
                    print(f"FORWARD DIAGNOSTIC: Invalid logit scale: {logit_scale.item():.6f}, using safe default")
                    logit_scale = torch.tensor(5.0, device=tabular_features.device)
                
                # Track the scale value used for metrics
                self._last_logit_scale = logit_scale.item()
                
                # Compute similarity: [batch_size, embed_dim] x [embed_dim, n_classes]
                # First compute raw similarity without scaling
                raw_similarity = torch.matmul(tabular_features, text_features.transpose(0, 1))
                
                # Print diagnostic info about the raw similarity values
                raw_max = raw_similarity.abs().max().item()
                
                # Apply a hard safety cap on the scale when raw values are already high
                if raw_max > 0.5:  # If raw cosine similarity is already high
                    adjusted_scale = min(logit_scale.item(), 5.0)
                    print(f"FORWARD DIAGNOSTIC: Raw similarity high ({raw_max:.6f}), capping scale to {adjusted_scale:.6f}")
                    semantic_logits = adjusted_scale * raw_similarity
                else:
                    semantic_logits = logit_scale * raw_similarity
                
                # Check for NaN or Inf in the resulting logits
                if torch.isnan(semantic_logits).any() or torch.isinf(semantic_logits).any():
                    memory_logger.warning("NaN or inf values detected in semantic_logits, applying numeric stabilization")
                    semantic_logits = torch.nan_to_num(semantic_logits, nan=0.0, posinf=100.0, neginf=-100.0)
            except Exception as e:
                memory_logger.error(f"Error computing contrastive similarities: {e}")
                # Create fallback semantic logits with small random values
                semantic_logits = torch.randn(
                    (tabular_features.shape[0], text_features.shape[0]),
                    device=tabular_features.device
                ) * 0.01
        else:
            # No class texts provided - use fallback approach
            
            # Import the semantic data on demand
            try:
                # Instead of using the huge semantic data, create a smaller tensor with the right class count
                semantic_logits = torch.zeros(
                    (tabular_features.shape[0], self.num_semantic_classes),
                    device=tabular_features.device
                )
                
                # Fill with small random values to prevent all-zero gradients
                semantic_logits = torch.randn_like(semantic_logits) * 0.01
            except (ImportError, Exception) as e:
                # Fallback to zeros
                semantic_logits = torch.zeros(
                    (tabular_features.shape[0], self.num_semantic_classes),
                    device=tabular_features.device
                )
        
        # Create the return dictionary with CLIP-style outputs
        result = {
            'class_logits': base_output,
            'semantic_logits': semantic_logits,    # [batch_size, num_classes]
            'tabular_features': tabular_features,  # Normalized tabular features
            'text_features': text_features,        # Normalized text features (if available)
            'token_texts': all_token_texts,        # Text tokenization info
        }
        
        # Clean up intermediate tensors
        del projected_features
        if 'text_tokens' in locals() and text_tokens is not None:
            del text_tokens
        
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
        # Process tabular data with our model
        if torch.cuda.is_available() and x.device.type == 'cuda':
            with torch.cuda.amp.autocast(enabled=True):
                outputs = self.forward(x)
        else:
            outputs = self.forward(x)
        
        # Process text description with CLIP
        with torch.no_grad():
            # Determine device
            if torch.cuda.is_available():
                process_device = "cuda"
            elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
                process_device = "mps"
            else:
                process_device = "cpu"
            
            # Tokenize the input text with the CLIP tokenizer
            text_tokens = self.tokenizer(
                text_description,
                padding="max_length",
                truncation=True,
                max_length=77,
                return_tensors="pt"
            )
            
            # Move to appropriate device
            text_tokens = {k: v.to(process_device) for k, v in text_tokens.items()}
            
            try:
                # Get text embeddings from CLIP text encoder
                text_features = self.clip_text_model(**text_tokens).pooler_output  # [1, hidden_size]
            except Exception as e:
                memory_logger.error(f"Error processing text: {e}")
                # Try with CPU as fallback
                if process_device != 'cpu':
                    cpu_tokens = {k: v.to('cpu') for k, v in text_tokens.items()}
                    model_device = next(self.clip_text_model.parameters()).device
                    self.clip_text_model = self.clip_text_model.to('cpu')
                    text_features = self.clip_text_model(**cpu_tokens).pooler_output
                    self.clip_text_model = self.clip_text_model.to(model_device)
            
            # Extract the tokenized text for interpretability
            try:
                # Access tokens through dictionary
                token_ids = text_tokens['input_ids'][0].tolist()
                # Filter out special tokens and get the actual tokens
                token_texts = self.tokenizer.convert_ids_to_tokens(token_ids)
                # Remove padding, BOS, EOS tokens
                token_texts = [t for t in token_texts if t not in ['<pad>', '<|startoftext|>', '<|endoftext|>']]
                query_token_texts = token_texts[:self.num_tokens_per_class]  # Keep just the first few tokens
            except Exception as e:
                # Fallback to a default value
                query_token_texts = ["<token_error>"]
            
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
                    try:
                        # Access tokens through dictionary
                        token_ids = text_tokens['input_ids'][j].tolist()
                        token_texts = self.tokenizer.convert_ids_to_tokens(token_ids)
                        token_texts = [t for t in token_texts if t not in ['<pad>', '<|startoftext|>', '<|endoftext|>']]
                    except Exception as e:
                        
                        token_texts = ["<token_error>"]
                    
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
    CLIP-style contrastive loss for aligning text descriptions with tabular features.
    This implements the InfoNCE/NT-Xent contrastive loss from the CLIP paper, combined
    with standard classification loss.
    """
    
    def __init__(self, semantic_weight=0.2):  # Reduced weight to 0.1 (from 0.5)
        """
        Initialize the loss function.
        
        Parameters:
        -----------
        semantic_weight : float
            Weight for the semantic contrastive loss component. Default is 0.1.
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
        Compute the combined loss with true CLIP-style contrastive learning.
        This implements symmetric cross-entropy loss over the similarity matrix
        using the InfoNCE formulation from the CLIP paper.
        Parameters:
        -----------
        outputs : dict
            Model outputs containing 'class_logits', 'semantic_logits',
            'tabular_features', and 'text_features'
        targets : dict
            Target values containing 'class_targets' and 'semantic_targets'
            semantic targets are semantic feature types (age, name)
            there can be more or fewer semantic targets than class targets
        Returns:
        --------
        torch.Tensor
            Combined loss value
        """
        # Get class prediction outputs (?, batch_size, n_classes)
        class_logits = outputs['class_logits'].permute(1, 2, 0)
        # Get targets for standard classification
        class_targets = targets['class_targets'].permute(1, 0)
        class_loss = self.class_loss(class_logits, class_targets)
        
        # Get normalized feature vectors
        tabular_features = outputs['tabular_features']
        text_features = outputs['text_features']
        
        # Note: These are already normalized vectors, so this is cosine similarity
        raw_logits = torch.matmul(tabular_features, text_features.t()) * 4.5
        
        # Get semantic targets
        semantic_targets = targets['semantic_targets']
        
         # Transpose targets to [batch_size, n_rows]
        targets_transposed = semantic_targets.permute(1, 0)  # [4, 302]

        batch_size = targets_transposed.shape[0]

        # Initialize loss
        total_loss = 0
        total_rows = 0

        # For each batch item
        for i in range(batch_size):
            # Get logits for this batch item [n_classes]
            batch_logits = raw_logits[i]  # [10]

            # Get all target rows for this batch item [n_rows]
            batch_targets = targets_transposed[i]  # [302]
            n_rows = batch_targets.shape[0]

            if n_rows > 0:  # Handle case where some batch items might have zero rows
                # Compute cross-entropy for this batch item
                batch_loss = F.cross_entropy(
                    batch_logits.unsqueeze(0).expand(n_rows, -1),
                    batch_targets
                )

                # Add to total loss, weighted by number of rows
                total_loss += batch_loss * n_rows
                total_rows += n_rows

            # Normalize loss by total number of rows across all batch items
            if total_rows > 0:
                normalized_loss = total_loss / total_rows
            else:
                normalized_loss = torch.tensor(0.0, device=raw_logits.device, requires_grad=True)
        
        breakpoint()
        #NOTE: make this weighting a hyperparameter, not hardcoded
        total_loss = (self.semantic_weight * normalized_loss) + class_loss
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


def get_clip_text_embeddings(texts, clip_model, tokenizer, batch_size=5, device=None):
    """
    Process texts using a CLIP text encoder to get their embeddings.
    Memory-optimized version that processes texts in batches.
    Works with CPU, CUDA, and MPS devices.
    
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
    device : str, optional
        Device to use for processing. If None, will auto-detect.
        
    Returns:
    --------
    torch.Tensor
        Text embeddings with shape [len(texts), hidden_size]
    """
    # Setup logging
    memory_logger = logging.getLogger("memory_profiling")
    
    # Auto-detect device if not specified
    if device is None:
        if torch.cuda.is_available():
            processing_device = "cuda"
        elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
            processing_device = "mps"
        else:
            processing_device = "cpu"
    else:
        processing_device = device
        
    # Get the device of the first input tensor if applicable
    # This helps ensure we process on the same device as our data
    if isinstance(clip_model, torch.nn.Module):
        model_device = next(clip_model.parameters()).device
        # If the model is on CUDA or MPS, prefer that device
        if model_device.type in ['cuda', 'mps']:
            processing_device = model_device
    
    # Get model's current device for restoration later
    model_device = next(clip_model.parameters()).device
    
    # Move model to processing device if needed
    if model_device != processing_device:
        clip_model = clip_model.to(processing_device)
    
    # Tokenize and encode texts in batches to save memory
    all_embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            # Extract batch
            batch_texts = texts[i:i+batch_size]
            
            try:
                # Tokenize with CLIP tokenizer and move to device
                tokens = tokenizer(
                    batch_texts, 
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=77  # CLIP's standard context length
                )
                tokens = {k: v.to(processing_device) for k, v in tokens.items()}
                
                # Get embeddings from CLIP model
                outputs = clip_model(**tokens)
                embeddings = outputs.pooler_output
                
                # Store batch results
                all_embeddings.append(embeddings)
                
                # Clean up this batch's tensors
                del tokens
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                memory_logger.error(f"Error processing batch of texts: {e}")
                if processing_device != 'cpu':
                    try:
                        # Get tokens on CPU
                        cpu_tokens = tokenizer(
                            batch_texts, 
                            return_tensors="pt",
                            padding="max_length",
                            truncation=True,
                            max_length=77
                        )
                        
                        # Process on CPU
                        cpu_model = clip_model.to('cpu')
                        outputs = cpu_model(**cpu_tokens)
                        embeddings = outputs.pooler_output
                        
                        # Move embeddings to original device
                        embeddings = embeddings.to(processing_device)
                        all_embeddings.append(embeddings)
                        
                        # Move model back
                        clip_model = clip_model.to(processing_device)
                    except Exception as e2:
                        memory_logger.error(f"CPU fallback also failed: {e2}")
                        # Continue with next batch
    
    # Move model back to original device if needed
    if model_device != processing_device:
        clip_model = clip_model.to(model_device)
    
    # Concatenate all batches
    if all_embeddings:
        text_embeddings = torch.cat(all_embeddings, dim=0)
        return text_embeddings
    else:
        # Return empty tensor if no texts were provided or all batches failed
        memory_logger.warning(f"No embeddings were generated - returning empty tensor")
        return torch.zeros((0, clip_model.config.hidden_size), device=processing_device)


def create_semantic_aware_model(base_model, num_semantic_classes=None, freeze_clip=True, semantic_column_metadata=None):
    """
    Factory function to create a CLIP-style semantic-aware model.
    
    Parameters:
    -----------
    base_model : nn.Module
        The base model to extend
    num_semantic_classes : int, optional
        Number of semantic classes to predict. If None, will be determined 
        automatically from available semantic data.
    freeze_clip : bool
        Whether to freeze the CLIP text encoder parameters (recommended)
    semantic_column_metadata : dict, optional
        Metadata about semantic columns in the dataset
        
    Returns:
    --------
    SemanticAwareClassifier
        The extended model with CLIP contrastive learning
    """
    if num_semantic_classes is None:
        num_semantic_classes = get_semantic_class_count()
        print(f"Creating semantic-aware model with {num_semantic_classes} semantic classes")
    
    model = SemanticAwareClassifier(base_model, num_semantic_classes)
    
    # If we have semantic column metadata, store it in the model
    if semantic_column_metadata:
        model.semantic_column_metadata = semantic_column_metadata
        print(f"Added semantic column metadata for {len(semantic_column_metadata)} columns")
    
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
    enhancer_params = sum(p.numel() for p in model.feature_enhancer.parameters())
    logit_scale_params = 1  # Single temperature parameter
    
    trainable_params = projection_params + enhancer_params + trainable_clip_params + logit_scale_params
    total_params = clip_params + projection_params + enhancer_params + logit_scale_params
    
    print(f"Semantic head has {total_params:,} parameters ({trainable_params:,} trainable)")
    print(f"  - CLIP Text Encoder: {clip_params:,} ({trainable_clip_params:,} trainable)")
    print(f"  - Projection: {projection_params:,}")
    print(f"  - Feature enhancer: {enhancer_params:,}")
    print(f"  - Logit scale: {logit_scale_params}")
    
    return model