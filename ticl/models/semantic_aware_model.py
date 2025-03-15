import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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
        
        # Project tabular features to CLIP embedding space
        self.semantic_projection = nn.Linear(self.emsize, transformer_dim)
        
        # Feature enhancement network to better align tabular features with text space
        self.feature_enhancer = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim * 2),
            nn.LayerNorm(transformer_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(transformer_dim * 2, transformer_dim)
        )
        
        # Temperature parameter for similarity scaling (learnable)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
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
        # Profiling: Log initial memory state
        log_gpu_memory("Start of SemanticAwareClassifier.forward")
        memory_logger.debug("Running CLIP-style contrastive matching")
        
        # Pass the input through the base model's forward method 
        # which properly handles single_eval_pos
        base_output = self.base_model(x, single_eval_pos=single_eval_pos)
        
        # Extract the transformer's encoded features for semantic processing
        # The semantic head expects features with shape [batch_size, emsize]
        memory_logger.debug(f"Base model type: {type(self.base_model).__name__}")
        
        # Before extracting features, check how many classes we might have
        # This helps align the semantic targets with the actual number of classes
        if isinstance(base_output, torch.Tensor):
            if len(base_output.shape) >= 3:  # Check last dimension for standard outputs
                num_output_classes = base_output.shape[-1]
                memory_logger.debug(f"Detected output tensor with {num_output_classes} classes")
                if num_output_classes != self.num_semantic_classes:
                    memory_logger.debug(f"Adjusting semantic class count: {self.num_semantic_classes} -> {num_output_classes}")
                    self.num_semantic_classes = num_output_classes
        
        # This will prioritize the class logits shape from base_output for class count
        memory_logger.debug(f"Using {self.num_semantic_classes} semantic classes to match output dimensions")
        
        # Check if the model has exposed features directly (best option)
        if hasattr(self.base_model, 'features') and self.base_model.features is not None:
            # If the base model directly exposes features (preferred method)
            features = self.base_model.features
            memory_logger.debug(f"Using directly exposed features: {features.shape}")
        
        # Check for TabFlex model
        elif hasattr(self.base_model, 'get_cls_embedding'):
            # TabFlex models have a get_cls_embedding method for classification token
            memory_logger.debug("Detected TabFlex model, using CLS embedding")
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
            # For TabPFN, we need to reconstruct the encoder features
            memory_logger.debug("Detected TabPFN model, reconstructing encoder features")
            
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
            # For MotherNet or similar models that store hidden states
            memory_logger.debug("Using model's hidden states")
            features = self.base_model.hidden_states[-1]  # Use last layer's hidden states
            
            # Extract test set features if applicable
            if single_eval_pos is not None and single_eval_pos < features.shape[0]:
                features = features[single_eval_pos:]
                
            # Average if needed
            if len(features.shape) > 2:
                features = features.mean(dim=0)
                
        else:
            # Last resort fallback - try to extract from the output
            memory_logger.debug(f"Using fallback feature extraction, base_output shape: {base_output.shape}")
            if len(base_output.shape) == 3:
                # Average over samples to get [batch_size, output_dim]
                features = base_output.mean(dim=0)
            else:
                # Otherwise use as is
                features = base_output
                
        memory_logger.debug(f"Extracted features shape: {features.shape}")
        
        # Project features to CLIP text model dimension
        projected_features = self.semantic_projection(features)
        
        # Add batch dimension if needed (for single sample case)
        if len(projected_features.shape) == 1:
            projected_features = projected_features.unsqueeze(0)
        
        batch_size = projected_features.shape[0]
        
        # Apply feature enhancement to match CLIP embedding space
        tabular_features = self.feature_enhancer(projected_features)
        
        # Average pooling over batch if needed
        if tabular_features.dim() > 2:
            tabular_features = tabular_features.mean(dim=1)
        
        # Normalize feature vectors to enable proper cosine similarity
        tabular_features = F.normalize(tabular_features, dim=1)
        
        # ===== Process class texts with CLIP text encoder =====
        text_features = None
        all_token_texts = []
        text_tokens = None  # Track for memory management
        
        # Debug class_texts parameter
        memory_logger.debug(f"Class texts parameter type: {type(class_texts)}")
        if class_texts is None:
            memory_logger.debug("class_texts is None - No class texts provided")
        elif len(class_texts) == 0:
            memory_logger.debug("class_texts is an empty list")
        else:
            memory_logger.debug(f"Found {len(class_texts)} class texts: {class_texts[:3]}...")
        
        if class_texts and len(class_texts) > 0:
            memory_logger.debug(f"Processing {len(class_texts)} class texts with CLIP")
            
            # Create CLIP embeddings for class texts
            # Using a balanced approach for different devices
            with torch.no_grad():
                # Determine the best device for running CLIP
                # For MPS (Apple Silicon) and CUDA, use the same device as model
                # This prevents "Expected all tensors to be on the same device" errors
                if tabular_features.device.type in ['mps', 'cuda']:
                    # For MPS and CUDA, it's safer to run on the same device
                    clip_device = tabular_features.device
                    memory_logger.debug(f"Using {clip_device} for CLIP processing")
                else:
                    # For CPU, continue using CPU
                    clip_device = "cpu"
                    memory_logger.debug(f"Using CPU for CLIP text processing")
                
                text_features_list = []
                
                # Process in batches to control memory usage
                batch_size = min(16, len(class_texts))
                memory_logger.debug(f"Processing {len(class_texts)} texts in batches of {batch_size}")
                
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
                        memory_logger.debug(f"Successfully processed batch {i//batch_size + 1}/{(len(class_texts) + batch_size - 1)//batch_size}")
                    except Exception as e:
                        memory_logger.error(f"Error processing text batch: {e}")
                        # Try again with CPU as fallback for any device-specific error
                        try:
                            check = clip_device.type != 'cpu'
                        except:
                            check = clip_device != 'cpu'
                        if check:
                            memory_logger.debug(f"Retrying with CPU as fallback")
                            # Move tokens to CPU
                            cpu_tokens = {k: v.to('cpu') for k, v in text_tokens.items()}
                            try:
                                # Process on CPU
                                batch_outputs = self.clip_text_model.to('cpu')(**cpu_tokens)
                                batch_text_features = batch_outputs.pooler_output
                                text_features_list.append(batch_text_features)
                                # Move model back to original device
                                self.clip_text_model.to(clip_device)
                                memory_logger.debug(f"Fallback to CPU succeeded")
                            except Exception as e2:
                                memory_logger.error(f"CPU fallback also failed: {e2}")
                                # Return empty features as ultimate fallback
                                memory_logger.warning(f"Skipping CLIP processing for this batch")
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
                            memory_logger.debug(f"Error processing token texts: {e}")
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
                    memory_logger.debug(f"Concatenating {len(text_features_list)} text feature batches")
                    text_features = torch.cat(text_features_list, dim=0)
                    
                    # Move to same device as tabular features if needed for computation
                    if tabular_features.device.type != text_features.device.type:
                        memory_logger.debug(f"Moving text features from {text_features.device} to {tabular_features.device}")
                        try:
                            # Try to move text features to match tabular features
                            text_features = text_features.to(tabular_features.device)
                        except RuntimeError as e:
                            memory_logger.error(f"Failed to move text features to {tabular_features.device}: {e}")
                            # Fall back to moving tabular features to text device as a last resort
                            memory_logger.debug(f"Falling back: Moving tabular features from {tabular_features.device} to {text_features.device}")
                            tabular_features = tabular_features.to(text_features.device)
                    
                    # Normalize text features for cosine similarity
                    text_features = F.normalize(text_features, dim=1)
                    memory_logger.debug(f"Normalized {len(text_features)} text features for cosine similarity")
                else:
                    # Handle the case where all batches failed
                    memory_logger.warning(f"No text features were successfully processed")
                    text_features = None
                
                # Clean up intermediates
                del text_features_list
        
        # ===== Compute CLIP-style contrastive similarities =====
        # If we have text features, compute cosine similarity scaled by temperature
        if text_features is not None:
            # Get temperature-scaled logits (similar to CLIP)
            logit_scale = self.logit_scale.exp()
            
            # Compute similarity: [batch_size, embed_dim] x [embed_dim, n_classes]
            semantic_logits = logit_scale * torch.matmul(
                tabular_features,
                text_features.transpose(0, 1)
            )
        else:
            # No class texts provided - use fallback approach
            memory_logger.debug(f"No class texts provided. Checking if we should use semantic_data...")
            
            # Import the semantic data on demand
            try:
                memory_logger.debug(f"Fallback mode - no class texts provided")
                
                # Instead of using the huge semantic data, create a smaller tensor with the right class count
                semantic_logits = torch.zeros(
                    (tabular_features.shape[0], self.num_semantic_classes),
                    device=tabular_features.device
                )
                memory_logger.debug(f"Created empty semantic_logits with shape: {semantic_logits.shape}")
                
                # Fill with small random values to prevent all-zero gradients
                semantic_logits = torch.randn_like(semantic_logits) * 0.01
                memory_logger.debug(f"Filled semantic_logits with small random values for stable training")
            except (ImportError, Exception) as e:
                memory_logger.debug(f"Error loading semantic data: {e}")
                # Fallback to zeros
                semantic_logits = torch.zeros(
                    (tabular_features.shape[0], self.num_semantic_classes),
                    device=tabular_features.device
                )
                memory_logger.debug(f"Created empty semantic_logits with shape: {semantic_logits.shape}")
        
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
        # Get model's outputs with CLIP-based approach
        memory_logger.debug(f"Running predict_from_text with description: '{text_description}'")
        
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
                
            memory_logger.debug(f"Processing text on {process_device} device")
            
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
                memory_logger.debug(f"Successfully processed text description with shape {text_features.shape}")
            except Exception as e:
                memory_logger.error(f"Error processing text: {e}")
                # Try with CPU as fallback
                if process_device != 'cpu':
                    memory_logger.debug(f"Falling back to CPU processing")
                    cpu_tokens = {k: v.to('cpu') for k, v in text_tokens.items()}
                    model_device = next(self.clip_text_model.parameters()).device
                    self.clip_text_model = self.clip_text_model.to('cpu')
                    text_features = self.clip_text_model(**cpu_tokens).pooler_output
                    self.clip_text_model = self.clip_text_model.to(model_device)
                    memory_logger.debug(f"CPU fallback successful")
            
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
                memory_logger.debug(f"Error extracting token texts: {e}")
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
                    try:
                        # Access tokens through dictionary
                        token_ids = text_tokens['input_ids'][j].tolist()
                        token_texts = self.tokenizer.convert_ids_to_tokens(token_ids)
                        token_texts = [t for t in token_texts if t not in ['<pad>', '<|startoftext|>', '<|endoftext|>']]
                    except Exception as e:
                        memory_logger.debug(f"Error extracting token texts: {e}")
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
    
    def __init__(self, semantic_weight=0.5):
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
        Compute the combined loss with true CLIP-style contrastive learning.
        This implements symmetric cross-entropy loss over the similarity matrix.
        
        Parameters:
        -----------
        outputs : dict
            Model outputs containing 'class_logits', 'semantic_logits',
            'tabular_features', and 'text_features'
        targets : dict
            Target values containing 'class_targets' and 'semantic_targets'
            
        Returns:
        --------
        torch.Tensor
            Combined loss value
        """
        memory_logger.debug(f"=== SemanticConsistencyLoss.forward START ===")
        
        # Log input details
        memory_logger.debug(f"Output keys: {list(outputs.keys())}")
        memory_logger.debug(f"Target keys: {list(targets.keys())}")
        
        # Get class prediction outputs
        class_logits = outputs['class_logits']
        memory_logger.debug(f"Class logits: shape={class_logits.shape}, dtype={class_logits.dtype}")
        
        # Get targets for standard classification
        class_targets = targets['class_targets']
        memory_logger.debug(f"Class targets: shape={class_targets.shape}, dtype={class_targets.dtype}")
        
        # Log sample of class predictions
        try:
            # Convert logits to probabilities
            class_probs = F.softmax(class_logits, dim=-1)
            predicted_classes = torch.argmax(class_probs, dim=-1)
            
            # Compute accuracy
            correct = (predicted_classes == class_targets).float().sum().item()
            total = class_targets.numel()
            accuracy = correct / total * 100
            
            memory_logger.debug(f"Class prediction sample (first 5):")
            for i in range(min(5, predicted_classes.shape[0])):
                true_label = class_targets[i].item()
                pred_label = predicted_classes[i].item()
                confidence = class_probs[i, pred_label].item() * 100
                memory_logger.debug(f"  Sample {i}: True={true_label}, Pred={pred_label}, Confidence={confidence:.1f}%")
            
            memory_logger.debug(f"Batch classification accuracy: {accuracy:.2f}% ({correct}/{total})")
        except Exception as e:
            memory_logger.debug(f"Error analyzing class predictions: {e}")
        
        # Compute standard classification loss
        memory_logger.debug(f"Computing class_loss with criterion type: {type(self.class_loss).__name__}")
        class_loss = self.class_loss(class_logits, class_targets)
        memory_logger.debug(f"Class loss value: {class_loss.item()}")
        
        # Initialize semantic loss
        semantic_loss = torch.tensor(0.0, device=class_loss.device)
        
        # Compute CLIP-style contrastive loss if we have the necessary components
        has_semantic_components = (
            'semantic_logits' in outputs and 
            'semantic_targets' in targets and 
            targets['semantic_targets'] is not None
        )
        
        memory_logger.debug(f"Has semantic components for loss: {has_semantic_components}")
        
        if has_semantic_components:
            semantic_logits = outputs['semantic_logits']
            semantic_targets = targets['semantic_targets']
            
            memory_logger.debug(f"Semantic logits: shape={semantic_logits.shape}, dtype={semantic_logits.dtype}")
            memory_logger.debug(f"Semantic targets: shape={semantic_targets.shape}, dtype={semantic_targets.dtype}")
            
            # Log semantic similarity matrix
            if semantic_logits.shape[0] < 10:  # Only for small batches
                memory_logger.debug(f"Semantic similarity matrix (logits):")
                for i in range(semantic_logits.shape[0]):
                    row_vals = semantic_logits[i].detach().cpu().tolist()
                    if len(row_vals) > 5:
                        row_str = "[" + ", ".join([f"{x:.2f}" for x in row_vals[:5]]) + "...]"
                    else:
                        row_str = "[" + ", ".join([f"{x:.2f}" for x in row_vals]) + "]"
                    memory_logger.debug(f"  Row {i}: {row_str}")
            
            # Check if we actually have valid semantic targets in this batch
            if semantic_targets.numel() == 0:
                memory_logger.debug("Empty semantic targets tensor - skipping semantic loss")
                # Skip semantic loss calculation for this batch
            else:
                # Ensure semantic targets are long type for cross_entropy
                if semantic_targets.dtype != torch.long:
                    memory_logger.debug(f"Converting semantic targets from {semantic_targets.dtype} to torch.long")
                    semantic_targets = semantic_targets.long()
                
                # Get batch size
                batch_size = semantic_logits.shape[0]
                memory_logger.debug(f"Batch size for semantic loss: {batch_size}")
                
                # Log semantic logits shape and class counts
                memory_logger.debug(f"Semantic logits shape: {semantic_logits.shape}")
                if 'text_features' in outputs and outputs['text_features'] is not None:
                    memory_logger.debug(f"Text features shape: {outputs['text_features'].shape}")
                    if outputs['text_features'].shape[0] < 100:  # Only log if reasonably small
                        memory_logger.debug(f"Number of text classes: {outputs['text_features'].shape[0]}")
                        if 'token_texts' in outputs and outputs['token_texts']:
                            memory_logger.debug(f"Text classes: {[t.get('text', 'unknown') for t in outputs['token_texts'][:5]]}...")
                else:
                    memory_logger.debug("No text features found in model output")
                
                # True CLIP-style loss uses the logits directly (already temperature-scaled in forward)
                # Create labels for contrastive learning - diagonal of similarity matrix should be 1s
                labels = torch.arange(batch_size, device=semantic_logits.device)
                
                # Log batch semantic targets stats
                memory_logger.debug(f"Semantic targets shape: {semantic_targets.shape}, min: {semantic_targets.min().item()}, max: {semantic_targets.max().item()}")
                valid_count = (semantic_targets != -100).sum().item()
                valid_percent = valid_count / semantic_targets.numel() * 100
                memory_logger.debug(f"Valid targets count: {valid_count}/{semantic_targets.numel()} ({valid_percent:.1f}%)")
                
                # Log class distribution for semantic targets
                valid_mask = semantic_targets != -100
                if valid_mask.sum() > 0:
                    try:
                        valid_targets = semantic_targets[valid_mask]
                        unique_classes, counts = torch.unique(valid_targets, return_counts=True)
                        class_counts = {int(cls.item()): int(count.item()) for cls, count in zip(unique_classes, counts)}
                        memory_logger.debug(f"Semantic target class distribution: {class_counts}")
                    except Exception as e:
                        memory_logger.debug(f"Error analyzing semantic target classes: {e}")
                
                # For samples with valid targets, compute contrastive loss
                valid_mask = semantic_targets != -100
                
                if valid_mask.sum() > 0:
                    # Filter to valid samples only
                    valid_logits = semantic_logits[valid_mask]
                    valid_targets = semantic_targets[valid_mask]
                    
                    # Ensure targets remain long type
                    valid_targets = valid_targets.long()
                    
                    # Log contrastive loss inputs
                    memory_logger.debug(f"Contrastive loss inputs - logits: {valid_logits.shape}, targets: {valid_targets.shape}, dtype: {valid_targets.dtype}")
                    memory_logger.debug(f"Target class distribution: {torch.bincount(valid_targets)}")
                    
                    # For proper contrastive loss, we need valid samples
                    if valid_logits.shape[0] > 0:
                        try:
                            # Row-wise (text->tabular)
                            memory_logger.debug("Computing text->tabular loss")
                            loss_text_to_tabular = F.cross_entropy(valid_logits, valid_targets)
                            memory_logger.debug(f"Text->tabular loss: {loss_text_to_tabular.item()}")
                            
                            # Column-wise (tabular->text)
                            memory_logger.debug("Computing tabular->text loss")
                            loss_tabular_to_text = F.cross_entropy(valid_logits.t(), valid_targets)
                            memory_logger.debug(f"Tabular->text loss: {loss_tabular_to_text.item()}")
                            
                            # Symmetric loss (mean of both directions)
                            semantic_loss = (loss_text_to_tabular + loss_tabular_to_text) / 2.0
                            memory_logger.debug(f"Combined symmetric semantic loss: {semantic_loss.item()}")
                            
                            # CLIP accuracy (diagnostic)
                            with torch.no_grad():
                                text_to_tabular_preds = valid_logits.argmax(dim=-1)
                                tabular_to_text_preds = valid_logits.t().argmax(dim=-1)
                                
                                text_to_tabular_correct = (text_to_tabular_preds == valid_targets).float().mean().item() * 100
                                tabular_to_text_correct = (tabular_to_text_preds == valid_targets).float().mean().item() * 100
                                
                                memory_logger.debug(f"Text->tabular accuracy: {text_to_tabular_correct:.2f}%")
                                memory_logger.debug(f"Tabular->text accuracy: {tabular_to_text_correct:.2f}%")
                        except Exception as e:
                            memory_logger.debug(f"Error computing semantic loss: {e}")
                            # Fallback to zero semantic loss
                            semantic_loss = torch.tensor(0.0, device=class_loss.device)
                else:
                    memory_logger.debug("No valid targets for semantic loss")
                    # No valid targets in this batch, use zero loss
            
        # Combine losses
        memory_logger.debug(f"Combining losses with semantic_weight={self.semantic_weight}")
        memory_logger.debug(f"  Class loss: {class_loss.item()}")
        memory_logger.debug(f"  Semantic loss: {semantic_loss.item()}")
        
        total_loss = class_loss + self.semantic_weight * semantic_loss
        memory_logger.debug(f"Total combined loss: {total_loss.item()}")
        
        # Store components for debugging
        self.last_class_loss = class_loss.item()
        self.last_semantic_loss = semantic_loss.item()
        
        memory_logger.debug(f"=== SemanticConsistencyLoss.forward END ===")
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
            memory_logger.debug(f"Using model's device {processing_device} for CLIP processing")
        
    memory_logger.debug(f"Processing text embeddings on {processing_device} device")
    
    # Get model's current device for restoration later
    model_device = next(clip_model.parameters()).device
    
    # Move model to processing device if needed
    if model_device != processing_device:
        memory_logger.debug(f"Moving CLIP model from {model_device} to {processing_device}")
        clip_model = clip_model.to(processing_device)
    
    # Tokenize and encode texts in batches to save memory
    all_embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            # Extract batch
            batch_texts = texts[i:i+batch_size]
            memory_logger.debug(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
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
                    memory_logger.debug(f"Trying CPU fallback for problematic batch")
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
                        memory_logger.debug(f"CPU fallback succeeded")
                    except Exception as e2:
                        memory_logger.error(f"CPU fallback also failed: {e2}")
                        # Continue with next batch
    
    # Move model back to original device if needed
    if model_device != processing_device:
        memory_logger.debug(f"Moving CLIP model back from {processing_device} to {model_device}")
        clip_model = clip_model.to(model_device)
    
    # Concatenate all batches
    if all_embeddings:
        memory_logger.debug(f"Concatenating {len(all_embeddings)} batches of embeddings")
        text_embeddings = torch.cat(all_embeddings, dim=0)
        return text_embeddings
    else:
        # Return empty tensor if no texts were provided or all batches failed
        memory_logger.warning(f"No embeddings were generated - returning empty tensor")
        return torch.zeros((0, clip_model.config.hidden_size), device=processing_device)


def create_semantic_aware_model(base_model, num_semantic_classes=None, freeze_clip=True):
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
        
    Returns:
    --------
    SemanticAwareClassifier
        The extended model with CLIP contrastive learning
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