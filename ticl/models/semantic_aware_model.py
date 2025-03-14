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
        self.num_tokens_per_class = 5  # Predict top-k tokens per semantic class
        
        # Initialize the CLIP tokenizer
        self.tokenizer_name = "openai/clip-vit-base-patch32"
        self.tokenizer = CLIPTokenizerFast.from_pretrained(self.tokenizer_name)
        
        # Optimized transformer architecture targeting ~20M parameters
        # Project input features to a larger dimension
        transformer_dim = 512
        self.semantic_projection = nn.Linear(self.emsize, transformer_dim)
        
        # Powerful transformer for semantic token prediction
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
        
        # Token-level prediction head
        # Expanded prediction space closer to full CLIP vocabulary
        self.max_pred_tokens = 2048
        
        # Initialize the token prediction head:
        # For each semantic class, predict probabilities over tokens
        self.token_prediction = nn.Linear(
            transformer_dim, 
            self.max_pred_tokens
        )
        
        # Enhanced intermediate projection for better feature extraction
        self.intermediate_projection = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim * 2),
            nn.LayerNorm(transformer_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(transformer_dim * 2, transformer_dim)
        )
        
        # Class embedding with larger dimension to condition token predictions
        self.class_embedding = nn.Embedding(
            num_semantic_classes,
            transformer_dim
        )
        
    def forward(self, x, single_eval_pos=None):
        """
        Forward pass with both class and semantic predictions.
        
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
        # Pass the input through the base model's forward method 
        # which properly handles single_eval_pos
        base_output = self.base_model(x, single_eval_pos=single_eval_pos)
        
        # Extract the transformer's encoded features for semantic processing
        # The semantic head expects features with shape [batch_size, emsize]
        # TabPFN outputs are of shape [samples, batch_size, output_dim]
        
        # Get the output of TabPFN's transformer before it goes to the decoder
        if hasattr(self.base_model, 'transformer_encoder') and hasattr(self.base_model, 'encoder'):
            # For TabPFN, we need to reconstruct the encoder features
            # This is a workaround until we refactor the model to expose intermediate features
            # Re-encode the input
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
            # If base_output has shape [samples, batch_size, output_dim]
            if len(base_output.shape) == 3:
                # Average over samples to get [batch_size, output_dim]
                features = base_output.mean(dim=0)
            else:
                # Otherwise use as is (might cause errors if dimensions don't match)
                features = base_output
                        
        # Process features through our CLIP tokenizer-based semantic head
        # Project the features to the transformer dimension
        projected_features = self.semantic_projection(features)
        
        # Add batch dimension if needed (for single sample case)
        if len(projected_features.shape) == 1:
            projected_features = projected_features.unsqueeze(0)
        
        batch_size = projected_features.shape[0]
        
        # Initialize results tensors for token predictions
        all_semantic_logits = []
        all_token_logits = []
        all_token_texts = []
        
        # Process each semantic class separately with class conditioning
        for class_idx in range(self.num_semantic_classes):
            # Get class embedding for this semantic class
            class_embed = self.class_embedding(
                torch.tensor(class_idx, device=projected_features.device)
            ).unsqueeze(0).expand(batch_size, -1)
            
            # Add class embedding to the features as a form of conditioning
            class_conditioned = projected_features + class_embed
            
            # Pass through the transformer
            transformer_output = self.semantic_transformer(class_conditioned)
            
            # Get the output representation (average pooling over batch)
            pooled_output = transformer_output.mean(dim=0, keepdim=True)
            
            # Apply intermediate projection for better feature representation
            enhanced_features = self.intermediate_projection(pooled_output)
            
            # Project to token space (predicting over subset of CLIP vocab)
            # Shape: [1, max_pred_tokens]
            token_logits = self.token_prediction(enhanced_features)
            
            # Store this class's token logits
            all_token_logits.append(token_logits)
            
            # Compute class-level score (max token probability)
            class_score = token_logits.max(dim=1)[0]
            all_semantic_logits.append(class_score)
            
            # For interpretability, decode the top predicted tokens
            # Get top-k token indices
            topk_values, topk_indices = torch.topk(
                token_logits.squeeze(), 
                k=self.num_tokens_per_class
            )
            
            # Convert token indices to actual token strings from CLIP vocab
            # (in practice, we'd convert the indices to the full CLIP vocab,
            # but for simplicity we're keeping it as indices for now)
            # This is a placeholder for actual token decoding
            class_tokens = {
                'indices': topk_indices.cpu().tolist(),
                'scores': torch.sigmoid(topk_values).cpu().tolist(),
                'class_idx': class_idx
            }
            all_token_texts.append(class_tokens)
        
        # Stack semantic logits for all classes
        semantic_logits = torch.cat(all_semantic_logits, dim=0).unsqueeze(0)
        
        # For token logits, maintain the class dimension
        token_logits = torch.cat(all_token_logits, dim=0)
        
        # Return all outputs
        return {
            'class_logits': base_output,
            'semantic_logits': semantic_logits,  # [batch, num_classes]
            'token_logits': token_logits,  # [num_classes, max_pred_tokens]
            'token_texts': all_token_texts  # List of dicts with token info
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
            Dictionary containing class predictions, semantic predictions, and token sets
        """
        outputs = self.forward(x)
        
        # Get class predictions from class logits
        class_preds = outputs['class_logits'].argmax(dim=-1)
        
        # Get semantic class predictions
        semantic_preds = outputs['semantic_logits'].argmax(dim=-1)
        
        # Process token predictions for interpretability
        token_texts = outputs['token_texts']
        
        # Decode the predicted tokens using the CLIP tokenizer
        decoded_tokens = []
        for class_tokens in token_texts:
            # Map to full CLIP vocabulary
            class_idx = class_tokens['class_idx']
            indices = class_tokens['indices']
            scores = class_tokens['scores']
            
            # Decode indices to actual words (if needed)
            try:
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
        # Get model's outputs
        outputs = self.forward(x)
        
        # Get token predictions for all semantic classes
        token_sets = outputs.get('token_texts', [])
        
        # Tokenize the input text description with CLIP
        query_tokens, query_token_texts = text_to_clip_tokens(
            text_description, 
            self.tokenizer, 
            max_tokens=self.max_pred_tokens
        )
        
        # Convert to a set for faster intersection calculation
        query_token_set = set(query_tokens)
        
        # Compare the query tokens with the predicted tokens for each class
        class_similarities = []
        for class_data in token_sets:
            class_idx = class_data['class_idx']
            token_indices = class_data['indices']
            token_scores = class_data['scores']
            
            # Calculate similarity as token overlap weighted by scores
            similarity = 0.0
            for i, token_idx in enumerate(token_indices):
                if token_idx in query_token_set:
                    # If the token from this class is in the query, add its score
                    similarity += token_scores[i]
            
            # Normalize similarity
            if len(token_indices) > 0:
                similarity /= len(token_indices)
                
            class_similarities.append((class_idx, similarity))
        
        # Find the best matching class
        if class_similarities:
            best_class, max_similarity = max(class_similarities, key=lambda x: x[1])
        else:
            best_class, max_similarity = -1, 0.0
            
        # Threshold for considering a match valid
        similarity_threshold = 0.2
        
        # Get class predictions
        semantic_classes = outputs['semantic_logits'].argmax(dim=-1)
        class_logits = outputs['class_logits']
        
        if best_class >= 0 and max_similarity >= similarity_threshold:
            # Find samples with the matched semantic class
            mask = semantic_classes == best_class
            
            # For vector inputs, ensure proper dimensions
            if len(mask.shape) == 1:
                mask = mask.unsqueeze(0)
                
            # Create initial prediction tensor with -1 (no match)
            class_preds = torch.full_like(semantic_classes, -1, dtype=torch.long)
            
            # For matching semantic classes, use the class predictions
            if mask.any():
                class_preds[mask] = class_logits[mask].argmax(dim=-1)
        else:
            # Fallback to regular class predictions if no good match
            class_preds = class_logits.argmax(dim=-1)
        
        # Return predictions with additional context
        return {
            'class_preds': class_preds,
            'mapped_class': best_class,
            'similarity': max_similarity,
            'query_tokens': query_token_texts,
            'matched_tokens': [token_texts[best_class] if best_class >= 0 and best_class < len(token_sets) else {}]
        }
    
    def generate_boundaries_from_text(self, x, class_descriptions, semantic_data=None, text_mapper=None):
        """
        Generate new class boundaries directly from text descriptions using CLIP tokenizer.
        
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
        # Get model outputs
        model_outputs = self.forward(x)
        
        # Get token predictions
        token_sets = model_outputs.get('token_texts', [])
        semantic_logits = model_outputs.get('semantic_logits', None)
        semantic_classes = semantic_logits.argmax(dim=-1) if semantic_logits is not None else None
        class_logits = model_outputs.get('class_logits', None)
        
        # Process each class description
        class_token_mappings = {}
        class_similarities = {}
        
        for class_idx, (class_name, description) in enumerate(class_descriptions.items()):
            # Tokenize the class description
            query_tokens, query_token_texts = text_to_clip_tokens(
                description, 
                self.tokenizer, 
                max_tokens=self.max_pred_tokens
            )
            
            # Store the tokenized description
            class_token_mappings[class_name] = {
                'tokens': query_tokens,
                'token_texts': query_token_texts,
            }
            
            # Calculate similarity to each semantic class
            similarities = []
            query_token_set = set(query_tokens)
            
            for sem_class_data in token_sets:
                sem_class_idx = sem_class_data['class_idx']
                token_indices = sem_class_data['indices']
                token_scores = sem_class_data['scores']
                
                # Calculate token overlap similarity
                sim_score = 0.0
                for i, token_idx in enumerate(token_indices):
                    if token_idx in query_token_set:
                        sim_score += token_scores[i]
                
                # Normalize
                if len(token_indices) > 0:
                    sim_score /= len(token_indices)
                    
                similarities.append((sem_class_idx, sim_score))
            
            # Find best matching semantic class
            if similarities:
                best_sem_class, max_sim = max(similarities, key=lambda x: x[1])
                class_similarities[class_name] = {
                    'semantic_class': best_sem_class,
                    'similarity': max_sim
                }
        
        # Generate predictions based on semantic class mappings
        # For each sample, find which class description it matches best
        predictions = torch.full_like(
            semantic_classes if semantic_classes is not None else torch.zeros_like(class_logits[:, 0]), 
            -1, 
            dtype=torch.long
        )
        
        # Class name to index mapping
        class_name_to_idx = {name: i for i, name in enumerate(class_descriptions.keys())}
        
        # For each sample, assign the class with the best matching semantic class
        if semantic_classes is not None:
            for sample_idx in range(semantic_classes.shape[0]):
                for batch_idx in range(semantic_classes.shape[1]):
                    sample_sem_class = semantic_classes[sample_idx, batch_idx].item()
                    
                    # Find which class description matches this semantic class best
                    best_match = None
                    best_sim = -1
                    
                    for class_name, sim_data in class_similarities.items():
                        if sim_data['semantic_class'] == sample_sem_class and sim_data['similarity'] > best_sim:
                            best_match = class_name
                            best_sim = sim_data['similarity']
                    
                    # Assign the class if we found a match
                    if best_match is not None and best_sim >= 0.1:  # Threshold
                        predictions[sample_idx, batch_idx] = class_name_to_idx[best_match]
        
        # For any unmatched samples, use regular class predictions
        mask = predictions == -1
        if mask.any() and class_logits is not None:
            predictions[mask] = class_logits[mask].argmax(dim=-1)
        
        return {
            'class_preds': predictions,
            'class_mapping': {i: name for i, name in enumerate(class_descriptions.keys())},
            'class_token_mappings': class_token_mappings,
            'class_similarities': class_similarities
        }


class SemanticConsistencyLoss(nn.Module):
    """
    Combined loss function for classification and CLIP token-based semantic prediction.
    """
    
    def __init__(self, semantic_weight=0.5, token_weight=0.3):
        """
        Initialize the loss function.
        
        Parameters:
        -----------
        semantic_weight : float
            Weight for the semantic class prediction loss
        token_weight : float
            Weight for the token-level prediction loss
        """
        super().__init__()
        self.semantic_weight = semantic_weight
        self.token_weight = token_weight
        
        # Main classification loss
        self.class_loss = nn.CrossEntropyLoss()
        
        # Semantic class loss
        self.semantic_loss = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Token prediction loss (for predicting CLIP tokens)
        self.token_loss = nn.BCEWithLogitsLoss(reduction='mean')
        
    def forward(self, outputs, targets):
        """
        Compute the combined loss with CLIP token supervision.
        
        Parameters:
        -----------
        outputs : dict
            Model outputs containing 'class_logits', 'semantic_logits', and 'token_logits'
        targets : dict
            Target values containing 'class_targets', 'semantic_targets',
            and optionally 'token_targets' for CLIP token prediction
            
        Returns:
        --------
        torch.Tensor
            Combined loss value
        """
        # Get the outputs
        class_logits = outputs['class_logits']
        semantic_logits = outputs['semantic_logits']
        token_logits = outputs.get('token_logits', None)
        
        # Get the targets
        class_targets = targets['class_targets']
        semantic_targets = targets.get('semantic_targets', None)
        token_targets = targets.get('token_targets', None)
        
        # Compute the classification loss
        class_loss = self.class_loss(class_logits, class_targets)
        
        # Initialize additional losses
        semantic_loss = torch.tensor(0.0, device=class_loss.device)
        token_loss = torch.tensor(0.0, device=class_loss.device)
        
        # Compute semantic class prediction loss if targets are provided
        if semantic_targets is not None:
            # Handle reshaping if needed
            if len(semantic_logits.shape) > 2:
                semantic_logits_flat = semantic_logits.reshape(-1, semantic_logits.size(-1))
                semantic_targets_flat = semantic_targets.reshape(-1)
            else:
                semantic_logits_flat = semantic_logits
                semantic_targets_flat = semantic_targets
                
            # Apply semantic class loss
            semantic_loss = self.semantic_loss(semantic_logits_flat, semantic_targets_flat)
        
        # Compute token prediction loss if we have token targets
        # The token_targets would be one-hot encoded target tokens from CLIP vocabulary
        if token_targets is not None and token_logits is not None:
            # Each semantic class has its own distribution over tokens
            # token_logits: [num_classes, max_pred_tokens]
            # token_targets: [num_classes, max_pred_tokens]
            
            # Apply sigmoid activation inside the loss
            token_loss = self.token_loss(token_logits, token_targets)
        
        # Combine all losses
        total_loss = class_loss + self.semantic_weight * semantic_loss + self.token_weight * token_loss
        
        # For debugging, store loss components in attributes
        self.last_class_loss = class_loss.item()
        self.last_semantic_loss = semantic_loss.item()
        self.last_token_loss = token_loss.item()
        
        # For backward compatibility, return only the total loss
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
    # Tokenize the text
    tokens = tokenizer(
        text, 
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=77  # CLIP's standard max length
    )
    
    # Get token IDs and convert to list
    token_ids = tokens.input_ids[0].tolist()
    
    # Filter out special tokens
    filtered_ids = [tid for tid in token_ids if tid not in [0, 49406, 49407]]  # Skip BOS, EOS, PAD
    
    # Truncate if needed
    if len(filtered_ids) > max_tokens:
        filtered_ids = filtered_ids[:max_tokens]
    
    # Get text representation for each token
    token_texts = [tokenizer.decode([tid]) for tid in filtered_ids]
    
    return filtered_ids, token_texts


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
    
    # Count parameters for our larger semantic head
    semantic_head_params = sum(p.numel() for p in model.semantic_projection.parameters())
    semantic_head_params += sum(p.numel() for p in model.semantic_transformer.parameters())
    semantic_head_params += sum(p.numel() for p in model.token_prediction.parameters())
    semantic_head_params += sum(p.numel() for p in model.class_embedding.parameters())
    semantic_head_params += sum(p.numel() for p in model.intermediate_projection.parameters())
    
    print(f"Semantic head has {semantic_head_params:,} parameters")
    
    # Verify we're using our 20M parameter budget effectively
    print(f"Transformer encoder: {sum(p.numel() for p in model.semantic_transformer.parameters()):,} parameters")
    print(f"Token prediction head: {sum(p.numel() for p in model.token_prediction.parameters()):,} parameters")
    print(f"Intermediate projection: {sum(p.numel() for p in model.intermediate_projection.parameters()):,} parameters")
    print(f"Using CLIP tokenizer with {model.clip_vocab_size:,} tokens")
    
    return model