import torch, wandb
import torch.nn as nn 

from ticl.models.encoders import Linear
from ticl.models.linear_attention import get_linear_attention_layers

from ticl.utils import SeqBN

class TabFlex(nn.Module):
    def __init__(
        self, 
        *, 
        model, 
        n_out, 
        emsize, 
        nhead, 
        nhid_factor, 
        nlayers, 
        n_features, 
        dropout=0.0,  
        y_encoder_layer=None,
        decoder=None, 
        input_normalization=False, 
        init_method=None, 
        pre_norm=False,
        activation='gelu', 
        recompute_attn=False, 
        classification_task=True,
        all_layers_same_init=False, 
        efficient_eval_masking=True, 
        y_encoder=None, 
        tabpfn_zero_weights=False,
        local_nhead=4, 
        norm_output = False,
        feature_map = 'identity',
        linear_attention_cfg=None,
        semantic_feature_p=None,
    ):
        super().__init__()
        self.classification_task = classification_task
        self.y_encoder = y_encoder_layer
        nhid = emsize * nhid_factor
        self.model = model
        
        # Store semantic feature probability if provided
        if semantic_feature_p is not None:
            self.semantic_feature_p = semantic_feature_p
            print(f"TabFlex configured with semantic_feature_p={semantic_feature_p}")
            print(f"TabFlex encoder will handle {n_features} features")
            if semantic_feature_p > 0.0:
                print(f"  - Original features: {n_features - 50}")
                print(f"  - Semantic features: 50")
        
        self.linear_attention = get_linear_attention_layers(
            d_model = emsize,
            n_layer = nlayers,
            d_intermediate = nhid,
            model = model,
            nheads = nhead,
            linear_attention_cfg = linear_attention_cfg,
            norm_output = norm_output,
            feature_map = feature_map,
        )
        backbone_size = sum(p.numel() for p in self.linear_attention.parameters())
        if wandb.run: wandb.log({"backbone_size": backbone_size})
        print("Number of parameters in backbone: ", backbone_size)

        self.emsize = emsize
        
        # Make sure to use the correct number of features for the encoder
        # If semantic features are enabled, we need to explicitly handle their effect
        if not hasattr(self, 'semantic_feature_p'):
            self.semantic_feature_p = 0.0
        
        # Check if we need to include 50 extra features
        if self.semantic_feature_p > 0.0:
            # Save original features count for debugging
            self.original_feature_count = n_features
            print(f"TabFlex encoder layer: Using {n_features} features (including 50 semantic features)")
        else:
            self.original_feature_count = n_features
            
        # Create the encoder with the right feature count
        self.encoder = Linear(n_features, emsize, replace_nan_by_zero=True)
        print(f"TabFlex encoder created with shape: {n_features} -> {emsize}")
        
        self.decoder = decoder(emsize, nhid, n_out) if decoder is not None else nn.Sequential(nn.Linear(emsize, nhid), nn.GELU(), nn.Linear(nhid, n_out))
        self.input_ln = SeqBN(emsize) if input_normalization else None
        self.init_method = init_method
        self.efficient_eval_masking = efficient_eval_masking
        self.tabpfn_zero_weights = tabpfn_zero_weights
        self.n_out = n_out
        self.nhid = nhid

    def forward(self, src, src_mask=None, single_eval_pos=None):
        # Enable debug mode for troubleshooting shape issues
        debug_shapes = True
        
        assert isinstance(src, tuple), 'inputs (src) have to be given as (x,y) or (style,x,y) tuple'
        if single_eval_pos is None: 
            raise ValueError('single_eval_pos has to be given, instead of None.')

        if len(src) == 3:  # style is given
            style_src, x_src, y_src = src
        else:
            x_src, y_src = src
        
        if debug_shapes:
            print(f"Input shapes: x_src={x_src.shape}, y_src={y_src.shape}")
            print(f"Encoder expects input with {self.encoder.in_features} features")
            print(f"Current input has {x_src.shape[-1]} features")
        
        # Check feature count mismatch
        if x_src.shape[-1] != self.encoder.in_features:
            print(f"ERROR: Feature count mismatch! Input has {x_src.shape[-1]} features but encoder expects {self.encoder.in_features}")
            print(f"semantic_feature_p = {getattr(self, 'semantic_feature_p', 'not set')}")
            
            # To help debugging, check if this is the 50 semantic features issue
            if abs(x_src.shape[-1] - self.encoder.in_features) == 50:
                print("This looks like the semantic features issue - input has 50 more/less features than encoder expects")
                
                if x_src.shape[-1] > self.encoder.in_features:
                    print(f"Input has 50 more features than encoder - looks like semantic features were added but encoder wasn't configured for them")
                    # We could try to fix by slicing, but that might lead to data corruption
                else:
                    print(f"Encoder expects 50 more features than input has - looks like encoder was configured for semantic features but they weren't added")
            
            # Add a rescue attempt when the model was created with semantic features but input doesn't have them
            # This allows graceful recovery for the semantic_aware_model case
            if hasattr(self, 'semantic_feature_p') and self.semantic_feature_p > 0.0 and x_src.shape[-1] < self.encoder.in_features:
                print(f"RESCUE ATTEMPT: Creating dummy semantic features to match encoder expectations")
                # Calculate how many features to add (usually 50)
                missing_features = self.encoder.in_features - x_src.shape[-1]
                # Create dummy features filled with zeros
                dummy_features = torch.zeros((*x_src.shape[:-1], missing_features), device=x_src.device, dtype=x_src.dtype)
                # Concatenate with original features
                x_src = torch.cat([x_src, dummy_features], dim=-1)
                print(f"Added {missing_features} dummy features to make shape {x_src.shape}")
        
        try:
            x_src = self.encoder(x_src) # transform n_features to emsize
        except RuntimeError as e:
            print(f"ERROR in encoder: {e}")
            print(f"Input shape: {x_src.shape}, Encoder weight shape: {self.encoder.weight.shape}")
            raise
        
        # transform y as one-hot encoding into emsize
        y_src = self.y_encoder(y_src.unsqueeze(-1) if len(y_src.shape) < len(x_src.shape) else y_src)

        if debug_shapes:
            print(f"After encoding: x_src={x_src.shape}, y_src={y_src.shape}")
            print(f"single_eval_pos={single_eval_pos}")

        assert src_mask is None
        assert self.efficient_eval_masking
        
        # Follow the same pattern as TabPFN for consistency
        train_x = x_src[:single_eval_pos] + y_src[:single_eval_pos]
        src = torch.cat([train_x, x_src[single_eval_pos:]], 0) # concatenate the training sequence and the test point

        if debug_shapes:
            print(f"After cat: src.shape={src.shape}")

        if self.input_ln is not None:
            src = self.input_ln(src)
            
        if self.model in ['linear_attention']:
            # Shape debugging and checking
            if debug_shapes:
                print(f"Before permute: src.shape = {src.shape}")
            
            # Convert from (seq_len, batch_size, emsize) to (batch_size, seq_len, emsize)
            # for the linear attention which expects batch_first=True
            src = src.permute(1, 0, 2)
            
            if debug_shapes:
                print(f"After permute: src.shape = {src.shape}")
            
            try:
                # Pass single_eval_pos as src_mask to match TabPFN's pattern
                output = self.linear_attention(src, single_eval_pos)
                
                if debug_shapes:
                    print(f"After linear_attention: output.shape = {output.shape}")
                
                # Convert back to (seq_len, batch_size, emsize) for the decoder
                output = output.permute(1, 0, 2)
                
                if debug_shapes:
                    print(f"After re-permute: output.shape = {output.shape}")
            except RuntimeError as e:
                print(f"ERROR in linear_attention with src.shape={src.shape}, single_eval_pos={single_eval_pos}")
                raise e
        else:
            raise NotImplementedError(f"Model {self.model} is not implemented yet.")
        
        try:
            # Apply the decoder
            output = self.decoder(output)
            
            if debug_shapes:
                print(f"After decoder: output.shape = {output.shape}")
                print(f"Returning output[{single_eval_pos}:] with shape {output[single_eval_pos:].shape}")
            
            # Save features for semantic head models to access
            self.features = output[single_eval_pos:].permute(1, 0, 2)  # shape: [batch, seq, emsize]
            
            # Return only the predictions for the test points
            return output[single_eval_pos:]
        except RuntimeError as e:
            print(f"ERROR in decoder or slicing with output.shape={output.shape}, single_eval_pos={single_eval_pos}")
            raise e
    
    def get_cls_embedding(self):
        """
        Return the features for use by the semantic head.
        This is needed by SemanticAwareClassifier to extract features.
        
        Returns:
        --------
        torch.Tensor
            The features tensor with shape [batch, seq, emsize]
        """
        if hasattr(self, 'features') and self.features is not None:
            return self.features
        else:
            print("WARNING: get_cls_embedding() called but no features are available")
            # Return an empty tensor as placeholder
            return torch.zeros((1, 1, self.emsize), device=next(self.parameters()).device)
            
    def store_feature_count(self, n_features, semantic_feature_p=None):
        """
        Store information about feature counts and semantic features.
        This is used for debugging and model introspection.
        
        Parameters:
        -----------
        n_features : int
            Total number of features
        semantic_feature_p : float, optional
            Probability of using semantic features
        """
        if not hasattr(self, 'semantic_feature_p'):
            if semantic_feature_p is not None:
                self.semantic_feature_p = semantic_feature_p
            else:
                self.semantic_feature_p = 0.0
                
        # Store the feature counts
        self.total_features = n_features
        
        if self.semantic_feature_p > 0.0:
            self.original_features = n_features - 50
            self.semantic_features = 50
        else:
            self.original_features = n_features
            self.semantic_features = 0
            
        print(f"TabFlex feature counts stored: total={self.total_features}, original={self.original_features}, semantic={self.semantic_features}")