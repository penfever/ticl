import torch
from torch.nn import Module, Linear, LayerNorm, Dropout
import torch.nn.functional as F
from torch import nn
import math
from ticl.models.layer import TransformerEncoderSimple


class AttentionLayer(Module):
    """Implement the attention layer. Namely project the inputs to multi-head
    queries, keys and values, call the attention implementation and then
    reproject the output.

    It can be thought of as a decorator (see decorator design patter) of an
    attention layer.

    Arguments
    ---------
        attention: Specific inner attention implementation that just computes a
                   weighted average of values given a similarity of queries and
                   keys.
        d_model: The input feature dimensionality for the queries source
        n_heads: The number of heads for the multi head attention
        d_keys: The dimensionality of the keys/queries
                (default: d_model/n_heads)
        d_values: The dimensionality of the values (default: d_model/n_heads)
        d_model_keys: The input feature dimensionality for keys source
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, d_model_keys=None, event_dispatcher=""):
        super(AttentionLayer, self).__init__()

        # Fill d_keys and d_values
        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)
        d_model_keys = d_model_keys or d_model

        self.inner_attention = attention
        self.query_projection = Linear(d_model, d_keys * n_heads)
        self.key_projection = Linear(d_model_keys, d_keys * n_heads)
        self.value_projection = Linear(d_model_keys, d_values * n_heads)
        self.out_projection = Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.d_keys = d_keys
        self.d_values = d_values
        self.d_model = d_model
        self.d_model_keys = d_model_keys

    def forward(self, queries, keys, values, attn_mask, query_lengths,
                key_lengths):
        """Apply attention to the passed in queries/keys/values after
        projecting them to multiple heads.

        In the argument description we make use of the following sizes

            - N: the batch size
            - L: The maximum length of the queries
            - S: The maximum length of the keys (the actual length per sequence
              is given by the length mask)
            - D: The input feature dimensionality passed in the constructor as
              'd_model'

        Arguments
        ---------
            queries: (N, L, D) The tensor containing the queries
            keys: (N, S, D) The tensor containing the keys
            values: (N, S, D) The tensor containing the values
            attn_mask: An implementation of BaseMask that encodes where each
                       query can attend to
            query_lengths: An implementation of  BaseMask that encodes how
                           many queries each sequence in the batch consists of
            key_lengths: An implementation of BaseMask that encodes how
                         many queries each sequence in the batch consists of

        Returns
        -------
            The new value for each query as a tensor of shape (N, L, D).
        """
        # Debug input shapes
        # print(f"AttentionLayer input shapes:")
        # print(f"  queries: {queries.shape}")
        # print(f"  keys: {keys.shape}")
        # print(f"  values: {values.shape}")
        
        # Extract the dimensions into local variables
        N, L, D = queries.shape
        _, S, D_keys = keys.shape
        
        # Verify shapes match expected dimensions
        if D != self.d_model:
            raise ValueError(f"Query feature dimension {D} doesn't match d_model {self.d_model}")
        if D_keys != self.d_model_keys:
            raise ValueError(f"Key feature dimension {D_keys} doesn't match d_model_keys {self.d_model_keys}")
            
        H = self.n_heads

        try:
            # Project the queries/keys/values
            queries = self.query_projection(queries).view(N, L, H, -1)
            keys = self.key_projection(keys).view(N, S, H, -1)
            values = self.value_projection(values).view(N, S, H, -1)
        except RuntimeError as e:
            print(f"Error in projection. Shapes: queries={queries.shape}, keys={keys.shape}, values={values.shape}")
            print(f"Expected dimensions: d_model={self.d_model}, d_model_keys={self.d_model_keys}, n_heads={H}")
            raise e
            
        # Debug projected shapes
        # print(f"  After projection:")
        # print(f"  queries: {queries.shape}")
        # print(f"  keys: {keys.shape}")
        # print(f"  values: {values.shape}")

        try:
            # Compute the attention
            new_values = self.inner_attention(
                queries,
                keys,
                values,
                attn_mask,
                query_lengths,
                key_lengths
            ).view(N, L, -1)
        except RuntimeError as e:
            print(f"Error in inner_attention. Shapes: queries={queries.shape}, keys={keys.shape}, values={values.shape}")
            raise e
            
        # Debug attention output shape
        # print(f"  new_values before projection: {new_values.shape}")

        try:
            # Project the output and return
            result = self.out_projection(new_values)
            # print(f"  Final output: {result.shape}")
            return result
        except RuntimeError as e:
            print(f"Error in out_projection. Shape: new_values={new_values.shape}")
            print(f"Expected output projection: {self.d_values * self.n_heads} -> {self.d_model}")
            raise e




class FeatureMap(Module):
    """Define the FeatureMap interface."""
    def __init__(self, query_dims):
        super().__init__()
        self.query_dims = query_dims

    def new_feature_map(self, device):
        """Create a new instance of this feature map. In particular, if it is a
        random feature map sample new parameters."""
        raise NotImplementedError()

    def forward_queries(self, x):
        """Encode the queries `x` using this feature map."""
        return self(x)

    def forward_keys(self, x):
        """Encode the keys `x` using this feature map."""
        return self(x)

    def forward(self, x):
        """Encode x using this feature map. For symmetric feature maps it
        suffices to define this function, but for asymmetric feature maps one
        needs to define the `forward_queries` and `forward_keys` functions."""
        raise NotImplementedError()

    @classmethod
    def factory(cls, *args, **kwargs):
        """Return a function that when called with the query dimensions returns
        an instance of this feature map.

        It is inherited by the subclasses so it is available in all feature
        maps.
        """
        def inner(query_dims):
            return cls(query_dims, *args, **kwargs)
        return inner


class ActivationFunctionFeatureMap(FeatureMap):
    """Define a feature map that is simply an element-wise activation
    function."""
    def __init__(self, query_dims, activation_function):
        super().__init__(query_dims)
        self.activation_function = activation_function

    def new_feature_map(self, device):
        return

    def forward(self, x):
        # Check if input shape matches expectations
        if x.dim() != 4:
            print(f"WARNING: ActivationFunctionFeatureMap expected 4D input [batch, seq, heads, dim], got shape {x.shape}")
            # Handle the case where dimensions might be different
            if x.dim() == 3:
                # Try to infer head dimension and reshape
                batch, seq, features = x.shape
                if features % self.query_dims == 0:
                    heads = features // self.query_dims
                    x = x.view(batch, seq, heads, self.query_dims)
                    print(f"  Reshaped to {x.shape}")
        return self.activation_function(x)


elu_feature_map = ActivationFunctionFeatureMap.factory(
    lambda x: torch.nn.functional.elu(x) + 1
)

class HedgehogFeatureMap(FeatureMap):
    def __init__(self, query_dims):
        super().__init__(query_dims)
        self.head_dim = query_dims
        self.layer = nn.Linear(self.head_dim, self.head_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.layer.weight, mean=0.0, std=1 / math.sqrt(self.head_dim))
        nn.init.zeros_(self.layer.bias)

    def new_feature_map(self, device):
        self.to(device)

    def forward(self, x):
        x_lin = self.layer(x)
        x_softmax = F.softmax(x_lin, dim=-1)
        x_neg = F.softmax(-x_lin, dim=-1)
        x_cat = torch.cat([x_softmax, x_neg], dim=-1)
        return x_cat

hedgehog_feature_map = HedgehogFeatureMap.factory()

identity_feature_map = ActivationFunctionFeatureMap.factory(
    lambda x: x
)

class LinearAttention(Module):
    """Implement unmasked attention using dot product of feature maps in
    O(N D^2) complexity.

    Given the queries, keys and values as Q, K, V instead of computing

        V' = softmax(Q.mm(K.t()), dim=-1).mm(V),

    we make use of a feature map function Φ(.) and perform the following
    computation

        V' = normalize(Φ(Q).mm(Φ(K).t())).mm(V).

    The above can be computed in O(N D^2) complexity where D is the
    dimensionality of Q, K and V and N is the sequence length. Depending on the
    feature map, however, the complexity of the attention might be limited.

    Arguments
    ---------
        feature_map: callable, a callable that applies the feature map to the
                     last dimension of a tensor (default: elu(x)+1)
        eps: float, a small number to ensure the numerical stability of the
             denominator (default: 1e-6)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, query_dimensions, feature_map=None, eps=1e-6):
        super(LinearAttention, self).__init__()
        self.feature_map = (
            feature_map(query_dimensions) if feature_map else
            elu_feature_map(query_dimensions)
        )
        self.eps = eps
        self.query_dimensions = query_dimensions

    def forward(self, queries, keys, values, attn_mask, query_lengths,
                key_lengths):
        # Debug input shapes
        # print(f"LinearAttention input shapes:")
        # print(f"  queries: {queries.shape}")
        # print(f"  keys: {keys.shape}")
        # print(f"  values: {values.shape}")
        
        # Check input dimensions
        batch_size, n_queries, n_heads, d_queries = queries.shape
        _, n_keys, _, d_keys = keys.shape
        
        # Check that dimensions are compatible
        if d_queries != self.query_dimensions:
            raise ValueError(f"Query dimension mismatch: got {d_queries}, expected {self.query_dimensions}")
        if d_keys != self.query_dimensions:
            raise ValueError(f"Key dimension mismatch: got {d_keys}, expected {self.query_dimensions}")
        
        # Apply the feature map to the queries and keys
        self.feature_map.new_feature_map(queries.device)
        Q = self.feature_map.forward_queries(queries)
        K = self.feature_map.forward_keys(keys)
        
        # Debug mapped shapes
        # print(f"  After feature mapping:")
        # print(f"  Q: {Q.shape}")
        # print(f"  K: {K.shape}")

        # Apply the key padding mask and make sure that the attn_mask is
        # all_ones
        if attn_mask is not None:
            if not attn_mask.all_ones:
                raise RuntimeError(("LinearAttention does not support arbitrary "
                                    "attention masks"))
        if key_lengths is not None:
            K = K * key_lengths.float_matrix[:, :, None, None]

        # Compute the KV matrix, namely the dot product of keys and values so
        # that we never explicitly compute the attention matrix and thus
        # decrease the complexity
        try:
            KV = torch.einsum("nshd,nshm->nhmd", K, values)
        except RuntimeError as e:
            # If error occurs, provide detailed shape information
            print(f"Error in KV einsum. Shapes: K={K.shape}, values={values.shape}")
            raise e

        # Debug KV matrix shape
        # print(f"  KV: {KV.shape}")

        # Compute the normalizer
        try:
            Z = 1/(torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1))+self.eps)
        except RuntimeError as e:
            print(f"Error in Z einsum. Shapes: Q={Q.shape}, K.sum(dim=1)={K.sum(dim=1).shape}")
            raise e

        # Debug normalizer shape
        # print(f"  Z: {Z.shape}")

        # Finally compute and return the new values
        try:
            V = torch.einsum("nlhd,nhmd,nlh->nlhm", Q, KV, Z)
        except RuntimeError as e:
            print(f"Error in V einsum. Shapes: Q={Q.shape}, KV={KV.shape}, Z={Z.shape}")
            raise e

        # Debug output shape
        # print(f"  V: {V.shape}")

        return V.contiguous()
    

class LinearAttentionTransformerEncoderLayer(Module):
    """Self attention and feed forward network with skip connections.

    This transformer encoder layer implements the same encoder layer as
    PyTorch but is a bit more open for extension by receiving the attention
    implementation as a constructor argument.

    Arguments
    ---------
        attention: The attention implementation to use given as a nn.Module
        d_model: The input feature dimensionality
        d_ff: The dimensionality of the intermediate features after the
              attention (default: d_model*4)
        dropout: The dropout rate to apply to the intermediate features
                 (default: 0.1)
        activation: {'relu', 'gelu'} Which activation to use for the feed
                    forward part of the layer (default: relu)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1,
                 activation="relu", event_dispatcher=""):
        super(LinearAttentionTransformerEncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = Dropout(dropout)
        self.activation = getattr(F, activation)

    def forward(self, x, src_mask=None):
        """Apply the transformer encoder to the input x.

        Arguments
        ---------
            x: The input features of shape (N, L, E) where N is the batch size,
               L is the sequence length (padded) and E is d_model passed in the
               constructor.
            src_mask: None or int
                Split the input into two parts. The first part is the training
                samples and the second part is the testing samples.
        """
        # Debug dimensions coming in
        # print(f"LinearAttentionTransformerEncoderLayer input shape: {x.shape}")

        if isinstance(src_mask, int):
            # tabpfn pattern - split into train and test samples
            single_eval_position = src_mask
            
            # Check dimensions before splitting
            if x.dim() != 3:
                raise ValueError(f"Expected 3D tensor input, got shape {x.shape}")
                
            # Split along sequence dimension (dim=1 for batch_first=True)
            train_samples = x[:,:single_eval_position,:]
            test_samples = x[:,single_eval_position:,:]
            
            # Debug split dimensions
            # print(f"  Train samples shape: {train_samples.shape}")
            # print(f"  Test samples shape: {test_samples.shape}")

            # No attention mask for linear attention
            src_mask = None

            # Run self attention and add it to the input
            # the training samples are only attend to themselves
            attn_left = self.attention(
                train_samples, 
                train_samples, 
                train_samples, 
                attn_mask=None, 
                query_lengths=None,
                key_lengths=None,
            )
            # print(f"  attn_left shape: {attn_left.shape}")

            # the testing samples attend to training samples
            attn_right = self.attention(
                test_samples,
                train_samples,
                train_samples,
                attn_mask=None, 
                query_lengths=None,
                key_lengths=None,
            )
            # print(f"  attn_right shape: {attn_right.shape}")

            # Concatenate results back together
            attn_output = torch.cat([attn_left, attn_right], dim=1)
            # print(f"  attn_output shape: {attn_output.shape}")
        else:
            raise ValueError("LinearAttentionTransformerEncoderLayer only supports integer src_mask")

        # Add residual connection and apply normalization
        x = x + self.dropout(attn_output)

        # Run the fully connected part of the layer
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y))

        output = self.norm2(x+y)
        # print(f"LinearAttentionTransformerEncoderLayer output shape: {output.shape}")

        return output



def get_linear_attention_layers(
    d_model: int,
    n_layer: int,
    d_intermediate: int,
    model = 'linear_attention',
    linear_attention_cfg=None,
    attn_layer_idx=None,
    attn_cfg=None,
    norm_epsilon: float = 1e-5,
    rms_norm: bool = False,
    initializer_cfg=None,
    fused_add_norm=False,
    residual_in_fp32=False,
    device=None,
    dtype=None,
    nheads = 2,
    dropout = 0.0,
    activation = 'gelu',
    pre_norm = False,
    recompute_attn = False,
    all_layers_same_init = False,
    norm_output = False,
    feature_map = 'identity',
):
    if dtype is None:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if model == 'linear_attention':

        if d_model % nheads != 0:
            raise ValueError(f"nheads {nheads} must divide d_model {d_model}!")
            
        if feature_map == "hedgehog":
            from ticl.models.linear_attention import hedgehog_feature_map
            feature_map_func = hedgehog_feature_map
        
        elif feature_map == "hedgehog_shared":
            from ticl.models.linear_attention import hedgehog_feature_map
            shared_feature_map = hedgehog_feature_map(d_model // nheads)
            feature_map_func = lambda x: shared_feature_map

        elif feature_map in ["identity", "elu"]:
            from ticl.models.linear_attention import elu_feature_map
            feature_map_func = elu_feature_map

        elif feature_map == "identity_for_real":
            from ticl.models.linear_attention import identity_feature_map
            feature_map_func = identity_feature_map


        def attention_layer_creator():
            return AttentionLayer(
                attention=LinearAttention(query_dimensions=d_model // nheads, feature_map=feature_map_func),
                d_model=d_model,
                n_heads=nheads,
                d_keys=d_model // nheads,
                d_values=d_model // nheads)
        
        def encoder_layer_creator():
            return LinearAttentionTransformerEncoderLayer(
                attention=attention_layer_creator(),
                d_model=d_model,
                d_ff=d_intermediate,
                dropout=dropout,
                # activation=activation
                )
        # TODO is norm good here?
        linear_model = TransformerEncoderSimple(encoder_layer_creator=encoder_layer_creator, num_layers=n_layer, norm=LayerNorm(d_model))
        
        # Enable debugging mode to help identify shape issues
        linear_model.debug_mode = True
        
        # Print model configuration for debugging
        print(f"Created linear attention model:")
        print(f"  - d_model: {d_model}")
        print(f"  - n_layers: {n_layer}")
        print(f"  - d_intermediate: {d_intermediate}")
        print(f"  - n_heads: {nheads}")
        print(f"  - feature_map: {feature_map}")

        return linear_model
    
    else:
        raise ValueError(f"Unknown model {model}")
