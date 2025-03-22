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
        # Extract the dimensions into local variables
        N, L, D = queries.shape
        _, S, D_keys = keys.shape
        H = self.n_heads

        # Project the queries/keys/values
        queries = self.query_projection(queries).view(N, L, H, -1)
        keys = self.key_projection(keys).view(N, S, H, -1)
        values = self.value_projection(values).view(N, S, H, -1)

        # Compute the attention
        new_values = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            query_lengths,
            key_lengths
        ).view(N, L, -1)

        # Project the output and return
        return self.out_projection(new_values)




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
        # Handle the case where dimensions might be different
        if x.dim() == 3:
            # Try to infer head dimension and reshape
            batch, seq, features = x.shape
            if features % self.query_dims == 0:
                heads = features // self.query_dims
                x = x.view(batch, seq, heads, self.query_dims)
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
    dimensionality of Q, K and V and N is the sequence length.
    """
    def __init__(self, query_dimensions, feature_map=None, eps=1e-4):
        super(LinearAttention, self).__init__()
        self.feature_map = (
            feature_map(query_dimensions) if feature_map else
            elu_feature_map(query_dimensions)
        )
        self.eps = eps
        self.query_dimensions = query_dimensions

    def forward(self, queries, keys, values, attn_mask, query_lengths,
                key_lengths):        
        # Apply the feature map to the queries and keys
        self.feature_map.new_feature_map(queries.device)
        Q = self.feature_map.forward_queries(queries)
        K = self.feature_map.forward_keys(keys)

        # Apply the key padding mask
        if key_lengths is not None:
            K = K * key_lengths.float_matrix[:, :, None, None]

        # Compute the KV matrix (dot product of keys and values)
        KV = torch.einsum("nshd,nshm->nhmd", K, values)

        # Compute the normalizer
        Z = 1/(torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1))+self.eps)

        # Compute and return the new values
        V = torch.einsum("nlhd,nhmd,nlh->nlhm", Q, KV, Z)

        return V.contiguous()
    

class LinearAttentionTransformerEncoderLayer(Module):
    """Self attention and feed forward network with skip connections.

    This transformer encoder layer implements the same encoder layer as
    PyTorch but is a bit more open for extension by receiving the attention
    implementation as a constructor argument.
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
        """Apply the transformer encoder to the input x."""
        if isinstance(src_mask, int):
            # TabPFN pattern - split into train and test samples
            single_eval_position = src_mask
            
            # Split along sequence dimension (dim=1 for batch_first=True)
            train_samples = x[:,:single_eval_position,:]
            test_samples = x[:,single_eval_position:,:]

            # Run self attention - training samples only attend to themselves
            attn_left = self.attention(
                train_samples, 
                train_samples, 
                train_samples, 
                attn_mask=None, 
                query_lengths=None,
                key_lengths=None,
            )

            # Testing samples attend to training samples
            attn_right = self.attention(
                test_samples,
                train_samples,
                train_samples,
                attn_mask=None, 
                query_lengths=None,
                key_lengths=None,
            )

            # Concatenate results back together
            attn_output = torch.cat([attn_left, attn_right], dim=1)
        else:
            raise ValueError("LinearAttentionTransformerEncoderLayer only supports integer src_mask")

        # Add residual connection and apply normalization
        x = x + self.dropout(attn_output)

        # Run the fully connected part of the layer
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y))

        return self.norm2(x+y)



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
                dropout=dropout)
                
        linear_model = TransformerEncoderSimple(encoder_layer_creator=encoder_layer_creator, 
                                                num_layers=n_layer, 
                                                norm=LayerNorm(d_model))

        return linear_model
    
    else:
        raise ValueError(f"Unknown model {model}")
