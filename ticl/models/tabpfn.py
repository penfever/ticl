
import torch, wandb
import torch.nn as nn

from ticl.models.layer import TransformerEncoderLayer, TransformerEncoderSimple
from ticl.utils import SeqBN, get_init_method
from ticl.models.encoders import Linear


class TabPFN(nn.Module):
    def __init__(self, *, n_out, emsize, nhead, nhid_factor, nlayers, n_features, dropout=0.0,  y_encoder_layer=None,
                 decoder=None, input_normalization=False, init_method=None, pre_norm=False,
                 activation='gelu', recompute_attn=False, classification_task=True,
                 all_layers_same_init=False, efficient_eval_masking=True, y_encoder=None, tabpfn_zero_weights=False,
                 semantic_feature_p=None):
        super().__init__()
        self.classification_task = classification_task
        self.y_encoder = y_encoder_layer
        nhid = emsize * nhid_factor
        
        # Store semantic feature probability if provided
        if semantic_feature_p is not None:
            self.semantic_feature_p = semantic_feature_p

        def encoder_layer_creator(): return TransformerEncoderLayer(
            emsize, 
            nhead, 
            nhid, 
            dropout, 
            activation=activation,
            pre_norm=pre_norm, 
            recompute_attn=recompute_attn,
        )
        self.transformer_encoder =  TransformerEncoderSimple(encoder_layer_creator, nlayers)
        backbone_size = sum(p.numel() for p in self.transformer_encoder.parameters())
        if wandb.run: wandb.log({"backbone_size": backbone_size})
        print("Number of parameters in backbone: ", backbone_size)

        self.emsize = emsize
        
        # Set up the encoder for the correct number of features
        # For TabPFN with semantic features, classification_adapter.py adds 50 features
        # The n_features parameter passed here should already include these 50 features
        # So we'll make the encoder accept exactly this number of features
        self.encoder = Linear(n_features, emsize, replace_nan_by_zero=True)
        
        # Store the semantic feature probability for reference
        if not hasattr(self, 'semantic_feature_p'):
            self.semantic_feature_p = 0.0
            
        # Print feature configuration for debugging
        if self.semantic_feature_p > 0.0:
            print(f"TabPFN encoder configured for {n_features} features")
            print(f"  - Original features: {n_features - 50}")
            print(f"  - Semantic features: 50")
        self.decoder = decoder(emsize, nhid, n_out) if decoder is not None else nn.Sequential(nn.Linear(emsize, nhid), nn.GELU(), nn.Linear(nhid, n_out))
        self.input_ln = SeqBN(emsize) if input_normalization else None
        self.init_method = init_method
        self.efficient_eval_masking = efficient_eval_masking
        self.tabpfn_zero_weights = tabpfn_zero_weights
        self.n_out = n_out
        self.nhid = nhid
        self.init_weights()

    def init_weights(self):
        if self.init_method is not None:
            self.apply(get_init_method(self.init_method))
        if self.tabpfn_zero_weights:
            for layer in self.transformer_encoder.layers:
                nn.init.zeros_(layer.linear2.weight)
                nn.init.zeros_(layer.linear2.bias)
                attns = layer.self_attn if isinstance(layer.self_attn, nn.ModuleList) else [layer.self_attn]
                for attn in attns:
                    nn.init.zeros_(attn.out_proj.weight)
                    nn.init.zeros_(attn.out_proj.bias)

    def forward(self, src, single_eval_pos=None):
        assert isinstance(src, tuple), 'inputs (src) have to be given as (x,y) or (style,x,y) tuple'
        if single_eval_pos is None: raise ValueError('single_eval_pos has to be given, instead of None.')

        if len(src) == 3:  # style is given
            style_src, x_src, y_src = src
        else:
            x_src, y_src = src
        
        # x_src: (num_samples, batch_size, d_model)
        x_src = self.encoder(x_src)
        y_src = self.y_encoder(y_src.unsqueeze(-1) if len(y_src.shape) < len(x_src.shape) else y_src)

        if self.efficient_eval_masking:
            src_mask = single_eval_pos
        else:
            raise NotImplementedError(f'efficient_eval_masking={self.efficient_eval_masking} is not implemented yet.')

        train_x = x_src[:single_eval_pos] + y_src[:single_eval_pos]
        src = torch.cat([train_x, x_src[single_eval_pos:]], 0)

        if self.input_ln is not None:
            src = self.input_ln(src)

        output = self.transformer_encoder(src, src_mask)
        # decoder is some position-wise operation
        output = self.decoder(output)
        return output[single_eval_pos:]