'''
Reformer.py
Based on: https://github.com/thuml/Autoformer/blob/main/models/Reformer.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import ReformerLayer
from layers.Embed import DataEmbedding
import numpy as np


class Model(nn.Module):
    """
    Reformer with O(LlogL) complexity
    - It is notable that Reformer is not proposed for time series forecasting, in that it cannot accomplish the cross attention.
    - Here is only one adaption in BERT-style, other possible implementations can also be acceptable.
    - The hyper-parameters, such as bucket_size and n_hashes, need to be further tuned.
    The official repo of Reformer (https://github.com/lucidrains/reformer-pytorch) can be very helpful, if you have any questiones.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.num_classes = configs.num_classes
        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    ReformerLayer(None, configs.d_model, configs.n_heads, bucket_size=configs.bucket_size,
                                  n_hashes=configs.n_hashes),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        self.fc1 = nn.Linear(configs.c_out * configs.pred_len, self.num_classes)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # add placeholder
        x_enc = torch.cat([x_enc, x_dec[:, -self.pred_len:, :]], dim=1)
        x_mark_enc = torch.cat([x_mark_enc, x_mark_dec[:, -self.pred_len:, :]], dim=1)
        # Reformer: encoder only
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        enc_out = self.projection(enc_out)
        
        enc_out = enc_out[:,-self.pred_len:,:]
        #print(dec_out.shape)
        batch_size,_,enc_in = enc_out.shape
        enc_out = enc_out.view(batch_size,self.pred_len*enc_in)
        
        #print(dec_out.shape)
        enc_out = self.fc1(enc_out)
        enc_out = torch.softmax(enc_out, dim=1)

        if self.output_attention:
            return enc_out, attns
        else:
            return enc_out  # [B, L, D]

