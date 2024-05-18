import os
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from flashattention.flash_attn.modules.mha import MHA

def build_model(args):
    model_name = args.model_name
    model = GWASTransformer(args)
    return model


class linear_module(nn.Module):
    
    def __init__(self, input_dim, output_dim, bias=False, batchnorm=False, activate=False, dropout=0.0):
        super(linear_module, self).__init__()
        
        layers = [nn.Linear(input_dim, output_dim, bias=bias)]
        if batchnorm == True:
            layers.append(nn.BatchNorm1d(output_dim))
        if activate == True:
            layers.append(nn.GELU())
        if dropout != 0.0:
            layers.append(nn.Dropout(dropout))

        self.layer = nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.layer(x)
        return x
    

class MLPEncoder(nn.Module):
    
    def __init__(self, args):
        super(MLPEncoder, self).__init__()
        self.args = args

        input_dim = args.input_dim
        hidden_dims = args.hidden_dims
        hidden_dims = [input_dim] + hidden_dims
        dropout = args.dropout

        layers = []
        for i in range(1, len(hidden_dims)):
            layers.append(
                linear_module(input_dim=hidden_dims[i-1], output_dim=hidden_dims[i],
                    bias=False, batchnorm=True, activate=True)
            )

        self.encoder = nn.Sequential(*layers)


    def forward(self, x):
        x = self.encoder(x)
        return x


class BaseMLP(MLPEncoder):
    
    def __init__(self, args):
        super(BaseMLP, self).__init__(args)

        class_num = args.class_num
        hidden_dims = args.hidden_dims

        # regressor
        self.regressor = linear_module(input_dim=hidden_dims[-1], output_dim=class_num,
                    bias=True, batchnorm=False, activate=False)

    def forward(self, x):
        x = self.encoder(x)
        x = self.regressor(x)
        return x


class GWASMLP(BaseMLP):

    def __init__(self, args):
        assert args.filter_dim <= args.input_dim
        args.input_dim = args.filter_dim
        super(GWASMLP, self).__init__(args)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.regressor(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)


class LayerNorm(nn.LayerNorm):

    """Subclass torch's LayerNorm to handle fp16."""
    def forward(self, x: torch.Tensor):
        ori_type = x.dtype
        ret = super().forward(x)
        return ret.type(ori_type)


class QuickGELU(nn.Module):

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):

    def __init__(self, d_model:int, n_head:int, attn_mask: torch.Tensor=None, dropout:float=0.0):
        super(ResidualAttentionBlock, self).__init__()
        self.attn_mask = attn_mask

        # self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.attn = MHA(embed_dim=d_model, num_heads=n_head, dropout=dropout, use_flash_attn=True)

        self.layer_norm1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.layer_norm2 = LayerNorm(d_model)
        
    def attention(self, x):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        # return self.attn(x,x,x, attn_mask=self.attn_mask)[0]
        return self.attn(x)

    def forward(self, x):
        x = x + self.attention(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm2(x))
        return x


class TransformerEncoder(nn.Module):

    def __init__(self, width:int, layers:int, heads:int, attn_mask:torch.Tensor=None, dropout:float=0.0):
        super(TransformerEncoder, self).__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[
            ResidualAttentionBlock(width, heads, attn_mask, dropout) for _ in range(layers)
        ])
    
    def forward(self, x):
        return self.resblocks(x)


class GWASTransformer(nn.Module):
    
    def __init__(self, args):

        super(GWASTransformer, self).__init__()
        self.args = args
        self.k = self.args.kmer
        self.max_len=self.args.max_len
        width = args.width 
        num_layers = args.num_layers
        num_heads = args.num_heads
        # num_embeddings 词汇表的大小， 未做tokenization

        self.embedding = nn.Embedding(num_embeddings=(5**self.k)+1, embedding_dim=width)
        self.pos_embedding = PositionalEncoding(num_hiddens=width, dropout=args.dropout, max_len=int(args.input_dim/self.k))
        filter_dim = int(args.input_dim/self.k)

        self.attn_encoder = TransformerEncoder(width=width, layers=num_layers, heads=num_heads)
        self.layer_norm = LayerNorm(width)

        self.dim_projector = nn.Linear(width, 2)
        
        hidden_dims = args.hidden_dims
        class_num = args.class_num
        self.regressor = nn.Sequential(
            nn.Linear(filter_dim*2, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.Linear(hidden_dims[1], class_num)
        )

    
    def forward(self, x):
        b, l = x.shape
        x = self.embedding(x.long())
        print(x.shape)
        x = self.pos_embedding(x)
        y = self.attn_encoder(x)
        embedding = self.layer_norm(y)
        x_embed = self.dim_projector(embedding).view(b, -1)

        output = self.regressor(x_embed)
        return output 
