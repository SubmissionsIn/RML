import torch.nn as nn
from torch.nn.functional import normalize
import torch
import copy
import math
import torch.nn.functional as F


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def setEmbedingModel(d_list, d_out):
    return nn.ModuleList([Mlp(d, d, d_out) for d in d_list])


class Mlp(nn.Module):
    """ Transformer Feed-Forward Block """

    def __init__(self, in_dim, mlp_dim, out_dim, dropout_rate=0.2):
        super(Mlp, self).__init__()

        # init layers
        self.fc1 = nn.Linear(in_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, out_dim)
        self.act = nn.GELU()
        if dropout_rate > 0.0:
            self.dropout1 = nn.Dropout(dropout_rate)
            self.dropout2 = nn.Dropout(dropout_rate)
        else:
            self.dropout1 = None
            self.dropout2 = None

    def forward(self, x):
        out = self.fc1(x)
        out = self.act(out)

        if self.dropout1:
            out = self.dropout1(out)
        out = self.fc2(out)
        if self.dropout1:
            out = self.dropout2(out)
        return out


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model

        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


def attention(q, k, v, d_k, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # scores shape is [bs heads view view]

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output, scores


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v):
        bs = q.size(0)

        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * N * sl * d_model/h
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        output_scores, scores = attention(q, k, v, self.d_k, self.dropout)
        # concatenate heads and put through final linear layer
        concat = output_scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)
        output = self.out(concat)

        return output, scores


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.2):
        super().__init__()

        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout_1 = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout_1(F.relu(self.linear_1(x)))
        x = self.dropout_2(self.linear_2(x))
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        x2 = self.norm_1(x)
        output, scores = self.attn(x2, x2, x2)
        x = x + self.dropout_1(output)
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x, scores


class T_Encoders(nn.Module):
    def __init__(self, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, src):
        x = src
        for i in range(self.N):
            x, scores = self.layers[i](x)
        return self.norm(x), scores


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, N, heads, dropout):
        super().__init__()
        self.encoders = T_Encoders(d_model, N, heads, dropout)

    def forward(self, src):
        e_outputs, scores = self.encoders(src)
        return e_outputs, scores


class Network(nn.Module):
    def __init__(self, class_num, feature_dim, contrastive_feature_dim, device, data_dim_list, view_num, multi_blocks=1, multi_heads=1):
        super(Network, self).__init__()
        self.view_num = view_num

        self.embeddinglayers_in = setEmbedingModel(d_list=data_dim_list, d_out=feature_dim).to(device)
        self.MMLEncoder = TransformerEncoder(d_model=feature_dim, N=multi_blocks, heads=multi_heads, dropout=0.3).to(device)        # N=3, H=4
        # self.MMLEncoder = TransformerEncoder(d_model=feature_dim, N=multi_blocks, heads=multi_heads, dropout=0.0).to(device)
        # self.MMLDecoder = TransformerEncoder(d_model=feature_dim, N=multi_blocks, heads=multi_heads, dropout=0.3).to(device)
        # self.embeddinglayers_out = nn.ModuleList([Mlp(self.view_num * feature_dim, d_out, d_out) for d_out in data_dim_list]).to(device)

        self.feature_contrastive_module = nn.Sequential(
            nn.Linear(feature_dim, contrastive_feature_dim),
        )
        self.label_module = nn.Sequential(
            # nn.Linear(contrastive_feature_dim, contrastive_feature_dim),
            # nn.ReLU(),
            nn.Linear(contrastive_feature_dim, class_num),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # [x1 xv] -> embeddings -> Hs -> fusion_H -> fusion_Z -> fusion_Q

        embeddings = []
        for v in range(self.view_num):
            embeddings.append(self.embeddinglayers_in[v](x[v]))
        Tensor = torch.stack(embeddings, dim=1)  # B,view,d

        # for v in range(self.view_num):
        #     x[v] = self.embeddinglayers[v](x[v])
        # T = torch.stack(x, dim=1)  # B,view,d

        H, scores = self.MMLEncoder(Tensor)
        fusion_H = torch.einsum('bvd->bd', H)
        fusion_Z = normalize(self.feature_contrastive_module(fusion_H), dim=1)

        hs = []
        for v in range(self.view_num):
            # hs.append(normalize(H.chunk(self.view_num, dim=1)[v][:, 0, :]))
            hs.append(H.chunk(self.view_num, dim=1)[v][:, 0, :])

        fusion_Q = self.label_module(fusion_Z)

        # Hf = torch.flatten(H, start_dim=1, end_dim=2)
        # xr = []
        # for v in range(self.view_num):
        #     xr.append(self.embeddinglayers_out[v](Hf))
        return fusion_H, fusion_Z, fusion_Q, scores, hs
