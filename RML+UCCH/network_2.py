import torch.nn as nn
from torch.nn.functional import normalize
import torch
from torch.nn import Parameter
import copy
import math
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)


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


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # scores shape is [bs heads view view]
    if mask is not None:
        mask = mask.unsqueeze(1).float()
        mask = mask.unsqueeze(-1).matmul(mask.unsqueeze(-2))  # mask shape is [bs 1 view view]
        # mask = mask.unsqueeze(1) #mask shape is [bs 1 1 view]
        scores = scores.masked_fill(mask == 0, -1e9)

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

    def forward(self, q, k, v, mask=None):
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
        output_scores, scores = attention(q, k, v, self.d_k, mask, self.dropout)
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

    def forward(self, x, mask):
        x2 = self.norm_1(x)

        output, scores = self.attn(x2, x2, x2, mask)
        x = x + self.dropout_1(output)
        # x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x, scores


# build a decoder layer with two multi-head attention layers and one feed-forward layer


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x, e_outputs, src_mask, trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, \
                                           src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x


class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        print(x)
        return self.embed(x)


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=200, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        pe = Parameter(self.pe[:, :seq_len])
        if x.is_cuda:
            pe.cuda()
        x = x + pe
        return self.dropout(x)


class Encoders(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        # self.embed = Embedder(vocab_size, d_model)
        # self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, src, mask):
        x = src
        # x = self.embed(src)
        # x = self.pe(x)
        for i in range(self.N):
            x, scores = self.layers[i](x, mask)
        return self.norm(x), scores


class T_Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        # self.embed = Embedder(vocab_size, d_model)
        # self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = trg
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, dropout):
        super().__init__()
        self.encoder = Encoders(src_vocab, d_model, N, heads, dropout)
        self.decoder = T_Decoder(trg_vocab, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab)

    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        # print("DECODER")
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output


class TransformerEncoder(nn.Module):
    def __init__(self, src_vocab, d_model, N, heads, dropout):
        super().__init__()
        self.encoders = Encoders(src_vocab, d_model, N, heads, dropout)
        # self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout)
        # self.out = nn.Linear(d_model, trg_vocab)

    def forward(self, src, src_mask=None):
        e_outputs, scores = self.encoders(src, src_mask)
        # print("DECODER")
        # d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        # output = self.out(e_outputs)
        return e_outputs, scores


class Network(nn.Module):
    def __init__(self, input_size, feature_dim, high_feature_dim, device, data_dim_list, view_num, multi_blocks=1, multi_heads=1):
        super(Network, self).__init__()
        # self.encoders = Encoder(input_size, feature_dim).to(device)
        self.view_num = view_num

        self.embeddinglayers = setEmbedingModel(d_list=data_dim_list, d_out=feature_dim).to(device)
        self.Trans = TransformerEncoder(src_vocab=view_num, d_model=feature_dim, N=multi_blocks, heads=multi_heads, dropout=0.3).to(device)        # N=3, H=4
        self.Trans_2 = TransformerEncoder(src_vocab=view_num, d_model=feature_dim, N=multi_blocks, heads=multi_heads, dropout=0.3).to(device)
        self.embeddinglayers_2 = nn.ModuleList([Mlp(feature_dim, feature_dim, d_out) for d_out in data_dim_list]).to(device)

        self.decoders = Decoder(input_size, feature_dim).to(device)
        self.feature_contrastive_module = nn.Sequential(
            nn.Linear(feature_dim, high_feature_dim),
            # nn.Linear(feature_dim * view_num, high_feature_dim),
        )
        self.feature_contrastive_module_m = nn.Sequential(
            nn.Linear(feature_dim, high_feature_dim),
            # nn.Linear(feature_dim * view_num, high_feature_dim),
        )
        self.label_contrastive_module = nn.Sequential(
            nn.Linear(high_feature_dim, 64),
            nn.Softmax(dim=1)
            # nn.Linear(high_feature_dim, high_feature_dim),
        )

    def forward(self, x):
        # h = self.encoders(x)

        # [x1 xv] -> T -> H -> fusion_H -> Z           Contrastive loss
        #                 H -> Ht -> [xr1 xrv]         Reconstruction loss
        es = []
        for v in range(self.view_num):
            es.append(self.embeddinglayers[v](x[v]))
        T = torch.stack(es, dim=1)  # B,view,d
        H, scores = self.Trans(T)
        # print(x.shape)
        fusion_H = torch.einsum('bvd->bd', H)
        Z = normalize(self.feature_contrastive_module(fusion_H), dim=1)

        hs = []
        for v in range(self.view_num):
            # hs.append(normalize(H.chunk(self.view_num, dim=1)[v][:, 0, :]))
            hs.append(H.chunk(self.view_num, dim=1)[v][:, 0, :])
            # print(x.chunk(self.view_num, dim=1)[i][:, 0, :].shape)

        # h = torch.flatten(x, start_dim=1, end_dim=2)
        # print(h.shape)
        middle_hs = []
        for i in range(self.view_num):
            middle_hs.append(normalize(self.feature_contrastive_module_m(hs[i])))

        Q = self.label_contrastive_module(Z)

        # xr = self.decoders(h)
        Ht, _ = self.Trans_2(H)
        xr = []
        # print(out.shape)
        # print(out.chunk(self.view_num, dim=1)[0][:, 0, :].shape)
        for v in range(self.view_num):
            xr.append(self.embeddinglayers_2[v](Ht.chunk(self.view_num, dim=1)[v][:, 0, :]))
        return xr, fusion_H, Z, Q, scores, hs
