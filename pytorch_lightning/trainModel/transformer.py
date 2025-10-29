import copy                  
import math                  

import torch                  
from torch import nn                  
from torch.functional import F

class Embedding(nn.Module):                  
    def __init__(self, d_model: int, vocab_size: int):                  
        """d_model:词嵌入的维度， vocab:词表的大小"""
        super(Embedding, self).__init__()                  
        self.d_model = d_model                  
        self.vocab_size = vocab_size                  
        self.embedding = nn.Embedding(vocab_size, d_model)                  

    def forward(self, x):                  
        # 乘上权重来自于论文 3.4                  
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):                  
    def __init__(self, d_model: int, dropout: float, max_len: int = 5000):                  
        """d_model:词嵌入维度，dropout:置0比率，max_len:最大长度"""
        super(PositionalEncoding, self).__init__()                  
        self.dropout = nn.Dropout(p=dropout)                  

        pe = torch.zeros(max_len, d_model)                  
        position = torch.arange(0, max_len).unsqueeze(1)                  
        # 实现公式 3.5, 10000^(-2i/d_model) = exp(2i × (-ln(10000)/d_model))                  
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))                  

        pe[:, 0::2] = torch.sin(position * div_term)                  
        pe[:, 1::2] = torch.cos(position * div_term)                  

        pe = pe.unsqueeze(0)# (1, max_len, d_model)                  
        self.register_buffer('pe', pe)                  

    def forward(self, x):                  
        # (batch, max_len, d_model)                  
        # 论文不参与反向传播        
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)                  
        return self.dropout(x)

def attention(self, query, key, value, mask=None, dropout=None):                  
    d_k = query.shape[-1] # d_model:词嵌入维度

    # torch.matmul A的最后一个维度 必须等于B的倒数第二个维度                  
    attn = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)                  
    if mask is not None:                  
        attn = attn.masked_fill(mask == 0, -1e9)                  
    # 最后一维进行softmax操作                  
    attn = F.softmax(attn, dim=-1)                  
    if dropout is not None:                  
        attn = dropout(attn)                  
    return torch.matmul(attn, value), attn

class MultiHeadAttention(nn.Module):                  
    def __init__(self, head: int, d_model: int, dropout: float = 0.1):                  
        """head代表头数，d_model代表词嵌⼊的维度"""
        super(MultiHeadAttention, self).__init__()                  

        assert d_model % head == 0, "d_model 必须可以整除 head"                  
        self.d_k = d_model // head# 每个头获得的分割词向量维度d_k                  
        self.head = head                  

        self.w_q = nn.Linear(d_model, d_model, bias=False)                  
        self.w_k = nn.Linear(d_model, d_model, bias=False)                  
        self.w_v = nn.Linear(d_model, d_model, bias=False)                  
        self.w_o = nn.Linear(d_model, d_model, bias=False)                  

        self.dropout = nn.Dropout(dropout)                  

    # 注意力                  
    def attention(self, query, key, value, mask=None, dropout=None):                  
        d_k = query.shape[-1]                  

        # torch.matmul A的最后一个维度 必须等于 B的倒数第二个维度                  
        attn = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)                  
        if mask is not None:                  
            attn = attn.masked_fill(mask == 0, -1e9)                  
        # 最后一维进行softmax操作                  
        attn = F.softmax(attn, dim=-1)                  
        if dropout is not None:                  
            attn = dropout(attn)                  
        return torch.matmul(attn, value), attn                  

    def forward(self, q, k, v, mask=None):                  
        if mask is not None:                  
            mask = mask.unsqueeze(0)                  

        query = self.w_q(q)# Q                  
        key = self.w_k(k)# K                  
        value = self.w_v(v)# V                  

        batch_size = query.size(0)                  
        # 多头切割                  
        # (batch_size, max_len, d_model) -->(batch_size, max_len, head, d_k) -->(batch_size, head, max_len, d_k)                  
        query = query.view(batch_size, -1, self.head, self.d_k).transpose(1, 2)                  
        key = key.view(batch_size, -1, self.head, self.d_k).transpose(1, 2)                  
        value = value.view(batch_size, -1, self.head, self.d_k).transpose(1, 2)                  

        x, self.attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)                  
        # (batch_size, head, max_len, d_k) -> (batch_size, max_len, head * d_k) -> (batch_size, max_len, d_model)                  
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)                  
        return self.w_o(x)

class FeedForward(nn.Module):                  
    def __init__(self, d_model: int, d_ff: int, dropout=0.1):                  
        """d_ff：第二个线性层的输入维度"""
        super(FeedForward, self).__init__()                  
        self.linear_1 = nn.Linear(d_model, d_ff)  # 包含 W₁ 和 b₁
        self.linear_2 = nn.Linear(d_ff, d_model)  # 包含 W₂ 和 b₂
        self.dropout = nn.Dropout(dropout)                  

    def forward(self, x):                  
        # 论文中 3.3 FFN(x)                  
        # self.linear1(x) 实际计算: x @ W₁.T + b₁
        # self.linear2(...) 实际计算: ... @ W₂.T + b₂        
        return self.linear_2(self.dropout(F.relu(self.linear_1(x))))

class LayerNorm(nn.Module):                  
    # 对每个样本不同位置的向量求均值和方差，然后进行归一化                  
    def __init__(self, features: int, eps: float = 1e-6):                  
        """ features, 表示词嵌⼊的维度，样本的特征数量
        eps是⼀个⾜够⼩的数, 在规范化公式的分⺟中出现,防⽌分⺟为0.默认是1e-6."""
        super(LayerNorm, self).__init__()                  
        self.eps = eps                  
        self.alpha = nn.Parameter(torch.ones(features))                  
        self.bias = nn.Parameter(torch.zeros(features))                  

    def forward(self, x):                  
        # 最后一个维度的均值                  
        mean = x.mean(dim=-1, keepdim=True)                  
        # 最后一个维度的标准差                  
        std = x.std(dim=-1, keepdim=True)                  
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class SublayerConnection(nn.Module):                  
    def __init__(self, features: int, dropout: float = 0.1):                  
        super(SublayerConnection, self).__init__()                  
        self.norm = LayerNorm(features)                  
        self.dropout = nn.Dropout(dropout)                  

    def forward(self, x, sublayer):                  
        # 此处和论文中 transformer 给的图略有不同, 先进行 norm 再进行 self-attention 或 feed-forward                  
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):                  
    def __init__(self, features: int, self_attn: MultiHeadAttention,                  
                 feed_forward: FeedForward, dropout: float = 0.1):                  
        super(EncoderLayer, self).__init__()                  

        self.features = features                  
        self.self_attn = self_attn                  
        self.feed_forward = feed_forward                  

        # 编码器层中有两个⼦层连接结构                  
        self.sublayer = nn.ModuleList(                  
            [SublayerConnection(features, dropout) for _ in range(2)]                  
        )                  

    def forward(self, x, mask):                  
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))                  
        return self.sublayer[1](x, self.feed_forward)

def cloneModules(module, N):                  
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):                  
    def __init__(self, layer: EncoderLayer, N: int):                  
        """初始化函数的两个参数分别代表编码器层和编码器层的个数"""
        super(Encoder, self).__init__()                  
        self.layers = cloneModules(layer, N)                  
        self.norm = LayerNorm(layer.features)                  

    def forward(self, x, mask):                  
        for layer in self.layers:                  
            x = layer(x, mask)                  
        return self.norm(x)

class DecoderLayer(nn.Module):                  
    def __init__(self, features: int, self_attn: MultiHeadAttention, cross_attn: MultiHeadAttention,                  
                 feed_forward: FeedForward, dropout: float):                  
        super(DecoderLayer, self).__init__()                  
        self.features = features                  
        self.self_attn = self_attn                  
        self.cross_attn = cross_attn                  
        self.feed_forward = feed_forward                  
        self.sublayer = cloneModules(SublayerConnection(features, dropout), 3)                  

    def forward(self, x, encoder_output, source_mask, target_mask):                  
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, target_mask))                  
        x = self.sublayer[1](x, lambda x: self.cross_attn(x, encoder_output, encoder_output, source_mask))                  
        return self.sublayer[2](x, self.feed_forward)

class Decoder(nn.Module):                  
    def __init__(self, layer: DecoderLayer, N: int):                  
        super(Decoder, self).__init__()                  
        self.layers = cloneModules(layer, N)                  
        self.norm = LayerNorm(layer.features)                  

    def forward(self, x, encoder_output, source_mask, target_mask):                  
        for layer in self.layers:                  
            x = layer(x, encoder_output, source_mask, target_mask)                  
        return self.norm(x)

class Generator(nn.Module):                  
    def __init__(self, d_model, vocab_size):                  
        super(Generator, self).__init__()                  
        self.project = nn.Linear(d_model, vocab_size)                  

    def forward(self, x):                  
        return F.log_softmax(self.project(x), dim=-1)

class Transformer(nn.Module):                  
    def __init__(self, encoder: Encoder, decoder: Decoder,                  
                 source_embed: Embedding, target_embed: Embedding,                  
                 source_pos: PositionalEncoding, target_pos: PositionalEncoding,                  
                 generator: Generator):                  
        super(Transformer, self).__init__()                  
        self.encoder = encoder                  
        self.decoder = decoder                  
        self.source_embed = source_embed                  
        self.target_embed = target_embed                  
        self.source_pos = source_pos                  
        self.target_pos = target_pos                  
        self.generator = generator                  

    def encode(self, source, source_mask):                  
        return self.encoder(self.source_pos(self.source_embed(source)), source_mask)                  

    def decode(self, encoder_output, source_mask, target, target_mask):                  
        return self.decoder(self.target_pos(self.target_embed(target)), encoder_output, source_mask, target_mask)                  

    def forward(self, source, target, source_mask, target_mask):                  
        return self.decode(self.encode(source, source_mask), source_mask, target, target_mask)

def make_transformer(source_vocab: int, target_vocab: int,                  
                      N: int = 6, d_model: int = 512, d_ff: int = 2048, head: int = 8, dropout: float = 0.1):                  
    source_embed = Embedding(d_model, source_vocab)                  
    target_embed = Embedding(d_model, target_vocab)                  

    source_pos = PositionalEncoding(d_model, dropout)                  
    target_pos = PositionalEncoding(d_model, dropout)                  

    c = copy.deepcopy                  
    attn = MultiHeadAttention(head, d_model)                  
    ff = FeedForward(d_model, d_ff, dropout)                  

    # nn.Sequential是一个顺序容器，用于将多个网络层按顺序组合在一起                  
    model = Transformer(                  
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),                  
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),                  
        source_embed, target_embed,                  
        source_pos, target_pos,                  
        Generator(d_model, target_vocab))                  
    return model

