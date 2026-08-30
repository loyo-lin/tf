import math

import torch
import torch.nn as nn


class InputEmbeddings(nn.Module):
    """token id 到向量的嵌入层。

    用法：
        embedding = InputEmbeddings(d_model=512, vocab_size=tokenizer.get_vocab_size())
        x = embedding(token_ids)

    输入形状通常是 (batch, seq_len)，输出形状是 (batch, seq_len, d_model)。
    """

    def __init__(self,d_model:int,vocab_size:int):
        """创建词嵌入表，并记录模型维度 d_model。"""
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self,x):
        """把 token id 映射成 embedding，并乘以 sqrt(d_model) 做尺度校正。"""
        return self.embedding(x)*math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """Transformer 的正弦/余弦位置编码。

    用法：
        pos = PositionalEncoding(d_model=512, seq_len=256, dropout=0.1)
        x = pos(x)

    作用是给 token embedding 加上位置信息，让模型知道词序。
    """

    def __init__(self,d_model:int,seq_len:int,dropout:float)->None:
        """预计算最大长度为 seq_len 的位置编码，并注册为 buffer。"""
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe=torch.zeros(seq_len,d_model)

        position = torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))

        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)

        pe=pe.unsqueeze(0)

        self.register_buffer('pe',pe)

    def forward(self,x):
        """给输入序列加上对应长度的位置编码，再经过 dropout。"""
        x=x+(self.pe[:,:x.shape[1],:]).requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):
    """LayerNorm 层，用于稳定每层的激活分布。

    用法：
        norm = LayerNormalization()
        x = norm(x)
    """

    def __init__(self,eps:float=1e-6):
        """创建可学习缩放参数 alpha 和偏置 bias。"""
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self,x):
        """在最后一个维度上做标准化。"""
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    """Transformer 子层中的前馈网络 FFN。

    用法：
        ffn = FeedForwardBlock(d_model=512, d_ff=2048, dropout=0.1)
        x = ffn(x)

    当前实现是 d_model -> d_ff -> d_model，用于对每个位置独立做特征变换。
    """

    def __init__(self,d_model:int,d_ff:int,dropout:float)->None:
        """创建两层全连接和中间 dropout。"""
        super().__init__()
        self.linear1 = nn.Linear(d_model,d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff,d_model)

    def forward(self,x):
        """执行前馈网络计算。"""
        return self.linear2(self.dropout(self.linear1(x)))


class MultiHeadAttentionBlock(nn.Module):
    """多头注意力模块，可用于 self-attention 或 cross-attention。

    用法：
        attn = MultiHeadAttentionBlock(d_model=512, h=8, dropout=0.1)
        out = attn(q, k, v, mask)

    q/k/v 可以来自同一个序列，也可以 q 来自 decoder、k/v 来自 encoder。
    """

    def __init__(self,d_model:int,h:int,dropout:float)->None:
        """创建 Q/K/V/O 四个线性层，并把 d_model 均分成 h 个头。"""
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0,"d_model must be divisible by h"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model,d_model)
        self.w_k = nn.Linear(d_model,d_model)
        self.w_v = nn.Linear(d_model,d_model)

        self.w_o = nn.Linear(d_model,d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query,key,value,mask,dropout:nn.Dropout):
        """计算缩放点积注意力。

        用法：
            x, scores = MultiHeadAttentionBlock.attention(query, key, value, mask, dropout)

        scores 会被 softmax 成注意力权重；mask 为 0 的位置会被屏蔽。
        返回注意力输出和注意力分数，分数可用于可视化。
        """
        d_k = query.shape[-1]

        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value),attention_scores

    def forward(self,q,k,v,mask):
        """把输入投影成多头 Q/K/V，计算注意力，再拼回 d_model 维度。"""
        query=self.w_q(q)
        key=self.w_k(k)
        value=self.w_v(v)

        query=query.view(query.shape[0],query.shape[1],self.h,self.d_k).transpose(1,2)
        key=key.view(key.shape[0],key.shape[1],self.h,self.d_k).transpose(1,2)
        value=value.view(value.shape[0],value.shape[1],self.h,self.d_k).transpose(1,2)

        x,self.attention_scores = MultiHeadAttentionBlock.attention(query,key,value,mask,self.dropout)

        x=x.transpose(1,2).contiguous().view(x.shape[0],-1,self.h*self.d_k)

        return self.w_o(x)


class ResidualConnect(nn.Module):
    """残差连接封装：LayerNorm -> 子层 -> Dropout -> 加回原输入。

    用法：
        x = residual(x, lambda x: attention(x, x, x, mask))
    """

    def __init__(self,dropout:float)->None:
        """创建 dropout 和 LayerNormalization。"""
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm=LayerNormalization()

    def forward(self,x,sublayer):
        """执行一个带残差连接的子层。sublayer 是可调用函数或 nn.Module。"""
        return x+self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    """一个 Encoder 层：自注意力 + 前馈网络。

    用法：
        block = EncoderBlock(self_attention, feed_forward, dropout)
        x = block(x, src_mask)
    """

    def __init__(self,self_attention_block:MultiHeadAttentionBlock,feed_forward_block:FeedForwardBlock,dropout:float)->None:
        """组合 self-attention、FFN 和两个残差连接。"""
        super().__init__()
        self.self_attention_block=self_attention_block
        self.feed_forward_block=feed_forward_block
        self.residual_connections=nn.ModuleList([ResidualConnect(dropout) for _ in range(2)])

    def forward(self,x,src_mask):
        """对源语言序列做一次 encoder block 计算。"""
        x=self.residual_connections[0](x,lambda x:self.self_attention_block(x,x,x,src_mask))
        x=self.residual_connections[1](x,self.feed_forward_block)
        return x


class Encoder(nn.Module):
    """由多个 EncoderBlock 堆叠而成的完整 Encoder。

    用法：
        encoder = Encoder(nn.ModuleList(blocks))
        memory = encoder(src_embedding, src_mask)
    """

    def __init__(self,layers:nn.ModuleList)->None:
        """保存 encoder 层列表，并创建最终 LayerNorm。"""
        super().__init__()
        self.layers=layers
        self.norm=LayerNormalization()

    def forward(self,x,mask):
        """依次通过所有 encoder 层，最后做归一化。"""
        for layer in self.layers:
            x=layer(x,mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    """一个 Decoder 层：目标自注意力 + 编码器交叉注意力 + 前馈网络。

    用法：
        block = DecoderBlock(self_attn, cross_attn, ffn, dropout)
        x = block(x, encoder_output, src_mask, tgt_mask)
    """

    def __init__(self,self_attention_block:MultiHeadAttentionBlock,cross_attention_block:MultiHeadAttentionBlock,feed_forward_block:FeedForwardBlock,dropout:float)->None:
        """组合 decoder self-attention、cross-attention、FFN 和三个残差连接。"""
        super().__init__()
        self.self_attention_block=self_attention_block
        self.cross_attention_block=cross_attention_block
        self.feed_forward_block=feed_forward_block
        self.residual_connections=nn.ModuleList([ResidualConnect(dropout) for _ in range(3)])

    def forward(self,x,encoder_output,src_mask,tgt_mask):
        """对目标语言序列做一次 decoder block 计算。"""
        x=self.residual_connections[0](x,lambda x:self.self_attention_block(x,x,x,tgt_mask))
        x=self.residual_connections[1](x,lambda x:self.cross_attention_block(x,encoder_output,encoder_output,src_mask))
        x=self.residual_connections[2](x,self.feed_forward_block)
        return x


class Decoder(nn.Module):
    """由多个 DecoderBlock 堆叠而成的完整 Decoder。

    用法：
        decoder = Decoder(nn.ModuleList(blocks))
        out = decoder(tgt_embedding, encoder_output, src_mask, tgt_mask)
    """

    def __init__(self,layers:nn.ModuleList)->None:
        """保存 decoder 层列表，并创建最终 LayerNorm。"""
        super().__init__()
        self.layers=layers
        self.norm=LayerNormalization()

    def forward(self,x,encoder_output,src_mask,tgt_mask):
        """依次通过所有 decoder 层，最后做归一化。"""
        for layer in self.layers:
            x=layer(x,encoder_output,src_mask,tgt_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    """把 decoder 输出投影到目标词表大小，得到每个 token 的 logits。

    用法：
        logits = projection(decoder_output)

    输出形状通常是 (batch, seq_len, tgt_vocab_size)。
    """

    def __init__(self,d_model:int,vocab_size:int)->None:
        """创建 d_model 到 vocab_size 的线性层。"""
        super().__init__()
        self.proj=nn.Linear(d_model,vocab_size)

    def forward(self,x):
        """返回每个位置对目标词表的未归一化分数。"""
        return self.proj(x)


class Transformer(nn.Module):
    """完整的 Encoder-Decoder Transformer 翻译模型。

    用法：
        model = build_transformer(...)
        encoder_output = model.encode(src, src_mask)
        decoder_output = model.decode(encoder_output, src_mask, tgt, tgt_mask)
        logits = model.project(decoder_output)
    """

    def __init__(self,encoder:Encoder,decoder:Decoder,src_embed:InputEmbeddings,tgt_embed:InputEmbeddings,src_pos:PositionalEncoding,tgt_pos:PositionalEncoding,projection_layer:ProjectionLayer)->None:
        """接收已经组装好的 encoder、decoder、embedding、position 和 projection 层。"""
        super().__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.src_embed=src_embed
        self.tgt_embed=tgt_embed
        self.src_pos=src_pos
        self.tgt_pos=tgt_pos
        self.projection_layer=projection_layer

    def encode(self,src,src_mask):
        """编码源语言 token，返回供 decoder 使用的 encoder_output。"""
        src=self.src_embed(src)
        src=self.src_pos(src)
        return self.encoder(src,src_mask)

    def decode(self,encoder_output,src_mask,tgt,tgt_mask):
        """根据 encoder_output 和已有目标 token，生成 decoder 隐状态。"""
        tgt=self.tgt_embed(tgt)
        tgt=self.tgt_pos(tgt)
        return self.decoder(tgt,encoder_output,src_mask,tgt_mask)

    def project(self,x):
        """把 decoder 隐状态投影成目标词表 logits。"""
        return self.projection_layer(x)


def build_transformer(src_vocab_size:int,tgt_vocab_size:int,src_seq_len:int,tgt_seq_len:int,d_model:int=512,N:int=6,h:int=8,dropout:float=0.1,d_ff:int=2048)->Transformer:
    """构建并初始化完整 Transformer。

    用法：
        model = build_transformer(
            src_vocab_size=tokenizer_src.get_vocab_size(),
            tgt_vocab_size=tokenizer_tgt.get_vocab_size(),
            src_seq_len=256,
            tgt_seq_len=256,
        )

    参数：
        N 是 encoder/decoder 层数，h 是注意力头数，d_ff 是 FFN 隐藏层维度。
    返回值是 Transformer 实例，并对多维参数做 Xavier 初始化。
    """
    src_embed=InputEmbeddings(d_model,src_vocab_size)
    tgt_embed=InputEmbeddings(d_model,tgt_vocab_size)

    src_pos=PositionalEncoding(d_model,src_seq_len,dropout)
    tgt_pos=PositionalEncoding(d_model,tgt_seq_len,dropout)

    encoder_blocks=[]
    for _ in range(N):
        encoder_self_attention_block=MultiHeadAttentionBlock(d_model,h,dropout)
        feed_forward_block=FeedForwardBlock(d_model,d_ff,dropout)
        encoder_block=EncoderBlock(encoder_self_attention_block,feed_forward_block,dropout)
        encoder_blocks.append(encoder_block)

    decoder_blocks=[]
    for _ in range(N):
        decoder_self_attention_block=MultiHeadAttentionBlock(d_model,h,dropout)
        decoder_cross_attention_block=MultiHeadAttentionBlock(d_model,h,dropout)
        feed_forward_block=FeedForwardBlock(d_model,d_ff,dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block,decoder_cross_attention_block,feed_forward_block,dropout)
        decoder_blocks.append(decoder_block)

    encoder=Encoder(nn.ModuleList(encoder_blocks))
    decoder=Decoder(nn.ModuleList(decoder_blocks))

    projection_layer = ProjectionLayer(d_model,tgt_vocab_size)

    transformer=Transformer(encoder,decoder,src_embed,tgt_embed,src_pos,tgt_pos,projection_layer)

    for p in transformer.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform_(p)

    return transformer
