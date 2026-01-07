# 2
# Transformer的编码器、解码器由多个相同的层（layer)组成，此文件用于定义编码器层和解码器层
import torch.nn as nn
from Transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward
import torch.nn.functional as F


class EncoderLayer(nn.Module):
    def __init__(self,
                 d_model,  # 模型维度
                 d_inner,  # 前馈神经网络层的隐藏层的维度
                 n_head,  # 多头注意力的头数
                 d_k,  # query和key的维度
                 d_v,  # value的维度
                 dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)  # 多头注意力
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)  # 前馈神经网络
        self.layer_norm1 = nn.LayerNorm(d_model, eps=1e-6)  # 归一化层1
        self.layer_norm2 = nn.LayerNorm(d_model, eps=1e-6)  # 归一化层2
        self.dropout = nn.Dropout(dropout)  # dropout层

    def forward(self,
                enc_input,  # 输入张量，形状(batch_size,20,d_model)
                self_attn_mask=None):  # 自注意力掩码，对于轨迹预测而言并不需要
        # 编码器的自注意力
        enc_output, enc_self_attn = self.self_attn.forward(enc_input, enc_input, enc_input, mask=self_attn_mask)
        enc_output = self.layer_norm1(enc_input + self.dropout(enc_output))  # 残差链接与层归一化
        # 保存中间结果用于残差连接
        residual = enc_output
        # 前馈神经网络
        ffn_output = self.pos_ffn.forward(enc_output)
        enc_output = self.layer_norm2(residual + self.dropout(ffn_output))  
        return enc_output, enc_self_attn


class DecoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 d_inner,
                 n_head,
                 d_k,
                 d_v,
                 dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.layer_norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm3 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                dec_input,
                enc_output,
                self_attn_mask=None,
                dec_enc_attn_mask=None):
        # 解码器的自注意力
        dec_output, dec_self_attn = (self.self_attn.forward(dec_input, dec_input, dec_input, mask=self_attn_mask))
        dec_output = self.layer_norm1(dec_input + self.dropout(dec_output))
        # 保存中间结果用于残差连接
        residual = dec_output
        # 编码-解码注意力
        dec_enc_output, dec_enc_attn = self.enc_attn.forward(dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output = self.layer_norm2(residual + self.dropout(dec_enc_output))  # 修复残差连接
        # 保存中间结果用于残差连接
        residual = dec_output
        # 前馈神经网络
        ffn_output = self.pos_ffn.forward(dec_output)
        dec_output = self.layer_norm3(residual + self.dropout(ffn_output))  # 修复残差连接
        return dec_output, dec_self_attn, dec_enc_attn