# 1
# 定义一个基于Transformer的行人轨迹预测模型。由位置编码、编码器、解码器、全链接层组成了整体的网络架构。
# Transformer的编码器、解码器由多个相同的层（layer)组成，而每个层又有多个子层（sublayer）组成。
# 包括Transformer、PositionalEncoding、Encoder、Decoder、MLP
# 对于每一个类，把所需的属性与方法放到init里，把计算的流程放到forward里
import torch
import torch.nn as nn
import numpy as np
from Transformer.Layers import EncoderLayer, DecoderLayer


class PositionalEncoding(nn.Module):
    def __init__(self,
                 d_hid,  # hidden dimension 隐藏层维度，位置编码的输入向量的维度
                 n_position=80):  
        super(PositionalEncoding, self).__init__()
        # 注册一个缓冲区，用于存储位置编码表，但不是可训练参数，register_buffer是nn.Module的方法，用法是register_buffer(name,tensor)
        # _get_sin_pos_enc_table是私有方法，用于生成位置编码表,sinusoidal_positional_encoding 正弦位置编码格式
        self.register_buffer('pos_table', self._get_sin_pos_enc_table(n_position, d_hid))

    def _get_sin_pos_enc_table(self, n_position, d_hid):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i 偶数维度
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1 奇数维度
        # 将位置编码转换成张量，并在第0维增加一个批次的维度(unsqueeze(0)）
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        # 将位置编码添加到输出张量x
        # self.pos_table[:, :x.size(1)]是将位置编码表中提取与张量x相同长度的部分
        # clone() 确保位置编码表的副本与原始张量独立，防止对位置编码表的修改影响原始数据
        # detach() 由于位置编码表是预计算的，不需要梯度，所以datch()可以避免反向传播的时候更新梯度
        return x + self.pos_table[:, :x.size(1)].clone().detach()

    # # 为了让PositionalEncoding的对象可以像函数一样调用，在类中实现__call__方法
    # def __call__(self, x):
    #     return self.forward(x)


class Encoder(nn.Module):
    def __init__(self,
                 n_features,  # 输入轨迹的特征数量
                 d_feature_vec,  # 每个特征的嵌入维度
                 n_layers,  # 编码器层的数量
                 n_head,  # 多头注意力的头数
                 d_k,  # 每个头的查询和键的维度
                 d_v,  # 每个头的值的维度
                 d_inner,  # 前馈神经网络隐藏层的维度
                 dropout=0.1,  # drop率
                 seq_len=20,  # 输入轨迹序列的长度
                 scale_emb=False):  # 是否特征嵌入进行缩放，缩放用于防止梯度消失、爆炸
        super(Encoder, self).__init__()
        # 特征投影层，将输入的特征向量映射到嵌入维度，不同于翻译
        self.feature_proj = nn.Linear(n_features, d_feature_vec)
        # 位置编码层
        self.position_enc = PositionalEncoding(d_feature_vec, n_position=seq_len)
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        # 编码器层堆栈，包含多个编码器层
        self.layer_stack = nn.ModuleList(
            [EncoderLayer(d_feature_vec, d_inner, n_head, d_k, d_v) for _ in range(n_layers)])
        # 归一化层，对于输入数据的每个时间步进行归一化，eps设置成很小的数防止除0
        self.layer_norm = nn.LayerNorm(d_feature_vec, eps=1e-6)
        # 输出层，将编码器的输出映射到预测的轨迹，不同于翻译
        self.output_proj = nn.Linear(d_feature_vec, n_features)
        self.scale_emb = scale_emb
        # 模型的维度=特征嵌入的维度, model_dimension=编码器维度=解码器维度=嵌入层维度
        self.d_model = d_feature_vec

    def forward(self,
                src_seq,  # 输入轨迹的特征向量，(batch_size,N，T，F),N指交通参与者的数量，T指轨迹序列的长度(20),F指特征的数量
                return_attns=False):  # 是否返回注意力权重，否
        # 初始化注意力权重列表
        enc_sef_attn_list = []
        # 获取输入张量的形状
        batch_size, N, T, F = src_seq.shape
        # 重塑张量的形状
        src_seq = src_seq.view(batch_size * N, T, F)
        # 特征投影，将输入的特征向量经过线性层投影后输出
        enc_output = self.feature_proj(src_seq)
        # 将output乘sqrt(d_model)，用于平衡输入特征的尺度
        if self.scale_emb:
            enc_output *= self.d_model ** 0.5
        enc_output = self.dropout(
            self.position_enc.forward(enc_output))  # 也可以不用forward,但是我不想看见它有波浪线标出
        # 层归一化
        enc_output = self.layer_norm(enc_output)
        # 重塑为（batch_size,N,T,d_model）
        enc_output = enc_output.view(batch_size, N, T, -1)
        for enc_layer in self.layer_stack:
            enc_output, enc_self_attn = enc_layer(enc_output)
            if return_attns:
                enc_sef_attn_list += [enc_self_attn]
        if return_attns:
            return enc_output, enc_sef_attn_list
        else:
            return enc_output


class Decoder(nn.Module):
    def __init__(self,
                 n_features,
                 d_feature_vec,
                 n_layers,
                 n_head,
                 d_k,
                 d_v,
                 d_inner,
                 dropout=0.1,
                 seq_len=80,
                 scale_emb=False):
        super(Decoder, self).__init__()
        # 特征投影层，将输入向量投影到嵌入维度
        self.feature_proj = nn.Linear(n_features, d_feature_vec)
        # 位置编码层
        self.position_enc = PositionalEncoding(d_feature_vec, n_position=seq_len)
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        # 解码器堆栈
        self.layer_stack = nn.ModuleList(
            [DecoderLayer(d_feature_vec, d_inner, n_head, d_k, d_v, dropout) for _ in range(n_layers)])
        # 层归一化
        self.layer_norm = nn.LayerNorm(d_feature_vec, eps=1e-6)
        self.output_proj = nn.Linear(d_feature_vec, n_features)
        self.scale_emb = scale_emb
        # 模型的维度=特征嵌入的维度
        self.d_model = d_feature_vec

    def forward(self,
                target_seq,  # (batch_size,N,T,F)
                enc_output,  # 编码器的输出(batch_size,N,T,d_model)
                return_attns=False):
        # 初始化解码器自注意力权重列表，初始化编码器-解码器注意力权重列表
        dec_self_attn_list, dec_enc_attn_list = [], []
        # 获取输入张量的形状
        batch_size, N, T, F = target_seq.shape
        # 重塑为 (batch_size * N, T, F)
        target_seq = target_seq.view(batch_size * N, T, F)
        # 特征投影，将输入的向量经过线性层投影后输出
        dec_output = self.feature_proj(target_seq)
        if self.scale_emb:
            dec_output *= self.d_model ** 0.5
        # 位置编码、dropout
        dec_output = self.dropout(self.position_enc.forward(dec_output))
        # 层归一化
        dec_output = self.layer_norm(dec_output)
        # 重塑为 (batch_size, N, T, d_model)
        dec_output = dec_output.view(batch_size, N, T, -1)
        # 确保编码器输出的维度正确（后加的）
        enc_output = enc_output.view(batch_size, N, -1, self.d_model)
        # 编码器堆栈
        for dec_layer in self.layer_stack:
            dec_output, dec_self_attn, dec_enc_attn = dec_layer(dec_output, enc_output)
            if return_attns:
                dec_self_attn_list += [dec_self_attn]
                dec_enc_attn_list += [dec_enc_attn]
        output = self.output_proj(dec_output)
        if return_attns:
            return dec_output, dec_self_attn_list, dec_enc_attn_list
        else:
            return dec_output


class Transformer(nn.Module):
    def __init__(self,
                 n_features,
                 d_feature_vec,
                 n_layers,
                 n_head,
                 d_k,
                 d_v,
                 d_inner,
                 dropout=0.1,
                 obs_len=20,  # 观测序列长度
                 pred_len=80,  # 预测序列长度
                 scale_emb=False):
        super(Transformer, self).__init__()
        # 编码器层
        self.encoder = Encoder(n_features=n_features,
                               d_feature_vec=d_feature_vec,
                               n_layers=n_layers,
                               n_head=n_head,
                               d_k=d_k,
                               d_v=d_v,
                               d_inner=d_inner,
                               dropout=dropout,
                               seq_len=obs_len,
                               scale_emb=scale_emb)
        # 解码器层
        self.decoder = Decoder(n_features=n_features,
                               d_feature_vec=d_feature_vec,
                               n_layers=n_layers,
                               n_head=n_head,
                               d_k=d_k,
                               d_v=d_v,
                               d_inner=d_inner,
                               dropout=dropout,
                               seq_len=pred_len,
                               scale_emb=scale_emb)
        # 输出层
        self.output_proj = nn.Linear(d_feature_vec, n_features)
        self.d_feature_vec = d_feature_vec  # 保存特征向量维度
        self.n_features = n_features  # 保存特征数量

    def forward(self, src_seq, tgt_seq, mask=None):
        batch_size, num_agents = src_seq.size(0), src_seq.size(1)
        # 处理掩码
        if mask is not None:
            # 创建注意力掩码
            attn_mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
            # 扩展掩码以匹配注意力权重的维度
            attn_mask = attn_mask & attn_mask.transpose(-2, -1)  # (B, 1, N, N)
        else:
            attn_mask = None
        # 编码器
        enc_output = self.encoder.forward(src_seq)
        # 解码器
        dec_output = self.decoder.forward(tgt_seq, enc_output)
        # 重塑解码器输出以适应输出投影层
        batch_size, N, seq_len, _ = dec_output.shape
        dec_output = dec_output.view(batch_size * N * seq_len, self.d_feature_vec)
        # 输出投影
        output = self.output_proj(dec_output)
        # 重塑为原始维度
        output = output.view(batch_size, N, seq_len, self.n_features)
        return output
