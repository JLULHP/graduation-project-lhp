# 3
# Transformer的编码器、解码器由多个相同的层（layer)组成，此文件用于定义编码器层和解码器层的子层。
import torch
import torch.nn as nn
import torch.nn.functional as F


# 缩放点积注意力，有一个经典的公式
class ScaleDotProductAttention(nn.Module):
    def __init__(self,
                 temperature,  # 缩放因子，用于缩放点积的结果，通常设置为sqrt(d_k)
                 attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        # q/self.temperature 是查询的向量q除以缩放因子，以缩放点积的结果
        # k.transpose(2,3）将键向量k的最后两个维度进行转置
        batch_size, N, n_head, len_q, d_k = q.shape
        attn = torch.matmul(q / self.temperature, k.transpose(-2, -1))
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # 形状: (batch_size, 1, 1, len_q, len_k)
            attn = attn.masked_fill(mask == 0, -1e9)
        # F.softmax(attn, dim=-1)是对注意力分数矩阵的最后一个维度应用softmax，将其转换为注意力权重（概率分布）
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 n_head,  # 注意力头的数量
                 d_model,  # 模型维度
                 d_k,  # 每个头的键和查询的维度
                 d_v,  # 每个头的值的维度
                 dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        # 定义线性变换层，将输入映射到查询、键、值空间
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        # 定义最终的线性变换层，目的是为了将多头注意力的输出映射到原始的维度
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        # 缩放点积注意力
        self.attention = ScaleDotProductAttention(temperature=d_k ** 0.5)
        self.dropout = nn.Dropout(dropout)
        # 层归一化
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self,
                q,
                k,
                v,
                mask=None):
        # 获取输入张量的维度信息
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        batch_size, N, len_q, d_model = q.shape
        _, _, len_k, _ = k.shape  # 获取k的维度信息
        # 保存残差连接的输入
        residual = q
        # 将输入映射到查询、键、值的空间(调用init()里的w_qs、w_ks、w_vs方法)，并重塑为多头形式
        # view()是PyTorch中的一个方法，用于改变张量的形状，但不改变其数据。
        q = self.w_qs(q).view(batch_size, N, len_q, n_head, d_k)
        k = self.w_ks(k).view(batch_size, N, len_k, n_head, d_k)
        v = self.w_vs(v).view(batch_size, N, len_k, n_head, d_v)
        # 转置张量，转置第2和第3个维度(len_q和n_head)，用于缩放点积注意力计算每个头的注意力权重
        q, k, v = q.transpose(2, 3), k.transpose(2, 3), v.transpose(2, 3)
        if mask is not None:
            mask = mask.unsqueeze(1)
        # 计算注意力权重和输出
        q, attn = self.attention.forward(q, k, v, mask=mask)
        # 拼接多头，将注意力的输出拼接回原始形状
        # .contiguous()方法是确保张量在内存中是连续存储的，transpose会改变张量形状而不改变内存布局，可能导致张量在内存的存储不连续
        # view()方法要求张量在内存中是连续的，故使用.contiguous()
        q = q.transpose(2, 3).contiguous().view(batch_size, N, len_q, -1)
        # 通过最终的线性层，并Dropout
        q = self.dropout(self.fc(q))
        # 残差连接
        q += residual
        # 层归一化
        q = self.layer_norm(q)
        # 返回输出和注意力权重
        return q, attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self,
                 d_in,  # 输入维度
                 d_hid,  # 隐藏层维度
                 dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
