import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Attn(torch.nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # 根据给定的方法计算注意力（能量）
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


class Encoder(nn.Module):
    r""" 解码器 """

    def __init__(self, cell_type,  # rnn类型
                 input_size,  # 输入维度
                 output_size,  # 输出维度
                 num_layer,  # rnn层数
                 bi_direction,
                 dropout=0.1):  # dropout
        super(Encoder, self).__init__()
        assert cell_type in ['GRU', 'LSTM']  # 限定rnn类型

        self.cell_type = cell_type
        self.rnn_cell = getattr(nn, cell_type)(
            input_size=input_size,
            hidden_size=output_size,
            num_layers=num_layer,
            dropout=dropout)
        self.attn = Attn('concat', output_size)
        self.concat = nn.Linear(output_size * 2, output_size)

    def forward(self, x,  # 输入 [seq, batch, dim] 或者单步输入 [1, batch, dim]
                len_encoder=None, state=None, encoder_outputs=None):  # 初始状态 [layers*directions, batch, dim]
        # output: [seq, batch, dim*directions] 每个时间步的输出
        # final_state: [layers*directions, batch, dim] 每一层的最终状态
        # 从当前GRU输出计算注意力
        if encoder_outputs == None:
            x = pack_padded_sequence(x, len_encoder, enforce_sorted=False)

            # output: [seq, batch, dim*directions] 每个时间步的输出
            # final_state = [layers*directions, batch, dim] 每一层的最终状态
            output, final_state = self.rnn_cell(x)
            output = pad_packed_sequence(output)[0]
            # output = [seq, batch, dim]
            # final_state = [layers, batch, dim]
            return output, final_state
        else:
            rnn_output, hidden = self.rnn_cell(x, state)
            encoder_outputs = torch.cat((encoder_outputs, rnn_output),0)
            att_weights_h_0 = self.attn(hidden[0][0], encoder_outputs)
            # 将注意力权重乘以编码器输出以获得新的“加权和”上下文向量
            context_h_0 = att_weights_h_0.bmm(encoder_outputs.transpose(0, 1))
            att_weights_h_1 = self.attn(hidden[0][1], encoder_outputs)
            # 将注意力权重乘以编码器输出以获得新的“加权和”上下文向量
            context_h_1 = att_weights_h_1.bmm(encoder_outputs.transpose(0, 1))
            att_weights_c_0 = self.attn(hidden[1][0], encoder_outputs)
            # 将注意力权重乘以编码器输出以获得新的“加权和”上下文向量
            context_c_0 = att_weights_c_0.bmm(encoder_outputs.transpose(0, 1))
            att_weights_c_1 = self.attn(hidden[1][1], encoder_outputs)
            # 将注意力权重乘以编码器输出以获得新的“加权和”上下文向量
            context_c_1 = att_weights_c_1.bmm(encoder_outputs.transpose(0, 1))
            # 返回输出和在最终隐藏状态
            return rnn_output, (torch.cat((context_h_0, context_h_1), 0),
                                torch.cat((context_c_0, context_c_1), 0)),encoder_outputs
