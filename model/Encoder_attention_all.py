from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
import torch.nn.functional as F
import torch

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
    r""" 编码器 """
    def __init__(self, cell_type,  # rnn类型
                 input_size,  # 输入维度
                 output_size,  # 输出维度
                 num_layers,  # rnn层数
                 bidirectional=False,  # 是否双向
                 dropout=0.1):  # dropout
        super(Encoder, self).__init__()
        assert cell_type in ['GRU', 'LSTM']  # 限定rnn类型

        if bidirectional:  # 如果双向
            assert output_size % 2 == 0
            cell_size = output_size // 2  # rnn维度
        else:
            cell_size = output_size

        self.bidirectional = bidirectional
        self.cell_type = cell_type
        self.rnn_cell = getattr(nn, cell_type)(input_size=input_size,
                                               hidden_size=cell_size,
                                               num_layers=num_layers,
                                               bidirectional=bidirectional,
                                               dropout=dropout)
        self.attn = Attn('dot', output_size)
        self.concat = nn.Linear(output_size * 2, output_size)
        self.contextsTocontext = nn.Linear(output_size, output_size)

    def forward(self, x,  # [seq, batch, dim]
                length):  # [batch]
        x = pack_padded_sequence(x, length, enforce_sorted=False)

        # output: [seq, batch, dim*directions] 每个时间步的输出
        # final_state = [layers*directions, batch, dim] 每一层的最终状态
        output, final_state = self.rnn_cell(x)
        output = pad_packed_sequence(output)[0]
        contexts = []
        for out in output:
            attn_weights= self.attn(out, output)
            context = attn_weights.bmm(output.transpose(0, 1))
            contexts.append(context.squeeze(1))
        contexts = torch.stack(contexts).sum(0)
        context = self.contextsTocontext(contexts)
        # b = torch.zeros(attn_weights.size()) - 2
        # attn_weights = attn_weights + b.cuda()
        # attn_weights = F.softmax(attn_weights, dim=2)
        # 将注意力权重乘以编码器输出以获得新的“加权和”上下文向量
        # 使用Luong的公式五连接加权上下文向量和GRU输出
        # context = context.squeeze(1) #batch dim
        # print(context.size())

        if self.bidirectional:  # 如果是双向的，对双向进行拼接作为每层的最终状态
            if self.cell_type == 'GRU':
                final_state_forward = final_state[0::2, :, :]  # [layers, batch, dim]
                final_state_back = final_state[1::2, :, :]  # [layers, batch, dim]
                final_state = torch.cat([final_state_forward, final_state_back], 2)  # [layers, batch, dim*2]
            else:
                final_state_h, final_state_c = final_state
                final_state_h = torch.cat([final_state_h[0::2, :, :], final_state_h[1::2, :, :]], 2)
                final_state_c = torch.cat([final_state_c[0::2, :, :], final_state_c[1::2, :, :]], 2)
                final_state = (final_state_h, final_state_c, context)

        # output = [seq, batch, dim]
        # final_state = [layers, batch, dim]
        return output, final_state
