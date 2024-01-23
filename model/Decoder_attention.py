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

class Decoder(nn.Module):
    r""" 解码器 """
    def __init__(self, cell_type,  # rnn类型
                 input_size,  # 输入维度
                 output_size,  # 输出维度
                 num_layer,  # rnn层数
                 dropout=0.1):  # dropout
        super(Decoder, self).__init__()
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
                state, encoder_outputs=None):  # 初始状态 [layers*directions, batch, dim]
        # output: [seq, batch, dim*directions] 每个时间步的输出
        # final_state: [layers*directions, batch, dim] 每一层的最终状态
        rnn_output, hidden = self.rnn_cell(x, state)
        # 从当前GRU输出计算注意力
        if encoder_outputs == None:
            return rnn_output, hidden
        attn_weights = self.attn(rnn_output, encoder_outputs)
        b = torch.zeros(attn_weights.size())-2
        attn_weights = attn_weights+b.cuda()
        attn_weights = F.softmax(attn_weights, dim=2)
        # 将注意力权重乘以编码器输出以获得新的“加权和”上下文向量
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # 使用Luong的公式五连接加权上下文向量和GRU输出
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        concat_output = concat_output.unsqueeze(dim=0)
        # 返回输出和在最终隐藏状态
        return concat_output, hidden
