import torch
import torch.nn as nn
from torchviz import make_dot

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

    def forward(self, x, h0):
        out, h = self.rnn(x, h0)
        return out, h

# 超参数设置
input_size = 100  # 输入数据编码的维度
hidden_size = 20  # 隐含层维度
num_layers = 4  # 隐含层层数
seq_len = 10  # 序列长度
batch_size = 1

model = SimpleRNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
x = torch.randn(seq_len, batch_size, input_size)  # 输入数据
h0 = torch.zeros(num_layers, batch_size, hidden_size)  # 初始隐藏状态
out, h = model(x, h0)
print("out.shape:",out.shape)
print("h.shape:",h.shape)
dot = make_dot(out, params=dict(model.named_parameters()))
dot.format = 'pdf'
dot.save(filename='rnn_model1')

