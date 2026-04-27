import torch
import torch.nn as nn

input_size = 4
hidden_size = 3
squence_length = 5
batch_size = 2
# (2,5,4)의 3D 0 텐서
inputs = torch.Tensor(batch_size, squence_length, input_size)
cell = nn.RNN(input_size, hidden_size, batch_first=True)
# rnn_basic 예제에서 total_hidden_states, last hidden_state_t 라고...
outputs, _status = cell(inputs)
# (batch, sequence, hidden) 입력에서 input이 hidden으로만 바뀌었다...
print(outputs.shape)
print(outputs)
# (cell, batch, hidden_size), cell은 num_layers * num_directions?
print(_status.shape)
print(_status)
