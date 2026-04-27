import torch
import torch.nn as nn

word_dim = 4
hidden_dim = 5
squence_length = 6
batch_size = 7

# (2,5,4)의 3D 0 텐서
inputs = torch.Tensor(batch_size, squence_length, word_dim)

# 이전 예제와 다르게 num_layers=3
cell = nn.RNN(word_dim, hidden_dim, batch_first=True, num_layers=3)

# rnn_basic 예제에서 total_hidden_states, last hidden_state_t 라고...
outputs, _status = cell(inputs)

# (batch, sequence, hidden) 입력에서 input이 hidden으로만 바뀌었다...
print("total_hidden_states:", outputs.shape)
# print(outputs)

# (cell, batch, hidden_size), cell은 num_layers * num_directions?
print("last hidden_state:", _status.shape)
# print(_status)
