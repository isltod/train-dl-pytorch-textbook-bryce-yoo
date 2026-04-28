import numpy as np

seq_len = 10
word_dim = 4
hidden_size = 8

sequence = np.random.random((seq_len, word_dim))
hidden_state_t = np.zeros((hidden_size,))

Wx = np.random.random((hidden_size, word_dim))
print(sequence.shape)
print(Wx.shape)
print(sequence[0].shape)
input0 = sequence[0]
print(input0)
# 이건 (8,4) x (4,) -> (8,4) x (4,1) = (8,1) -> (8,)
print(Wx.dot(input0).shape)
# 이건 (4,) x (4,8) -> (1,4) x (4,8) = (1,8) -> (8,)
print(input0.dot(Wx.T).shape)
bb = input0.dot(Wx.T)
print(bb)
# 그러니까 (4,)는 벡터나 행렬이 아니고 필요하면 대충 (4,1) 또는 (1,4)로 본다...
# 그런게 섞여있는 연산 결과가 (1,8)로 나오면 대충 (8,)로 내놓는다...
Wh = np.random.random((hidden_size, hidden_size))
b = np.random.random((hidden_size,))

# 이게 곱이 어떻게 돌아가는 거냐...
print("X: {}".format(sequence[0].shape))
print("Wx: {}".format(Wx.shape))
print("곱: {}".format(np.dot(Wx, sequence[0]).shape))


total_hidden_states = []

for word in sequence:
    # Wx (8,4) x word (4,) -> (8,1) + Wh (8,8), h (8,) -> (8,) => stat (8,)
    # 첫 번째 항은 word를 state로 바꾸고, 두 번째 항은 과거 state를 현재 state로 바꾸고, 더해서 현재 state
    output_t = np.tanh(np.dot(Wx, word) + np.dot(Wh, hidden_state_t) + b)
    # out = total_hidden_states = stack((8,), 10, dim=0) => (10,8)
    # 그걸 out1, out2로 내놓는데, RNN 모듈에서는 행렬로 stack 하니까, [out1, out2, ...]가 된다...
    # 즉 최종 아웃은 (seq, hidden) -> (10,8)이 된다...
    # 그럼 결국 input (batch, sequence, word) x W (batch, word, hidden) -> (batch, sequnce, hidden)
    total_hidden_states.append(list(output_t))
    hidden_state_t = output_t

# 원래 total_hidden_states는 리스트[리스트] 형태로 stack은 방향 0으로 결합해 차원을 증가시킨다...
# 근데 여기서는 리스트 안에 np.float64 객체인데, ndarray로 묶어서 np.float64을 털어내는 효과인 듯...
total_hidden_states = np.stack(total_hidden_states, axis=0)
print(total_hidden_states)
print(total_hidden_states.shape)
print(type(total_hidden_states))
