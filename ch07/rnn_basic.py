import numpy as np

timesteps = 10
input_size = 4
hidden_size = 8

inputs = np.random.random((timesteps, input_size))
hidden_state_t = np.zeros((hidden_size,))

Wx = np.random.random((hidden_size, input_size))
Wh = np.random.random((hidden_size, hidden_size))
b = np.random.random((hidden_size,))

# 이게 곱이 어떻게 돌아가는 거냐...
print("X: {}".format(inputs[0].shape))
print("Wx: {}".format(Wx.shape))
print("곱: {}".format(np.dot(Wx, inputs[0]).shape))


total_hidden_states = []

for input_t in inputs:
    # Wx (8,4), input (4,) -> (8,1), Wh (8,8), h (8,) -> (8,1)
    output_t = np.tanh(np.dot(Wx, input_t) + np.dot(Wh, hidden_state_t) + b)
    total_hidden_states.append(list(output_t))
    hidden_state_t = output_t

# 원래 total_hidden_states는 리스트[리스트] 형태로 stack은 방향 0으로 결합해 차원을 증가시킨다...
# 근데 여기서는 리스트 안에 np.float64 객체인데, ndarray로 묶어서 np.float64을 털어내는 효과인 듯...
total_hidden_states = np.stack(total_hidden_states, axis=0)
print(total_hidden_states)
print(total_hidden_states.shape)
print(type(total_hidden_states))
