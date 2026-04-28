import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

input_str = "apple"
label_str = "pple!"
char_vocab = sorted(list(set(input_str + label_str)))
vocab_size = len(char_vocab)

char_to_index = dict((c, i) for i, c in enumerate(char_vocab))
index_to_char = {}
for key, value in char_to_index.items():
    index_to_char[value] = key

x_data = [char_to_index[c] for c in input_str]
y_data = [char_to_index[c] for c in label_str]

# vocab_size 정방행렬로 identity 행렬을 만들고, 거기서 apple을 숫자로 바꾼 모든 값들을 인덱스로 뽑아서, 배치하면...
x_one_hot = [np.eye(vocab_size)[x] for x in x_data]
# 이것도 리스트[np.array()]들을 np.array([])로 바꿔주기..
x_one_hot = np.array(x_one_hot)
print(x_one_hot)

X = torch.FloatTensor(x_one_hot)
Y = torch.LongTensor(y_data)


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 이렇게 이해하는게 맞나? 일단 output은 이거 같은데...
        # output = input (batch, sequence, input) * Wx (batch, input, out) = (batch, sequence, out)
        # 그게 아니고, RNN은 일단 seq를 분해해야 할 듯...
        # 1. input(batch, sequenc, word)[s0] -> input0(batch, word)
        # 2. _stat0 = input0(batch, word) * Wx(word, hidden) + h(hidden,) * Wh(hidden,hidden) = (batch,hidden)
        # 3. out = stack(_stat0, dim=1) -> (batch, sequence, hidden)에 affine 해서 (batch, sequence, out으로...)
        # 4. _stat = unsqueeze(_stat9) -> (seq9, batch, hidden)
        x, _status = self.rnn(x)
        x = self.fc(x)
        return x


input_size = vocab_size
hidden_size = 5
output_size = 5
learning_rate = 0.1

net = Net(input_size, hidden_size, output_size)
# (batch 2, sequence 6, char 5)를 넣어보자...
out, stat = net(torch.stack((X, X), dim=0))
print(out.shape)
print(stat.shape)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), learning_rate)

for i in range(100):
    optimizer.zero_grad()
    # X는 (6,5)
    outputs = net(X)
    # 이 예제는 배치 차원이 없으므로 결과로 나온 데이터에서 바깥은 제외시키고...
    loss = criterion(outputs.view(-1, input_size), Y.view(-1))
    loss.backward()
    optimizer.step()

    result = outputs.data.numpy().argmax(axis=1)

    result_str = "".join([index_to_char[c] for c in np.squeeze(result)])
    print(
        i,
        "loss: ",
        loss.item(),
        "prediction: ",
        result,
        "true Y: ",
        y_data,
        "prediction str: ",
        result_str,
    )
