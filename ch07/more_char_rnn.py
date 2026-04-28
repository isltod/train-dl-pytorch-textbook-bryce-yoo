import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

torch.manual_seed(0)

sentence = (
    "if you want to build a ship, don't drum up people together to "
    "collect wood and don't assign them tasks and work, but rather "
    "teach them to long for the endless immensity of the sea."
)

char_set = list(set(sentence))
char_dic = {c: i for i, c in enumerate(char_set)}
dic_size = len(char_dic)
hidden_size = dic_size
sequence_length = 10
learning_rate = 0.1

# sequence_length 만큼씩 끊어서 문장 만들고, 그 한 칸 뒤 문장을 정답으로 데이터셋 만들기...
x_data = []
y_data = []
for i in range(0, len(sentence) - sequence_length):
    x_str = sentence[i : i + sequence_length]
    y_str = sentence[i + 1 : i + sequence_length + 1]
    # print(i, x_str, "->", y_str)

    x_data.append([char_dic[c] for c in x_str])
    y_data.append([char_dic[c] for c in y_str])

# 이건 char_rnn에서 했던 원핫 인코딩을 각 문장에 대해서...
x_one_hot = [np.eye(dic_size)[x] for x in x_data]

# 원핫 벡터를 왜 FloatTensor로? 가중치 곱할 때 정수면 문제가 되나?
X = torch.FloatTensor(x_one_hot)
Y = torch.LongTensor(y_data)


class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers):
        super(Net, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers=layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x):
        x, _status = self.rnn(x)
        x = self.fc(x)
        return x


net = Net(dic_size, hidden_size, 2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), learning_rate)

for i in range(100):
    optimizer.zero_grad()
    # 170개 문장을 통으로 넣으니 결과는 (배치 179, 문장 길이 10, 단어 차원 25)
    outputs = net(X)
    loss = criterion(outputs.view(-1, dic_size), Y.view(-1))
    loss.backward()
    optimizer.step()

    result = outputs.argmax(axis=2)
    result_str = ""
    # 처음에는 모든 글자, 그 뒤로는 마지막 글자만 추려서...
    for j, result in enumerate(result):
        if j == 0:
            result_str += "".join([char_set[t] for t in result])
        else:
            result_str += char_set[result[-1]]
    print(
        i,
        "loss: ",
        loss.item(),
        "prediction str: ",
        result_str,
    )
