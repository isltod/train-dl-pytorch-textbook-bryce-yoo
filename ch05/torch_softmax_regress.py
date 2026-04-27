import torch
import torch.nn as nn
import torch.nn.functional as F


torch.manual_seed(1)

x_train = torch.FloatTensor(
    [
        [1, 2, 1, 1],
        [2, 1, 3, 2],
        [3, 1, 3, 4],
        [4, 1, 5, 5],
        [1, 7, 5, 5],
        [1, 2, 5, 6],
        [1, 6, 6, 6],
        [1, 7, 7, 7],
    ]
)
# 이대로 (8,4) -> (8,)을 위해 W=(4,1)이 아니라, y가 (8,3)으로 봐야 한다...
y_train = torch.LongTensor([2, 2, 2, 1, 1, 1, 0, 0])
y_one_hot = torch.zeros(8, 3)
y_one_hot.scatter_(dim=1, index=y_train.unsqueeze(1), value=1)
print(y_one_hot)

# # 그래서 W=(4,3)
# W = torch.zeros((4, 3), requires_grad=True)
# b = torch.zeros(1, requires_grad=True)
# nn 사용하면...매개변수 필요 없고...softmax는 cross_entropy에 있으니 linear만...
# model = torch.nn.Linear(4, 3)


# 더 torch 스타일로
class SoftmaxClassifierModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 3)

    def forward(self, x):
        return self.linear(x)


model = SoftmaxClassifierModel()

# optimizer도 model로...
# optimizer = torch.optim.SGD([W, b], lr=0.1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
# 비용도 클래스로...
criterion = nn.CrossEntropyLoss()

nb_epochs = 1000
for epoch in range(nb_epochs + 1):
    # 비용을 cross_entropy 함수로 하면 softmax도 거기에서...
    hypothesis = model(x_train)

    # cross_entropy 함수 이용 - softmax 포함...
    # cost = F.cross_entropy(hypothesis, y_train)
    # 더 토치 스타일로 클래스 이용...좀 더 간결한 느낌...
    cost = criterion(hypothesis, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print("Epoch {:4d}/{} Cost: {:.6f}".format(epoch, nb_epochs, cost.item()))
