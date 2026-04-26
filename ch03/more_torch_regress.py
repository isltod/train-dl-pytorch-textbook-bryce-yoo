import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 더 토치 스타일의 클래스로 선형회귀
torch.manual_seed(1)
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])


# 토치 스타일은 직접 nn 모듈을 쓰는게 아니라, 이렇게 미리 클래스로 만들어놓고 사용한다...
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


model = LinearRegressionModel()

# 옵티마이저
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 반복
nb_epochs = 2000
for epoch in range(nb_epochs + 1):
    # 가설
    prediction = model(x_train)
    # 비용
    cost = F.mse_loss(prediction, y_train)
    # 최적화
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print("Epoch {:4d}/{} Cost: {:.6f}".format(epoch, nb_epochs, cost.item()))

# 다중 선형 회귀를 클래스로...
print("다중 선형 회귀------------------------------------------------")
torch.manual_seed(1)

x_train = torch.FloatTensor(
    [[73, 80, 75], [93, 88, 93], [89, 91, 90], [96, 98, 100], [73, 66, 70]]
)
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])


class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)


model = MultivariateLinearRegressionModel()
optimizer = optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 2000
for epoch in range(nb_epochs + 1):
    # 가설
    prediction = model(x_train)
    # 비용
    cost = F.mse_loss(prediction, y_train)
    # 최적화
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print("Epoch {:4d}/{} Cost: {:.6f}".format(epoch, nb_epochs, cost.item()))
