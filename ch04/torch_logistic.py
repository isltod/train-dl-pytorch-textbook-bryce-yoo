import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# 이게 뭐든, 이걸로 아래 y가 나오면 되는건데...
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)
print(x_train.shape)
print(y_train.shape)


# 매개변수 부분이 일단 모델로 흡수되고
# model = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())
# 더 토치 스타일로 클래스 만들어서 사용...
class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # x가 (1,2) 형태니까 스칼라 나오려면 W는 (2,1)
        self.linear = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))


model = BinaryClassifier()
print(model(x_train))


# 옵티마이저 매개변수 부분으로 모델로...
optimizer = optim.SGD(model.parameters(), lr=1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):
    # forward 계산을 모델로...
    hypothesis = model(x_train)

    # 손실도 binary_cross_entropy 함수로...
    cost = F.binary_cross_entropy(hypothesis, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    if epoch % 100 == 0:
        # 정확도 값도 리포트
        prediction = hypothesis >= torch.FloatTensor([0.5])
        correct_prediction = prediction.float() == y_train
        accuracy = correct_prediction.sum().item() / len(prediction)
        print(
            "Epoch {:4d}/{} Cost: {:.6f} Accuracy {:2.2f}%".format(
                epoch, nb_epochs, cost.item(), accuracy * 100
            )
        )

# 예측인 no_grad 없이 그냥하네...
hypothesis = model(x_train)
print(hypothesis)
prediction = hypothesis >= torch.FloatTensor([0.5])
print(prediction)
correct_prediction = prediction.float() == y_train
print(correct_prediction)
accuracy = correct_prediction.sum().item() / len(correct_prediction)
print(
    "The model has an accuracy of {:2.2f}% for the training set.".format(accuracy * 100)
)
