import torch
import torch.optim as optim

torch.manual_seed(1)

# 이게 뭐든, 이걸로 아래 y가 나오면 되는건데...
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)
print(x_train.shape)
print(y_train.shape)

# x가 (1,2) 형태니까 스칼라 나오려면 W는 (2,1)
W = torch.zeros((2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
print(W)
print(b)

# 학습률을 1? 이래도 되나?
optimizer = optim.SGD([W, b], lr=1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):
    # sigmoid는 functional이 아니라 그냥 torch에 있네...
    hypothesis = torch.sigmoid(x_train.matmul(W) + b)
    # binary_cross_entropy 함수 쓰지 않고 그냥 계산하면...
    cost = -(
        y_train * torch.log(hypothesis) + (1 - y_train) * torch.log(1 - hypothesis)
    ).mean()
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print("Epoch {:4d}/{} Cost: {:.6f}".format(epoch, nb_epochs, cost.item()))

# 예측인 no_grad 없이 그냥하네...
hypothesis = torch.sigmoid(x_train.matmul(W) + b)
print(hypothesis)
prediction = hypothesis >= torch.FloatTensor([0.5])
print(prediction)
correct_prediction = prediction.float() == y_train
print(correct_prediction)
accuracy = correct_prediction.sum().item() / len(correct_prediction)
print(
    "The model has an accuracy of {:2.2f}% for the training set.".format(accuracy * 100)
)
