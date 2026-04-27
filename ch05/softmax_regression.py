import torch
import torch.nn.functional as F


torch.manual_seed(1)

x_train = torch.FloatTensor([1, 2, 3])

# 소프트맥스 구할 때 dim을 데이터(샘플 차원)에 맞춰서 지정해야 한다...
hypothesis = F.softmax(x_train, dim=0)
print(hypothesis)

z = torch.rand(3, 5, requires_grad=True)
hypothesis = F.softmax(z, dim=1)
print(hypothesis)

# 원핫 벡터에서는 y가 샘플 뭐가 맞나가 아니라, 특성 뭐가 맞나에 1이 가야 하므로 특성 차원 수에 맞게...
y = torch.randint(5, (3,)).long()
print(y)
y_one_hot = torch.zeros_like(hypothesis)
# unsqueeze로 (3,)->(3,1) 행벡터가 열벡터로,
y_one_hot.scatter_(dim=1, index=y.unsqueeze(1), value=1)
print(y_one_hot)

# sum은 ∑j이므로 특성 방향(1), log(hypothesis)는 log(softmax)이므로 log_softmax 함수 사용...
cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()
print(cost)
# Negative Log Likelyhood 사용할 때는 y one hot이 아니라 그냥 y
print(F.nll_loss(F.log_softmax(z, dim=1), y))
# 더 간단하게 합쳐서...cross_entropy는 소프트맥스를 포함하고 있음에 주의...
print(F.cross_entropy(z, y))
# 하지만 함수가 아니라 nn.CrossEntropyLoss 클래스를 쓴다...

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

# 그래서 W=(4,3)
W = torch.zeros((4, 3), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
optimizer = torch.optim.SGD([W, b], lr=0.1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):
    # sigmoid와 다르게 softmax는 방향을 특성 방향으로 줘야 한다...
    hypothesis = F.softmax(x_train.matmul(W) + b, dim=1)

    # cross entropy 함수를 직접 구현...
    cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print("Epoch {:4d}/{} Cost: {:.6f}".format(epoch, nb_epochs, cost.item()))
