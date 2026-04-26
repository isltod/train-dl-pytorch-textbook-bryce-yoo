import torch
import torch.optim as optim

torch.manual_seed(1)

# 데이터는 열벡터로 넣어야 되나? 아니면 그냥 이 예제가 그런건가...
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])
print("x_train:", x_train)
print("x_train.shape:", x_train.shape)
print("y_train:", y_train)
print("y_train.shape:", y_train.shape)

# 가중치와 편향 초기화 - 이게 조정 대상이니까 requires_grad=True...
W = torch.zeros(1, requires_grad=True)
print("W:", W)
b = torch.zeros(1, requires_grad=True)
print("b:", b)

# 실제로는 여기서 먼저 옵티마이저 설정하고
optimizer = optim.SGD([W, b], lr=0.01)

# 반복으로 최적화한다...
nb_epochs = 2000
for epoch in range(1, nb_epochs + 1):
    # 가설(모델) 세우기
    hypothesis = x_train * W + b

    # 비용 계산
    cost = torch.mean((hypothesis - y_train) ** 2)

    # 경사 하강법으로 모델 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print(
            "Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}".format(
                epoch, nb_epochs, W.item(), b.item(), cost.item()
            )
        )

#
