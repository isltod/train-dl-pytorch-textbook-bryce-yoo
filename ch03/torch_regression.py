import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 아래에서 nn.Linear 할 때 W, b의 랜덤 값을 고정하려고...
torch.manual_seed(1)

# 데이터
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

# 모델을 선언 및 초기화. 단순 선형 회귀로 입력 1, 출력 1
model = nn.Linear(1, 1)

# 이렇게 만들면 매개변수 W, b는 이미 들어있다...
print(list(model.parameters()))

# 옵티마이저 - 여기서 직접 입력이 아니라 model.parameters()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 반복
nb_epochs = 2000
for epoch in range(nb_epochs + 1):
    # 가설
    prediction = model(x_train)

    # 비용 - 직접 계산이 아니라 mse_loss 이용
    cost = F.mse_loss(prediction, y_train)

    # 경사 하강
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력 - W, b를 0이 아니라 난수로 만들어서인지 좀 더 잘 수렴한다...
    if epoch % 100 == 0:
        print("Epoch {:4d}/{} Cost: {:.6f}".format(epoch, nb_epochs, cost.item()))

# 임의의 입력 4를 선언
new_var = torch.FloatTensor([[4.0]])
pred_y = model(new_var)
print("훈련 후 예측값:", pred_y)

# 학습 후의 W, b
print(list(model.parameters()))

# 다중 선형 회귀
print("다중 선형 회귀------------------------------------------------")
torch.manual_seed(1)

x_train = torch.FloatTensor(
    [[73, 80, 75], [93, 88, 93], [89, 91, 90], [96, 98, 100], [73, 66, 70]]
)
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

# 모델을 선언 및 초기화. 모양이 단순 선형회귀와 거의 같고 입력만 3으로...
model = nn.Linear(3, 1)
print(list(model.parameters()))

# 옵티마이저 - 단순 선형회귀는 0.01인데, 여기서는 발산한다...
optimizer = optim.SGD(model.parameters(), lr=1e-5)

# 나머지는 단순 선형회귀와 같은 코드(print만 제외)...뿐만 아니라, 여러 모델에 대해서도 같은 꼴을 유지한다...
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

# 새 값으로 테스트
new_var = torch.FloatTensor([[73, 80, 75]])
pred_y = model(new_var)
print("훈련 후 예측값:", pred_y)

# 학습 후의 W, b
print(list(model.parameters()))
