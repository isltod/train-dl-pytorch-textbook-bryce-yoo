import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# 이번에는 다중 선형 회귄데...
# 훈련 데이터 준비 - 이렇게 하면 개별 곱셈으로 해야해서 불편하고...행렬로 바꾸려면 데이터베이스 형식을 준용...
x1_train = torch.FloatTensor([[73], [93], [89], [96], [73]])
x2_train = torch.FloatTensor([[80], [88], [91], [98], [66]])
x3_train = torch.FloatTensor([[75], [93], [90], [100], [70]])
# 열 벡터였으니 열 방향으로 묶어주면 가로는 quiz, 세로는 record로 쌓인다...
x_train = torch.cat([x1_train, x2_train, x3_train], dim=1)
print("x_train:", x_train)
print("x_train.shape:", x_train.shape)
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

# 가중치 w와 편향 b 초기화 - 이것도 행렬곱에 맞춰서 벡터로...
w1 = torch.zeros(1, requires_grad=True)
w2 = torch.zeros(1, requires_grad=True)
w3 = torch.zeros(1, requires_grad=True)
# 이렇게 기존 텐서로 만들었더니 can't optimize a non-leaf Tensor 오류 난다...
# W = torch.cat([w1, w2, w3], dim=0)
W = torch.zeros(3, 1, requires_grad=True)
print("w:", W)
print("w.shape:", W.shape)
b = torch.zeros(1, requires_grad=True)

# optimizer 설정...도 행렬로 넣으면 간단해지고...
# optimizer = optim.SGD([w1, w2, w3, b], lr=1e-5)
optimizer = optim.SGD([W, b], lr=1e-5)


nb_epochs = 20
for epoch in range(nb_epochs + 1):
    # 가설(모델) 계산...도 행렬 버전으로 간단하게...
    # hypothesis = x1_train * w1 + x2_train * w2 + x3_train * w3 + b
    hypothesis = x_train.matmul(W) + b

    # 비용 함수 계산
    cost = torch.mean((hypothesis - y_train) ** 2)

    # 경사 하강법으로 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    # if epoch % 100 == 0:
    # 데이터 형태가 행렬로 바뀌어서 프린트 변수들도 달라져야 한다...
    # print(
    #     "Epoch {:4d}/{} w1: {:.3f} w2: {:.3f} w3: {:.3f} b: {:.3f} Cost: {:.6f}".format(
    #         epoch, nb_epochs, w1.item(), w2.item(), w3.item(), b.item(), cost.item()
    #     )
    # )
    print(
        "Epoch {:4d}/{} hypothesis: {} Cost: {:.6f}".format(
            # 모델 값을 프린트할 때도 자동 미분 그래프에서 떼어내야(detach) 하나?
            epoch,
            nb_epochs,
            hypothesis.squeeze().detach(),
            cost.item(),
        )
    )

with torch.no_grad():
    # 학습한 데이터가 (5,3), 3개씩 한 세트로 5개 데이터니까...입력 자료는 (1,3)으로 만들고
    new_input = torch.FloatTensor([[75, 85, 72]])
    prediction = new_input.matmul(W) + b
    # 값 하나도 텐서라서 스칼라로 받으려면 item()
    print("예측값:", prediction.item())
