import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(777)
if device == "cuda":
    torch.cuda.manual_seed_all(777)

# XOR 데이터와 정답
x = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]]).to(device)
y = torch.FloatTensor([[0], [1], [1], [0]]).to(device)

# 이게 단층 퍼셉트론...을 머신러닝으로 구현...
linear = nn.Linear(2, 1, bias=True)
# 원래 단층 퍼셉트론은 계단 함수를 썼지만 여기서는 시그모이드...
sigmoid = nn.Sigmoid()
model = nn.Sequential(linear, sigmoid).to(device)
criterion = nn.BCELoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=1)

for step in range(1001):
    optimizer.zero_grad()
    hypothesis = model(x)
    cost = criterion(hypothesis, y)
    cost.backward()
    optimizer.step()
    if step % 100 == 0:
        # 비용이 0.693...에서 줄어들지를 않음...못 푼다는 말...
        print(step, cost.item())

with torch.no_grad():
    hypothesis = model(x)
    predicted = (hypothesis > 0.5).float()
    # 단층 퍼셉트론으로는 반만 맞춘다...
    accuracy = (predicted == y).float().mean()
    print("\nHypothesis: ", hypothesis.detach().cpu().numpy())
    print("\nPredicted: ", predicted.detach().cpu().numpy())
    print("\nCorrect: ", y.cpu().numpy())
    print("\nAccuracy: ", accuracy.item())

# 다층 퍼셉트론
model = nn.Sequential(
    nn.Linear(2, 10, bias=True),
    nn.Sigmoid(),
    nn.Linear(10, 10, bias=True),
    nn.Sigmoid(),
    nn.Linear(10, 10, bias=True),
    nn.Sigmoid(),
    nn.Linear(10, 1, bias=True),
    nn.Sigmoid(),
).to(device)
criterion = nn.BCELoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=1)

# 대략 3천번 넘어가면서 비용이 1 이하가 된다...
for step in range(10001):
    optimizer.zero_grad()
    hypothesis = model(x)
    cost = criterion(hypothesis, y)
    cost.backward()
    optimizer.step()
    if step % 100 == 0:
        print(step, cost.item())

with torch.no_grad():
    hypothesis = model(x)
    predicted = (hypothesis > 0.5).float()
    accuracy = (predicted == y).float().mean()
    print("\nHypothesis: ", hypothesis.detach().cpu().numpy())
    print("\nPredicted: ", predicted.detach().cpu().numpy())
    print("Correct: ", y.cpu().numpy())
    print("Accuracy: ", accuracy.item())
