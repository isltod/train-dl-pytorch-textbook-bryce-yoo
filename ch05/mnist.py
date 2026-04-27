import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

# 왜 랜덤시드를 두 가지나...
random.seed(777)
torch.manual_seed(777)
# 약간 귀찮치만 gpu 랜덤시드는 따로 해야 되는 모양...
if device == "cuda":
    torch.cuda.manual_seed_all(777)

# 하이퍼파라미터
training_epochs = 15
batch_size = 100

# 데이터셋
mnist_train = datasets.MNIST(
    root="data/",
    train=True,
    # 0~1 정규화하면서 텐서로 바꾸기
    transform=transforms.ToTensor(),
    download=True,
)
mnist_test = datasets.MNIST(
    root="data/",
    train=False,
    transform=transforms.ToTensor(),
    download=True,
)

# 데이터로더 - drop_last는 배치 마지막에 자투리 처리...
data_loader = DataLoader(
    dataset=mnist_train, batch_size=batch_size, shuffle=True, drop_last=True
)
# dataloader의 len은 총 배치 수 라고...
print(len(data_loader))

# 일단 모델은 직접 만들기로...bias=True는 기본값...
linear = nn.Linear(784, 10, bias=True).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(linear.parameters(), lr=0.1)

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = len(data_loader)
    for X, Y in data_loader:
        # X는 (배치, 28*28)로 Y는 원핫 아니고 0~9
        X = X.view(-1, 28 * 28).to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = linear(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print("Epoch: {:4d}/{} Cost: {:.6f}".format(epoch + 1, training_epochs, avg_cost))
print("Learning finished")

with torch.no_grad():
    # 데이터를 직접 받는 방법이 .test_data, .test_labels...
    X_test = mnist_test.test_data.view(-1, 28 * 28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    # 한 방에 넣으면 몽땅 테스트...행렬이 편하긴 하다...
    prediction = linear(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print("Accuracy:", accuracy.item())

    # 무작위 1개로 예측
    r = random.randint(0, len(mnist_test) - 1)
    # 왜 다시 .test_data에서 받아서 gpu로 보내지? 암튼 X_test에서 건지면 안되는데, 이게 날 데이터이인가?
    X_single_data = mnist_test.test_data[r : r + 1].view(-1, 28 * 28).float().to(device)
    Y_single_data = mnist_test.test_labels[r : r + 1].to(device)

    print("Label: ", Y_single_data.item())
    single_prediction = linear(X_single_data)
    print(
        "Prediction: ",
        torch.argmax(single_prediction, 1).item(),
    )

    plt.imshow(
        mnist_test.test_data[r : r + 1].view(28, 28),
        cmap="Greys",
        interpolation="nearest",
    )
    plt.show()
#
