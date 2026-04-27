import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from torch import nn
from torch import optim

device = "cuda" if torch.cuda.is_available() else "cpu"

mnist = fetch_openml(
    "mnist_784", data_home="data/", version=1, as_frame=False, cache=True
)

X = mnist.data
y = mnist.target

print(mnist.data.shape)
print(mnist.target.shape)

# plt.imshow(X[0].reshape(28, 28), cmap="gray")
# plt.title(y[0])
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1 / 7, random_state=0
)
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
# fetch_openml로 MNIST를 받으니 라벨이 "0"...astype 안하면 오류...
y_train = torch.LongTensor(y_train.astype(np.int64))
y_test = torch.LongTensor(y_test.astype(np.int64))

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 선언한 후에 add_module 하는 방법도....
model = nn.Sequential()
model.add_module("fc1", nn.Linear(28 * 28 * 1, 100))
model.add_module("relu1", nn.ReLU())
model.add_module("fc2", nn.Linear(100, 100))
model.add_module("relu2", nn.ReLU())
model.add_module("fc3", nn.Linear(100, 10))
model.to(device)


loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10
for epoch in range(epochs):
    for data, targets in train_dataloader:
        data = data.to(device)
        targets = targets.to(device)
        y_pred = model(data)
        loss = loss_fn(y_pred, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

model.eval()
correct = 0
with torch.no_grad():
    for data, targets in test_dataloader:
        data = data.to(device)
        targets = targets.to(device)
        outputs = model(data)
        # torch.max는 dim 주면 argmax, max 둘 반환...predicted는 argmax
        _, predicted = torch.max(outputs.data, 1)
        # 이건 AI가 제안한 코드인데...같은거 같은데...
        correct += (predicted == targets).sum().item()
        # view_as는 predicted와 같은 모양으로 바꾼다?
        # correct += predicted.eq(targets.data.view_as(predicted)).sum()

data_size = len(test_dataset)
test_accuracy = correct / data_size
print(f"Test Accuracy: {test_accuracy}")
