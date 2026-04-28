import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(0)

inputs = torch.Tensor(1, 1, 28, 28)
conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
pool = nn.MaxPool2d(kernel_size=2, stride=2)

print(inputs.shape)
out = conv1(inputs)
print(out.shape)
out = pool(out)
print(out.shape)
out = conv2(out)
print(out.shape)
out = pool(out)
print(out.shape)
out = out.view(out.size(0), -1)
print(out.shape)
fc = nn.Linear(3136, 10)
out = fc(out)
print(out.shape)
print(out)

import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init
from torch.utils.data import DataLoader


device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(777)
if device == "cuda":
    torch.cuda.manual_seed_all(777)

learning_rate = 0.001
training_epochs = 15
batch_size = 100

mnist_train = dsets.MNIST(
    root="data/",
    train=True,
    transform=transforms.ToTensor(),
    download=True,
)
mnist_test = dsets.MNIST(
    root="data/",
    train=False,
    transform=transforms.ToTensor(),
    download=True,
)

data_loader = DataLoader(
    dataset=mnist_train, batch_size=batch_size, shuffle=True, drop_last=True
)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            # 입력 채널과 출력 채널만, 이미지 사이즈는 생각 안해도 맞춰준다...
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            # 마찬가지로 위에서 받은 채널과 출력 채널만...
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # 요게 받은 데이터 수 계산해서 맞춰줘야 하는데, 결국 처음부터 중간 과정을 다 알고 있어야 한다...
        self.fc = nn.Linear(7 * 7 * 64, 10, bias=True)
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        # X(?,1,28,28)
        out = self.layer1(x)
        # conv로 1->32, pool로 28->14(?,32,14,14)
        out = self.layer2(out)
        # conv로 32->64, pool로 14->7(?,64,7,7)
        out = out.view(out.size(0), -1)
        # 마지막 (?,10)을 위해 flatten해서 (?,3136)
        out = self.fc(out)
        return out


model = CNN().to(device)
# out = model(inputs.to(device))
# print(out)
# print(out.shape)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

total_batch = len(data_loader)
print("총 배치의 수 : {}".format(total_batch))

for epoch in range(training_epochs):
    avg_cost = 0

    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)

        optimizer.zero_grad()
        hypothesis = model(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print("[Epoch: {:>4}] cost = {:>.9}".format(epoch + 1, avg_cost))

# 테스트
with torch.no_grad():
    # 왜 dataloader로 받을 때는 이런 전처리를 안했는데, 여기선 하는거지?
    X_test = mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)
    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print("Accuracy:", accuracy.item())

#
