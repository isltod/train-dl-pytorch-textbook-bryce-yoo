import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

x_train = torch.FloatTensor(
    [[73, 80, 75], [93, 88, 93], [89, 91, 90], [96, 98, 100], [73, 66, 70]]
)
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])
dataset = TensorDataset(x_train, y_train)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

model = nn.Linear(3, 1)
optimizer = optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 20
for epoch in range(nb_epochs + 1):
    for batch_idx, samples in enumerate(dataloader):
        # 5개를 배치 2로 돌리니 인덱스는 0,1,2
        print(batch_idx)
        # 당연히, 셔플 켰으니 데이터는 순서대로 나오질 않는다...
        print(samples)
        x_train, y_train = samples
        prediction = model(x_train)
        cost = F.mse_loss(prediction, y_train)
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        print(
            "Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}".format(
                epoch, nb_epochs, batch_idx + 1, len(dataloader), cost.item()
            )
        )

new_var = torch.FloatTensor([[73, 80, 75]])
pred_y = model(new_var)
print("훈련 후 예측값:", pred_y)

# 학습 후의 W, b
print(list(model.parameters()))

# 커스텀 데이터셑
print("커스텀 데이터셋------------------------------------------------")


class CustomDataset(Dataset):
    def __init__(self):
        # 이건 입력으로 받아서 할당할테고...
        self.x_data = [
            [73, 80, 75],
            [93, 88, 93],
            [89, 91, 90],
            [96, 98, 100],
            [73, 66, 70],
        ]
        self.y_data = [[152], [185], [180], [196], [142]]

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        # 인덱싱에 x, y를 다 줘야 하고, 그것도 텐서로...
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x, y


dataset = CustomDataset()
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

model = torch.nn.Linear(3, 1)
# 책대로 lr=1e-5 했더니 Cost가 들쑥날쑥, 그래서 lr=1e-6, epoch=20으로 했더니 수렴하는 모양...
optimizer = optim.SGD(model.parameters(), lr=1e-6)

nb_epochs = 40
for epoch in range(nb_epochs + 1):
    for batch_idx, samples in enumerate(dataloader):
        x_train, y_train = samples
        prediction = model(x_train)
        cost = F.mse_loss(prediction, y_train)
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        print(
            "Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}".format(
                epoch, nb_epochs, batch_idx + 1, len(dataloader), cost.item()
            )
        )
