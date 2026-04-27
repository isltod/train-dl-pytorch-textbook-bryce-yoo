from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torch.optim as optim


digits = load_digits()
print(digits.images[0])
print(digits.target[0])
print("전체 샘플의 수: {}".format(len(digits.images)))

# images_and_labels = list(zip(digits.images, digits.target))
# for index, (image, label) in enumerate(images_and_labels[:8]):
#     plt.subplot(2, 4, index + 1)
#     plt.axis("off")
#     plt.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
#     plt.title("Training: %i" % label)

# plt.show()

X = digits.data
y = digits.target
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.int64)

model = nn.Sequential(
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 10),
)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

losses = []
for epoch in range(100):
    optimizer.zero_grad()
    y_pred = model(X)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if epoch % 10 == 0:
        print("Epoch: {}, Loss: {:.5f}".format(epoch, loss.item()))


plt.plot(losses)
plt.show()
