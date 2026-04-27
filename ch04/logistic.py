import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# W는 경사도
x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(0.5 * x)
y2 = sigmoid(x)
y3 = sigmoid(2 * x)

plt.plot(x, y1, "r", linestyle="--")
plt.plot(x, y2, "b")
plt.plot(x, y3, "g", linestyle="--")
plt.plot([0, 0], [1.0, 0.0], ":")
plt.show()

# b는 좌우 이동
y1 = sigmoid(x + 0.5)
y2 = sigmoid(x)
y3 = sigmoid(x - 0.5)

plt.plot(x, y1, "r", linestyle="--")
plt.plot(x, y2, "b")
plt.plot(x, y3, "g", linestyle="--")
plt.plot([0, 0], [1.0, 0.0], ":")
plt.show()


# 이 W, b를 학습해서 꺽이는 부분을 맞추기...


# 손실 함수는 아래 두 개를 합치기...
def loss1(x):
    return -np.log(sigmoid(x))


def loss0(x):
    return -np.log(1 - sigmoid(x))


l1 = loss1(x)
l2 = loss0(x)

plt.plot(x, l1, "r")
plt.plot(x, l2, "b")
plt.show()
