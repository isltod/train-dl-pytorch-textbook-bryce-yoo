import torch

w = torch.tensor(2.0, requires_grad=True)

# 자동 미분...backward로 호출하고 grad에서 확인
y = w**2
z = 2 * y + 5
z.backward()
print("수식 2w^2 + 5를 w로 미분한 값: ", w.grad)

nb_epochs = 20
for epoch in range(nb_epochs + 1):
    z = 2 * w
    z.backward()
    # zero_grad 없이 backward 호출하면 미분값이 더해져서 누적된다...분기와 새로 시작하는 걸 구분할 수 없으니까...
    print("수식을 w로 미분한 값: ", w.grad)
