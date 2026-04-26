import numpy as np

print("넘파이----------------------------------------")
t = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
print(t)
# 이게 텐서 차원
print("Rank of t: ", t.ndim)
# shape는 size, (7,)은 (1,7)
print("Shape of t: ", t.shape)

t = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
print(t)
print("Rank of t: ", t.ndim)
print("Shape of t: ", t.shape)
# 넘파이 size는 메서드 아니라 속성으로 호출하고 len와 같은 값을 주는 듯...
print("Size of t:", t.size)

# PyTorch에서는..
print("토치----------------------------------------")
import torch

t = torch.FloatTensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
print(t)
print(t.dim())  # rank. 즉, 차원
# 토치 size는 메서드 호출 형식이고, size와 shape는 같은 결과가...
print(t.shape)  # shape
print(t.size())  # shape

# 인덱싱이나 슬라이싱은 일단 넘파이와 같다..
print(t[0], t[1], t[-1])  # 인덱스
print(t[2:5], t[4:-1])  # 슬라이싱
print(t[:2], t[3:])  # 슬라이싱

# 2차원 텐서
t = torch.FloatTensor(
    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
)
print(t)
# 넘파이와 달리 ndim이 아니라 dim
print(t.dim())
print(t.shape)
print(t.size())
print(t[:, 1])
print(t[:, 1].size())
print(t[:, :-1])
print(t[:, :-1].size())

# 브로드캐스팅
print("브로드캐스팅--------------------------------")
# 벡터 + 벡터...당연히 되고...
m1 = torch.FloatTensor([[3, 3]])
m2 = torch.FloatTensor([[2, 2]])
print(m1 + m2)

# Vector + scalar -> 브로드캐스팅
m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([3])  # [3] -> [3, 3]
print(m1 + m2)
# 그냥 텐서에 스칼라를 더해도 브로드캐스팅...
print(m1 + 3)

# 2 x 1 Vector + 1 x 2 Vector
# 일단 벡터 뭉치를 깨지 않는 방향으로 브로드캐스팅...
# [1,2]가 열 방향으로 늘어나 (2,2)가 되고
m1 = torch.FloatTensor([[1, 2]])
# [3,4].T가 행 방향으로 늘어나 (2,2)가 되고...헷갈린다..
m2 = torch.FloatTensor([[3], [4]])
print(m1 + m2)

# 자주 쓰는 기능
print("자주 쓰는 기능--------------------------------")
# 행렬 곱셈
m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1], [2]])
print("Shape of Matrix 1: ", m1.shape)  # 2 x 2
print("Shape of Matrix 2: ", m2.shape)  # 2 x 1
print("행렬곱: matmul")
print(m1.matmul(m2))  # 2 x 1

# element-wise 곱셈
print("element-wise 곱셈: mul, *")
print(m1 * m2)  # 2 x 2
print(m1.mul(m2))

# 평균
t = torch.FloatTensor([1, 2])
print(t.mean())
print(m1)
print("2차원 텐서 평균(축 없이):", m1.mean())
# 둘 다 (1,2) 모양의 텐서가 나오고, dim 방향으로 평균낸다..
print("2차원 텐서 평균(행 방향):", m1.mean(dim=0))
print("2차원 텐서 평균(열 방향):", m1.mean(dim=1))
# dim= 없이 해도 되고, -1은 마지막 축, 즉 열방향...
print("2차원 텐서 평균(-1 방향):", m1.mean(-1))

# 맥스와 아그맥스
print("맥스와 아그맥스--------------------------------")
print(m1)
print("맥스:", m1.max())
# 3 텐서? 원소를 인덱스 0부터 세서 마지막 4는 인덱스 3이라는 건가? 순서는?
print("아그맥스:", m1.argmax())
# max에 dim을 주면 torch.return_types.max라는 걸 반환하고, values(맥스)와 indices(아그맥스) 속성으로 들어있다...
print("행 방향 맥스:", m1.max(dim=0))
print("행 방향 [0]:", m1.max(dim=0)[0])
print("행 방향 .values:", m1.max(dim=0).values)

# 뷰
print("뷰----------------------------------------------")
t = np.array([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]])
ft = torch.FloatTensor(t)
print(ft.shape)
print(ft)

# 2차원 텐서로
print(ft.view([-1, 3]))
print(ft.view([-1, 3]).shape)

# 3차원 텐서로
print(ft.view([-1, 1, 3]))
print(ft.view([-1, 1, 3]).shape)

# 스퀴즈
print("스퀴즈 언스퀴즈----------------------------------------------")
ft = torch.FloatTensor([[0], [1], [2]])
print(ft)
print(ft.shape)
print(ft.squeeze())
print(ft.squeeze().shape)
ft = torch.FloatTensor([0, 1, 2])
print(ft.shape)
print(ft.unsqueeze(0))
print(ft.unsqueeze(0).shape)
print(ft.view(1, -1))
print(ft.view(1, -1).shape)
print(ft.unsqueeze(1))
print(ft.unsqueeze(1).shape)
print(ft.unsqueeze(-1))
print(ft.unsqueeze(-1).shape)
print(ft.view(-1, 1))
print(ft.view(-1, 1).shape)

# 타입 캐스팅
print("타입 캐스팅----------------------------------------------")
# 만들때는 torch.타입() 형식이고...
lt = torch.LongTensor([1, 2, 3, 4])
print(lt)
# 변환은 간단하게 메서드 호출
print(lt.float())
# 롱 타입은 그냥 숫자만 나오는데 바이트 타입은 dtype 속성이 나오네...torch.uint8이라고...
bt = torch.ByteTensor([True, False, False, True])
print(bt)
print(bt.long())
print(bt.float())

# 연결
print("연결----------------------------------------------")
x = torch.FloatTensor([[1, 2], [3, 4]])
y = torch.FloatTensor([[5, 6], [7, 8]])
print("0축 연결:", torch.cat([x, y], dim=0))
print("1축 연결:", torch.cat([x, y], dim=1))

# 스택킹
x = torch.FloatTensor([1, 4])
y = torch.FloatTensor([2, 5])
z = torch.FloatTensor([3, 6])
# 스택킹은 이렇게 나열하면 되는데...
print("스택킹:", torch.stack([x, y, z]))
# 같은 효과를 내는 cat을 사용하면 언스퀴즈를 먼저하고 이어붙이는 꼴이다...
print("스택킹 cat:", torch.cat([x.unsqueeze(0), y.unsqueeze(0), z.unsqueeze(0)], dim=0))
print("스택킹 dim=1:", torch.stack([x, y, z], dim=1))

# 인플레이스 연산(이게 덮어쓰기라고 해야되나...)
print("인플레이스 연산----------------------------------------------")
x = torch.FloatTensor([[1, 2], [3, 4]])
print("mul 2:", x.mul(2.0))
print("원본:", x)
print("mul_ 2:", x.mul_(2.0))
print("원본:", x)
