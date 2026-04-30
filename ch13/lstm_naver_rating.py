import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

train_data = pd.read_table("data/ratings_train.txt")
test_data = pd.read_table("data/ratings_test.txt")

print("훈련용 리뷰 개수:", len(train_data))
print("테스트용 리뷰 개수:", len(test_data))

print(train_data[:5])
print(test_data[:5])

# 중복 샘플이 있다?
print(train_data["document"].nunique())
train_data.drop_duplicates(subset=["document"], inplace=True)
# nunique()에는 NULL이 포함되질 않는다...nunique()와 len이 다르면 NULL이 있다는 얘기...
print("중복 제거 후:", len(train_data))

# 긍정 부정 라벨은 비슷한 정도로 있고...
# train_data["label"].value_counts().plot(kind="bar")
# plt.show()
# test_data["label"].value_counts().plot(kind="bar")
# plt.show()

# 라벨이 NULL이 있다?
print(train_data.isnull().values.any())
print(train_data.isnull().sum())
train_data = train_data.dropna(how="any")
print("라벨 NULL 제거 후:", len(train_data))

# 한글과 공백 외에는 제거
# 가-힣 조건은 완성형 한글,
# regex=True는 원래는 replace 안되고 re 써야 하지만 pandas에서는 이 옵션으로 정규식 사용 가능..
train_data["document"] = train_data["document"].str.replace(
    "[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", regex=True
)

# 결과로 공백이나 널만 남은 것들...
# " " 하나 이상으로 시작하는 경우를 ""으로
train_data["document"] = train_data["document"].str.replace("^ +", "", regex=True)
# ""을 nan으로 바꾸고, 그걸 제거
# 책대로 replace(inplace=true)로 하면 ChainedAssignmentError 나오고 바뀌질 않는다...
train_data["document"] = train_data["document"].replace("", np.nan)
print(train_data.isnull().sum())
train_data = train_data.dropna(how="any")
print("한글 외 제거 후:", len(train_data))

# 테스트 데이터도 같은 걸 반복하라는데, 이럴거면 처음부터 함수로 만들지...정말...
# 중복 샘플이 있다?
print(test_data["document"].nunique())
test_data.drop_duplicates(subset=["document"], inplace=True)
# nunique()에는 NULL이 포함되질 않는다...nunique()와 len이 다르면 NULL이 있다는 얘기...
print("중복 제거 후:", len(test_data))

# 긍정 부정 라벨은 비슷한 정도로 있고...
# test_data["label"].value_counts().plot(kind="bar")
# plt.show()
# test_data["label"].value_counts().plot(kind="bar")
# plt.show()

# 라벨이 NULL이 있다?
print(test_data.isnull().values.any())
print(test_data.isnull().sum())
test_data = test_data.dropna(how="any")
print("라벨 NULL 제거 후:", len(test_data))

test_data["document"] = test_data["document"].str.replace(
    "[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", regex=True
)

# 결과로 공백이나 널만 남은 것들...
# " " 하나 이상으로 시작하는 경우를 ""으로
test_data["document"] = test_data["document"].str.replace("^ +", "", regex=True)
# ""을 nan으로 바꾸고, 그걸 제거
# 책대로 replace(inplace=true)로 하면 ChainedAssignmentError 나오고 바뀌질 않는다...
test_data["document"] = test_data["document"].replace("", np.nan)
print(test_data.isnull().sum())
test_data = test_data.dropna(how="any")
print("한글 외 제거 후:", len(test_data))

# 토큰화
stopwords = [
    "도",
    "는",
    "다",
    "의",
    "가",
    "이",
    "은",
    "한",
    "에",
    "하",
    "고",
    "을",
    "를",
    "인",
    "듯",
    "과",
    "와",
    "네",
    "들",
    "듯",
    "지",
    "임",
    "게",
]

from konlpy.tag import Mecab

mecab = Mecab()
