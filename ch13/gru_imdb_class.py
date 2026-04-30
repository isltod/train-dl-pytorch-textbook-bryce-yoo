import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import torch
from collections import Counter
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from tqdm import tqdm

df = pd.read_csv("data/IMDB Dataset.csv")
print(df.head())
print(df.shape)

# 판다스 결측값 확인 방법 3가지
print(df.isnull().values.any())
print(df.isnull().sum())
print(df.info())

# # 라벨 바차트 확인...비슷하게 라벨링...
# df["sentiment"].value_counts().plot(kind="bar")
# plt.show()

# 훈련, 검증, 시험 데이터 나누기
X_data = df["review"]
y_data = df["sentiment"]
print("영화 리뷰의 갯수:", len(X_data))
print("영화 라벨의 갯수:", len(y_data))
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_data, y_data, test_size=0.5, random_state=0, stratify=y_data
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.2, random_state=0, stratify=y_train_val
)
print(X_train.shape, X_val.shape, X_test.shape)
print(y_train.shape, y_val.shape, y_test.shape)


def print_label_ratio(data):
    print(f"부정 리뷰 = {round(data.value_counts()['positive'] / len(data) * 100, 3)}%")
    print(f"긍정 리뷰 = {round(data.value_counts()['negative'] / len(data) * 100, 3)}%")


print("훈련 데이터 비율--------------")
print_label_ratio(y_train)
print("검증 데이터 비율--------------")
print_label_ratio(y_val)
print("시험 데이터 비율--------------")
print_label_ratio(y_test)


# 토큰화
def tokenize(data):
    tokenized_data = []
    for sentence in tqdm(data):
        tokenized_data.append(word_tokenize(sentence.lower()))
    return tokenized_data


X_train_tokenized = tokenize(X_train)
X_val_tokenized = tokenize(X_val)
X_test_tokenized = tokenize(X_test)

# vocab 만들기
word_list = []
for sentence in X_train_tokenized:
    for word in sentence:
        word_list.append(word)

word_counter = Counter(word_list)
print("총 단어 수", len(word_counter))
print("훈련 데이터에서 단어 the의 등장 횟수:", word_counter["the"])
print("훈련 데이터에서 단어 love의 등장 횟수:", word_counter["love"])
vocab = sorted(word_counter, key=word_counter.get, reverse=True)
print("등장 빈도수 상위 10개 단어")
print(vocab[:10])

# 등장 빈도수 3 이하 단어들 제외
threshold = 3
# 총 단어 수
total_cnt = len(word_counter)
# 등장 빈도수가 threshold보다 적은 단어들 개수
rare_cnt = 0
# 훈련 데이터 전체 단어 빈도수의 합
total_freq = 0
# 등장 빈도수가 threshold보다 적은 단어들의 빈도수 합
rare_freq = 0

for word, cnt in word_counter.items():
    total_freq += cnt
    if cnt < threshold:
        rare_cnt += 1
        rare_freq += cnt

print("단어 집합의 크기:", total_cnt)
print("등장 빈도수가 %s번 이하인 희귀 단어의 수: %s" % (threshold - 1, rare_cnt))
print("희귀 단어들의 비율: %s" % ((rare_cnt / total_cnt) * 100))
ratio = (rare_freq / total_freq) * 100
print("전체 등장 빈도수에서 희귀 단어들의 비율: %s" % ratio)
print(
    "단어 집합에서 희귀 단어를 제외시킬 경우의 단어 집합의 크기 %s"
    % (total_cnt - rare_cnt)
)


vocab_size = total_cnt - rare_cnt
vocab = vocab[:vocab_size]
print("단어 집합의 크기:", len(vocab))

# 단어에 정수 인코딩
word_to_index = {}
word_to_index["<pad>"] = 0
word_to_index["<unk>"] = 1
for idx, word in enumerate(vocab):
    word_to_index[word] = idx + 2
print("패딩과 UNK 고려한 단어 집합 크기:", len(word_to_index))
