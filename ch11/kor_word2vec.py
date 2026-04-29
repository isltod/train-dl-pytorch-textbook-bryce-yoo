import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
from gensim.models.word2vec import Word2Vec
from konlpy.tag import Okt

training_data = pd.read_table("data/ratings.txt")
print(training_data.head())
print(len(training_data))
# NULL 있는지 확인...제거...
print(training_data.isnull().values.any())
training_data = training_data.dropna(how="any")
print(len(training_data))
print(training_data.isnull().values.any())
# 한글 외에 제거
training_data["document"] = training_data["document"].str.replace(
    "[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", ""
)
print(training_data.head())
# 토큰화
stopwords = [
    "의",
    "가",
    "이",
    "은",
    "들",
    "는",
    "좀",
    "잘",
    "걍",
    "과",
    "도",
    "를",
    "으로",
    "자",
    "에",
    "와",
    "한",
    "하다",
]
okt = Okt()
tokenized_data = []
for sentence in training_data["document"]:
    temp_X = okt.morphs(sentence, stem=True)
    temp_X = [word for word in temp_X if not word in stopwords]
    tokenized_data.append(temp_X)
print(tokenized_data[:10])

# 리뷰 길이 분포
print("리뷰의 최대 길이 : %d" % max(len(l) for l in tokenized_data))
print("리뷰의 평균 길이 : %f" % (sum(map(len, tokenized_data)) / len(tokenized_data)))
plt.hist([len(s) for s in tokenized_data], bins=50)
plt.xlabel("length of samples")
plt.ylabel("number of samples")
plt.show()

# 이게 학습이라고?
model = Word2Vec(
    sentences=tokenized_data, vector_size=100, window=5, min_count=5, workers=4, sg=0
)

# 그리고 학습이 다되?
print(model.wv.vectors.shape)
print(model.wv.most_similar("최민식"))
print(model.wv.most_similar("히어로"))
