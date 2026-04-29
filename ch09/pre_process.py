# en_text = "A Dog Run back corner near spare bedrooms"

# import spacy

# # python -m spacy download en_core_web_sm 해야 먹힌다...
# spacy_en = spacy.load("en_core_web_sm")


# def tokenize(en_text):
#     return [tok.text for tok in spacy_en.tokenizer(en_text)]


# print(tokenize(en_text))

# import nltk
# from nltk.tokenize import word_tokenize

# print(word_tokenize(en_text))

# kor_text = "사과의 놀라운 효능이라는 글을 봤어. 그래서 오늘 사과를 먹으려고 했는데 사과가 썩어서 슈퍼에 가서 사과랑 오렌지 사왔어"
from konlpy.tag import Mecab

# mecab = Mecab()
# print(mecab.morphs(kor_text))

import urllib.request
import pandas as pd

urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt",
    filename="data/ratings.txt",
)
data = pd.read_table("data/ratings.txt")
print(data[:10])
sample_data = data[:100]
# 한글 문자 아니면 다 제거
sample_data["document"] = sample_data["document"].str.replace(
    "[^ㄱ-ㅎㅏ-ㅣ가-힣]", "", regex=True
)
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
tokenizer = Mecab("data/mecab/mecab-ko-dic")
