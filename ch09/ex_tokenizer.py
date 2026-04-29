# 스크립트 파일 이름이 tokenize였더니 문제 발생...
from nltk.tokenize import TreebankWordTokenizer

tokenizer = TreebankWordTokenizer()

text = "Starting a home-based restaurant may be an ideal. it doesn't have a food chain or restaurant of their own."
print(tokenizer.tokenize(text))

import nltk
from nltk.tokenize import sent_tokenize

mydir = ".venv/nltk_data"
# nltk.download("punkt_tab", download_dir=".venv/nltk_data")
text = "His barber kept his word. But keeping such a huge secret to himself was driving him crazy. Finally, the barber went up a mountain and almost to the edge of a cliff. He dug a hole in the midst of some reeds. He looked about, to make sure no one was near."
print(sent_tokenize(text))

text = "I am actively looking for Ph.D. students. and you are a Ph.D student."
print(sent_tokenize(text))

import kss

text = "딥 러닝 자연어 처리가 재미있기는 합니다. 그런데 문제는 영어보다 한국어로 할 때 너무 어렵습니다. 이제 해보면 알걸요?"
print(kss.split_sentences(text))

from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

# nltk.download("averaged_perceptron_tagger_eng", download_dir=mydir)
text = "I am actively looking for Ph.D. students. and you are a Ph.D student."
tokenized_sentence = word_tokenize(text)

print(pos_tag(tokenized_sentence))

from konlpy.tag import Okt
from konlpy.tag import Kkma

okt = Okt()
kkma = Kkma()

print("OKT 형태소 분석 : ", okt.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
print("OKT 품사 태깅 : ", okt.pos("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
print("OKT 명사 추출 : ", okt.nouns("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))

print(
    "꼬꼬마 형태소 분석 : ", kkma.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요")
)
print("꼬꼬마 품사 태깅 : ", kkma.pos("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
print("꼬꼬마 명사 추출 : ", kkma.nouns("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
