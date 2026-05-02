import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import Counter
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from tqdm import tqdm

print(torch.__version__)

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
y_data = y_data.map({"positive": 1, "negative": 0})

print("영화 리뷰의 갯수:", len(X_data))
print("영화 라벨의 갯수:", len(y_data))
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.5, random_state=0, stratify=y_data
)
X_train, X_valid, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=0, stratify=y_train
)
print(X_train.shape, X_valid.shape, X_test.shape)
print(y_train.shape, y_val.shape, y_test.shape)


def print_label_ratio(data):
    print(f"부정 리뷰 = {round(data.value_counts()[1] / len(data) * 100, 3)}%")
    print(f"긍정 리뷰 = {round(data.value_counts()[0] / len(data) * 100, 3)}%")


print("훈련 데이터 비율--------------")
print_label_ratio(y_train)
print("검증 데이터 비율--------------")
print_label_ratio(y_val)
print("시험 데이터 비율--------------")
print_label_ratio(y_test)


# 토큰화
# nltk.download("punkt_tab", download_dir=".venv/nltk_data")


def tokenize(sentences):
    tokenized_sentences = []
    for sentence in tqdm(sentences):
        tokenized_sent = word_tokenize(sentence.lower())
        tokenized_sent = [word.lower() for word in tokenized_sent]
        tokenized_sentences.append(tokenized_sent)
    return tokenized_sentences


tokenized_X_train = tokenize(X_train)
tokenized_X_valid = tokenize(X_valid)
tokenized_X_test = tokenize(X_test)

# vocab 만들기
word_list = []
for sent in tokenized_X_train:
    for word in sent:
        word_list.append(word)

word_counts = Counter(word_list)
print("총 단어 수", len(word_counts))
print("훈련 데이터에서 단어 the의 등장 횟수:", word_counts["the"])
print("훈련 데이터에서 단어 love의 등장 횟수:", word_counts["love"])
vocab = sorted(word_counts, key=word_counts.get, reverse=True)
print("등장 빈도수 상위 10개 단어")
print(vocab[:10])

# 등장 빈도수 3 이하 단어들 제외
threshold = 3
# 총 단어 수
total_cnt = len(word_counts)
# 등장 빈도수가 threshold보다 적은 단어들 개수
rare_cnt = 0
# 훈련 데이터 전체 단어 빈도수의 합
total_freq = 0
# 등장 빈도수가 threshold보다 적은 단어들의 빈도수 합
rare_freq = 0

for word, value in word_counts.items():
    total_freq += value
    if value < threshold:
        rare_cnt += 1
        rare_freq += value

print("단어 집합의 크기:", total_cnt)
print("등장 빈도수가 %s번 이하인 희귀 단어의 수: %s" % (threshold - 1, rare_cnt))
print("희귀 단어들의 비율: %s" % ((rare_cnt / total_cnt) * 100))
ratio = (rare_freq / total_freq) * 100
print("전체 등장 빈도수에서 희귀 단어들의 비율: %s" % ratio)
print(
    "단어 집합에서 희귀 단어를 제외시킬 경우의 단어 집합의 크기 %s"
    % (total_cnt - rare_cnt)
)


# 전체 단어 개수 중 빈도수 2이하인 단어는 제거.
vocab_size = total_cnt - rare_cnt
vocab = vocab[:vocab_size]
print("단어 집합의 크기:", len(vocab))

word_to_index = {}
word_to_index["<PAD>"] = 0
word_to_index["<UNK>"] = 1
for index, word in enumerate(vocab):
    word_to_index[word] = index + 2
# 패팅과 UNK 넣어서 다시 단어집합 크기 설정
print("패딩과 UNK 고려한 단어 집합 크기:", len(word_to_index))
vocab_size = len(word_to_index)


# 단어에 정수 인코딩
def texts_to_sequences(tokenized_X_data, word_to_index):
    encoded_X_data = []
    for sentence in tqdm(tokenized_X_data):
        index_sequence = []
        for word in sentence:
            # 책은 try except 구문으로 했는데, 가능하면 if else가 낫겠다...
            if word in word_to_index:
                index_sequence.append(word_to_index[word])
            else:
                index_sequence.append(word_to_index["<UNK>"])
        encoded_X_data.append(index_sequence)
    return encoded_X_data


encoded_X_train = texts_to_sequences(tokenized_X_train, word_to_index)
encoded_X_val = texts_to_sequences(tokenized_X_valid, word_to_index)
encoded_X_test = texts_to_sequences(tokenized_X_test, word_to_index)

# 상위 샘플 2개 출력 - 책의 for문 보다는 인덱싱이 낫겠다...
# print(encoded_X_train[:2])

# 정수 인코딩 결과를 디코딩
index_to_word = {}
for word, value in word_to_index.items():
    index_to_word[value] = word

decoded_sample = [index_to_word[index] for index in encoded_X_train[0]]
# print(decoded_sample)

# 문장 길이 분포 확인하고 자르거나 패딩 넣어 맞추기
print("리뷰의 최대 길이:", max(len(review) for review in encoded_X_train))
print("리뷰의 평균 길이:", sum(map(len, encoded_X_train)) / len(encoded_X_train))
plt.hist([len(review) for review in encoded_X_train], bins=50)
plt.xlabel("length of samples")
plt.ylabel("number of samples")
# plt.show()


def below_threshold_len(max_len, nested_list):
    count = 0
    for sentence in nested_list:
        if len(sentence) <= max_len:
            count += 1
    print(
        f"전체 샘플 중 길이가 {max_len} 이하인 샘플의 비율: {round(count / len(nested_list), 4) * 100}%"
    )


max_len = 500
below_threshold_len(max_len, encoded_X_train)


# 패딩 넣기 함수
def pad_sequences(sequences, max_len):
    # 문장 최대 길이 만큼만을 0으로 채운 결과 행렬 미리 만들고
    features = np.zeros((len(sequences), max_len), dtype=int)
    # 문장마다 돌면서
    for i, sequence in enumerate(sequences):
        # 문장 길이가 0이 아니면? 0도 있나?
        if len(sequence) != 0:
            # 문장 행에, 문장 길이까지만, 최대 길이 이하로 자른 그 문장 내용을 넣기...
            features[i, : len(sequence)] = np.array(sequence)[:max_len]
    return features


padded_X_train = pad_sequences(encoded_X_train, max_len)
padded_X_val = pad_sequences(encoded_X_val, max_len)
padded_X_test = pad_sequences(encoded_X_test, max_len)

print("훈련 데이터 크기:", padded_X_train.shape)
print("검증 데이터 크기:", padded_X_val.shape)
print("시험 데이터 크기:", padded_X_test.shape)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 장치:", device)


# GRU 클래스 만들고
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        # gru_out(batch_size, seq_length, hidden_dim)
        # GRU의 마지막 은닉 상태(hidden state)를 사용
        gru_out, hidden = self.gru(embedded)  # hidden: (1, batch_size, hidden_dim)
        flat_hidden = hidden.squeeze(0)  # (batch_size, hidden_dim)
        logits = self.fc(flat_hidden)  # (batch_size, output_dim)
        return logits


# 데이터셋 준비하고
from torch.utils.data import DataLoader, TensorDataset

encoded_train = torch.tensor(padded_X_train).to(torch.int64)
train_label_tensor = torch.tensor(np.array(y_train))
train_dataset = TensorDataset(encoded_train, train_label_tensor)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

encoded_test = torch.tensor(padded_X_test).to(torch.int64)
test_label_tensor = torch.tensor(np.array(y_test))
test_dataset = TensorDataset(encoded_test, test_label_tensor)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

encoded_val = torch.tensor(padded_X_val).to(torch.int64)
val_label_tensor = torch.tensor(np.array(y_val))
val_dataset = TensorDataset(encoded_val, val_label_tensor)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

total_batch = len(train_dataloader)
print("총 배치 수:", total_batch)

# 모델 만들고
embedding_dim = 100
hidden_dim = 128
output_dim = 2
learning_rate = 0.01
num_epochs = 5

model = TextClassifier(vocab_size, embedding_dim, hidden_dim, output_dim).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def calculate_accuracy(logits, labels):
    # _, predicted = torch.max(logits, 1)
    predicted = torch.argmax(logits, dim=1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    return accuracy


def evaludate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            logits = model(data)
            loss = criterion(logits, labels)
            val_loss += loss.item()
            val_correct += calculate_accuracy(logits, labels) * labels.size(0)
            val_total += labels.size(0)
    # 일단 지금은 분모가 둘 다 같은 거 같고...len(dataloader)로 나누는게 맞나?
    val_loss /= len(dataloader)
    val_accuracy = val_correct / val_total
    return val_loss, val_accuracy


# 1. cuDNN 사용 안 함 (속도는 느려지지만 에러 해결 가능성 높음)
# torch.backends.cudnn.enabled = False

# 2. 또는, 비결정론적 알고리즘 사용 비활성화
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# 검증 손실 최소값을 찾을거니까 일단 제일 큰 값으로 시작
best_val_loss = float("inf")
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    # 이건 학습이고
    for batch_X, batch_y in train_dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        logits = model(batch_X)
        loss = criterion(logits, batch_y)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_correct += calculate_accuracy(logits, batch_y) * batch_y.size(0)
        train_total += batch_y.size(0)

    train_loss /= len(train_dataloader)
    train_accuracy = train_correct / train_total

    # 이건 검증
    val_loss, val_accuracy = evaludate(model, val_dataloader, criterion, device)

    print(
        f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}"
    )
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "ch13/best_gru_model.pth")

# 모델 평가
model.load_state_dict(torch.load("ch13/best_gru_model.pth"))
model.eval()
val_loss, val_accuracy = evaludate(model, val_dataloader, criterion, device)
print(f"Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_accuracy:.4f}")
test_loss, test_accuracy = evaludate(model, test_dataloader, criterion, device)
print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.4f}")

# 직접 넣어보기
index_to_tag = {0: "부정", 1: "긍정"}


def predict(text, model, word_to_index, index_to_tag):
    model.eval()
    tokenized_text = word_tokenize(text.lower())
    encoded_text = [word_to_index.get(word.lower(), 1) for word in tokenized_text]
    # 이건 왜 패딩을 안하지?
    # padded_text = pad_sequences([encoded_text], max_len)
    # tensor_text = torch.tensor(padded_text).to(torch.int64)
    tensor_text = torch.tensor([encoded_text], dtype=torch.long).to(device)
    with torch.no_grad():
        logits = model(tensor_text)
        # 이렇게 하나...
        probabilities = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        predicted_tag1 = index_to_tag[predicted_class]
        # 이렇게 하나...
        _, predicted_index = torch.max(logits, dim=1)
        predicted_tag2 = index_to_tag[predicted_index.item()]
    return predicted_tag1, predicted_tag2


test_input = (
    "This movie was just way too overrated. The fighting was not professional and in "
    "slow motion. I was expecting more from a 200 million budget movie. "
    "The little sister of T.Challa was just trying too hard to be funny. "
    "The story was really dumb as well. "
    "Don't watch this movie if you are going because others say its great "
    "unless you are a Black Panther fan or Marvels fan."
)
predict(test_input, model, word_to_index, index_to_tag)

test_input = (
    " I was lucky enough to be included in the group to see the advanced screening "
    "in Melbourne on the 15th of April, 2012. "
    "And, firstly, I need to say a big thank-you to Disney and Marvel Studios. "
    "Now, the film... how can I even begin to explain how I feel about this film? "
    "It is, as the title of this review says a 'comic book triumph'. "
    "I went into the film with very, very high expectations and I was not disappointed. "
    "Seeing Joss Whedon's direction and envisioning of the film come to life on the big screen is perfect. "
    "The script is amazingly detailed and laced with sharp wit a humor. "
    "The special effects are literally mind-blowing and the action scenes are "
    "both hard-hitting and beautifully choreographed."
)
predict(test_input, model, word_to_index, index_to_tag)
