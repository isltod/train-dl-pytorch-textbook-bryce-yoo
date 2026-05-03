import gensim
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
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

df = pd.read_csv("data/IMDB Dataset.csv")
df["sentiment"] = df["sentiment"].replace(["positive", "negative"], [1, 0])
print(df.head())

X_data = df["review"]
y_data = df["sentiment"]
print("영화 리뷰의 갯수:", len(X_data))
print("영화 라벨의 갯수:", len(y_data))

X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.5, random_state=0, stratify=y_data
)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train, test_size=0.2, random_state=0, stratify=y_train
)
print(X_train.shape, X_valid.shape, X_test.shape)
print(y_train.shape, y_valid.shape, y_test.shape)
print(f"-----훈련 데이터의 비율-----")
print(f"부정 리뷰 = {round(y_train.value_counts()[0] / len(y_train) * 100, 3)}%")
print(f"긍정 리뷰 = {round(y_train.value_counts()[1] / len(y_train) * 100, 3)}%")
print(f"-----검증 데이터의 비율-----")
print(f"부정 리뷰 = {round(y_valid.value_counts()[0] / len(y_valid) * 100, 3)}%")
print(f"긍정 리뷰 = {round(y_valid.value_counts()[1] / len(y_valid) * 100, 3)}%")
print(f"-----시험 데이터의 비율-----")
print(f"부정 리뷰 = {round(y_test.value_counts()[0] / len(y_test) * 100, 3)}%")
print(f"긍정 리뷰 = {round(y_test.value_counts()[1] / len(y_test) * 100, 3)}%")


def tokenize(sentences):
    tokenized_sentences = []
    for sent in tqdm(sentences):
        tokenized_sent = word_tokenize(sent.lower())
        tokenized_sent = [word.lower() for word in tokenized_sent]
        tokenized_sentences.append(tokenized_sent)
    return tokenized_sentences


tokenized_X_train = tokenize(X_train)
tokenized_X_valid = tokenize(X_valid)
tokenized_X_test = tokenize(X_test)

word_list = []
for sent in tokenized_X_train:
    for word in sent:
        word_list.append(word)

word_counts = Counter(word_list)
vocab = sorted(word_counts, key=word_counts.get, reverse=True)
print("등장 빈도수 상위 10개 단어")
print(vocab[:10])

threshold = 3
total_cnt = len(word_counts)
rare_cnt = 0
total_freq = 0
rare_freq = 0

for word, value in word_counts.items():
    total_freq += value
    if value < threshold:
        rare_cnt += 1
        rare_freq += value

vocab_size = total_cnt - rare_cnt
vocab = vocab[:vocab_size]

word_to_index = {}
word_to_index["<PAD>"] = 0
word_to_index["<UNK>"] = 1
for i, word in enumerate(vocab):
    word_to_index[word] = i + 2
print("패딩과 UNK 고려한 단어 집합 크기:", len(word_to_index))
vocab_size = len(word_to_index)


def texts_to_sequences(tokenized_X_data, word_to_index):
    encoded_X_data = []
    for sent in tokenized_X_data:
        index_sequences = []
        for word in sent:
            # 책은 try except 구문으로 했는데, 가능하면 if else가 낫겠다...
            if word in word_to_index:
                index_sequences.append(word_to_index[word])
            else:
                index_sequences.append(word_to_index["<UNK>"])
        encoded_X_data.append(index_sequences)
    return encoded_X_data


encoded_X_train = texts_to_sequences(tokenized_X_train, word_to_index)
encoded_X_val = texts_to_sequences(tokenized_X_valid, word_to_index)
encoded_X_test = texts_to_sequences(tokenized_X_test, word_to_index)


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


max_len = 500
padded_X_train = pad_sequences(encoded_X_train, max_len)
padded_X_val = pad_sequences(encoded_X_val, max_len)
padded_X_test = pad_sequences(encoded_X_test, max_len)

train_label_tensor = torch.tensor(np.array(y_train))
valid_label_tensor = torch.tensor(np.array(y_valid))
test_label_tensor = torch.tensor(np.array(y_test))

encoded_train = torch.tensor(padded_X_train).to(torch.int64)
encoded_valid = torch.tensor(padded_X_val).to(torch.int64)
encoded_test = torch.tensor(padded_X_test).to(torch.int64)

train_dataset = TensorDataset(encoded_train, train_label_tensor)
valid_dataset = TensorDataset(encoded_valid, valid_label_tensor)
test_dataset = TensorDataset(encoded_test, test_label_tensor)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(
    "data/GoogleNews-vectors-negative300.bin", binary=True
)
embedding_matrix = np.zeros((vocab_size, 300))


def get_vector(word):
    if word in word2vec_model:
        return word2vec_model[word]
    else:
        return None


for word, i in word_to_index.items():
    if i > 2:
        temp = get_vector(word)
        if temp is not None:
            embedding_matrix[i] = temp
print(word_to_index["apple"])
print(np.all(embedding_matrix[word_to_index["apple"]] == word2vec_model["apple"]))


class CNN(nn.Module):
    def __init__(self, vocab_size, num_labels):
        super(CNN, self).__init__()
        self.num_filter_size = 1
        self.num_filters = 256
        # 여기가 자체 임베딩과 사전 훈련 임베딩을 쓸 때 차이...
        # self.word_embed = nn.Embedding(
        #     num_embeddings=vocab_size, embedding_dim=128, padding_idx=0
        # )
        # 요게 AI 추천 코드고 맞을거 같은데...
        # self.word_embed = nn.Embedding.from_pretrained(
        #     torch.FloatTensor(embedding_matrix), freeze=True
        # )
        self.word_embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=300)
        self.word_embed.weight = nn.Parameter(
            torch.tensor(embedding_matrix, dtype=torch.float32)
        )
        # 그리고 외부 임베딩도 훈련시킨다...
        self.word_embed.weight.requires_grad = True

        # 컨볼류션 층 하나만, 커널 5의 필터 하나만 사용? 아니고 num_filters만큼 256개 필터를 사용하는데?
        self.conv1 = nn.Conv1d(
            # 요것도 외부 임베딩에 맞춰 바꾸기...
            # in_channels=128,
            in_channels=300,
            out_channels=self.num_filters,
            kernel_size=5,
            stride=1,
        )
        self.dropout = nn.Dropout(p=0.5)
        # 여기서 1은 num_filter_size 말하는거 같은데...
        self.fc1 = nn.Linear(1 * self.num_filters, num_labels, bias=True)

    def forward(self, inputs):
        # inputs: (batch_size, seq_len)
        # inputs -> (batch_size, seq_len, embedding_dim) -> (batch_size, embedding_dim, seq_len)
        embedded = self.word_embed(inputs).permute(0, 2, 1)
        # 컨볼루션 결과는 임베딩 차원을 없애면서 1 문장 500개 단어를 커널 5로 도니까 4가 줄어서 496
        # conv1(배치 32, 커널 수 256, 컨볼루션 결과 496) -> permute(배치, 컨볼루션, 커널)
        # max(1) 컨볼루션 차원(1) 압축 -> [0] max_val과 arg_max 중 max_val -> (배치, 커널 수)
        x = F.relu(self.conv1(embedded).permute(0, 2, 1).max(1)[0])
        y_pred = self.fc1(self.dropout(x))
        return y_pred


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 장치:", device)

model = CNN(vocab_size, num_labels=len(set(y_train)))
model.to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)


def calculate_accuracy(logits, labels):
    # _, predicted = torch.max(logits, 1)
    predicted = torch.argmax(logits, dim=1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    return accuracy


def evaludate(model, valid_dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch_X, batch_y in valid_dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            val_loss += loss.item()
            val_correct += calculate_accuracy(logits, batch_y) * batch_y.size(0)
            val_total += batch_y.size(0)
    val_accuracy = val_correct / val_total
    val_loss /= len(valid_dataloader)
    return val_loss, val_accuracy


# 검증 손실 최소값을 찾을거니까 일단 제일 큰 값으로 시작
num_epochs = 5
best_val_loss = float("inf")
for epoch in range(num_epochs):
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    model.train()

    # 이건 학습이고
    for batch_X, batch_y in train_dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        logits = model(batch_X)
        loss = criterion(logits, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_correct += calculate_accuracy(logits, batch_y) * batch_y.size(0)
        train_total += batch_y.size(0)

    train_accuracy = train_correct / train_total
    train_loss /= len(train_dataloader)

    # 이건 검증
    val_loss, val_accuracy = evaludate(model, valid_dataloader, criterion, device)

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.4f}")
    print(f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")
    if val_loss < best_val_loss:
        print(
            f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving model..."
        )
        best_val_loss = val_loss
        torch.save(model.state_dict(), "ch13/best_word2vec_model.pth")

# 모델 평가
model.load_state_dict(torch.load("ch13/best_word2vec_model.pth"))
model.to(device)
val_loss, val_accuracy = evaludate(model, valid_dataloader, criterion, device)
print(
    f"Best model Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_accuracy:.4f}"
)
test_loss, test_accuracy = evaludate(model, test_dataloader, criterion, device)
print(f"Best model Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.4f}")

# 직접 넣어보기
index_to_tag = {0: "부정", 1: "긍정"}


def predict(text, model, word_to_index, index_to_tag):
    model.eval()
    tokens = word_tokenize(text.lower())
    token_indices = [word_to_index.get(token.lower(), 1) for token in tokens]
    # 이건 왜 패딩을 안하지?
    # padded_text = pad_sequences([encoded_text], max_len)
    # tensor_text = torch.tensor(padded_text).to(torch.int64)
    input_tensor = torch.tensor([token_indices], dtype=torch.long).to(device)
    with torch.no_grad():
        logits = model(input_tensor)
        _, predicted_index = torch.max(logits, dim=1)
        predicted_tag = index_to_tag[predicted_index.item()]
    print(predicted_tag)


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
