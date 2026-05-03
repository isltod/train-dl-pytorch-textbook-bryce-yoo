import nltk
from nltk import word_tokenize, pos_tag, ne_chunk

# nltk.download("averaged_perceptron_tagger_eng", download_dir=".venv/nltk_data")
# nltk.download("maxent_ne_chunker_tab", download_dir=".venv/nltk_data")
# nltk.download("words", download_dir=".venv/nltk_data")

sent = "James is working at Disney in London"
# 토큰화 뿐 아니라 품사 태깅까지...
tokenized_sentence = pos_tag(word_tokenize(sent))
print(tokenized_sentence)

# 개체명 인식
ner_sentence = ne_chunk(tokenized_sentence)
print(ner_sentence)

# 그러니까 여기까진 뭔가 사전에 정리된 개체명 인식이고, 이 아래는 인공지능으로 그걸 하게 만든다...이런건가?
import matplotlib.pyplot as plt
import numpy as np
import re
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

with open("data/train.txt", "r") as f:
    tagged_sentences = []
    sent = []
    # 데이터 형식이 '단어 뭐뭐 뭐뭐 품사\n' 이렇게 되어 있는 모양...그 중 단어와 품사만 추리기...
    for line in f:
        # 그걸 다시 '', -DOCSTART로 시작, '\n'만 있는 줄로 문장 단위로 잘라놓은 모양...
        if len(line) == 0 or line.startswith("-DOCSTART") or line[0] == "\n":
            # 문장 구분 표시와 만났고, 받아놓은 단어들 있으면 업데이트하고
            if len(sent) > 0:
                tagged_sentences.append(sent)
                # 그릇은 비우고
                sent = []
            continue
        # 띄어쓰기로 나누고
        splits = line.split(" ")
        # 마지막 단어의 \n 지우고
        splits[-1] = re.sub(r"\n", "", splits[-1])
        # 단어는 소문자로, [단어, 품사] 저장
        word = splits[0].lower()
        sent.append([word, splits[-1]])

print("전체 샘플 개수:", len(tagged_sentences))
print(tagged_sentences[0])

# 다시 단어 태그 분리...처음부터 안하고 왜?
sentences, ner_tags = [], []
for tagged_sentence in tagged_sentences:
    sent, tag_info = zip(*tagged_sentence)
    sentences.append(list(sent))
    ner_tags.append(list(tag_info))

print(sentences[0])
print(ner_tags[0])

X_train, X_test, y_train, y_test = train_test_split(
    sentences, ner_tags, test_size=0.2, random_state=777
)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train, test_size=0.2, random_state=777
)
print("훈련 데이터 크기:", len(X_train))
print("검증 데이터 크기:", len(X_valid))
print("시험 데이터 크기:", len(X_test))

# Vocab 만들기
words_list = []
for sent in X_train:
    for word in sent:
        words_list.append(word)

word_counts = Counter(words_list)
print("총 단어 수", len(word_counts))
print("훈련 데이터에서 단어 the의 등장 횟수:", word_counts["the"])
print("훈련 데이터에서 단어 love의 등장 횟수:", word_counts["love"])
vocab = sorted(word_counts, key=word_counts.get, reverse=True)
print("등장 빈도수 상위 10개 단어")
print(vocab[:10])

word_to_index = {}
word_to_index["<PAD>"] = 0
word_to_index["<UNK>"] = 1
for value, word in enumerate(vocab):
    word_to_index[word] = value + 2
vocab_size = len(word_to_index)
print("패딩과 UNK 고려한 단어 집합 크기:", vocab_size)


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


encoded_X_train = texts_to_sequences(X_train, word_to_index)
encoded_X_valid = texts_to_sequences(X_valid, word_to_index)
encoded_X_test = texts_to_sequences(X_test, word_to_index)

# 라벨도 정수 인코딩을 한다...여긴 라벨이 정수가 아니고 품사 글자들이니까...
flatten_tags = [tag for sent in y_train for tag in sent]
tag_vocab = list(set(flatten_tags))
tag_to_index = {}
tag_to_index["<PAD>"] = 0
for value, word in enumerate(tag_vocab):
    tag_to_index[word] = value + 1
tag_vocab_size = len(tag_to_index)
print("패딩까지 고려한 태그 집합 크기:", tag_vocab_size)


def encoding_label(sequence, tag_to_index):
    label_sequence = []
    for seq in sequence:
        label_sequence.append([tag_to_index[tag] for tag in seq])
    return label_sequence


encoded_y_train = encoding_label(y_train, tag_to_index)
encoded_y_valid = encoding_label(y_valid, tag_to_index)
encoded_y_test = encoding_label(y_test, tag_to_index)


def pad_sequences(sentences, max_len):
    # 문장 최대 길이 만큼만을 0으로 채운 결과 행렬 미리 만들고
    features = np.zeros((len(sentences), max_len), dtype=int)
    # 문장마다 돌면서
    for index, sentence in enumerate(sentences):
        # 문장 길이가 0이 아니면? 0도 있나?
        if len(sentence) != 0:
            # 문장 행에, 문장 길이까지만, 최대 길이 이하로 자른 그 문장 내용을 넣기...
            features[index, : len(sentence)] = np.array(sentence)[:max_len]
    return features


max_len = 500
padded_X_train = pad_sequences(encoded_X_train, max_len)
padded_X_valid = pad_sequences(encoded_X_valid, max_len)
padded_X_test = pad_sequences(encoded_X_test, max_len)

padded_y_train = pad_sequences(encoded_y_train, max_len)
padded_y_valid = pad_sequences(encoded_y_valid, max_len)
padded_y_test = pad_sequences(encoded_y_test, max_len)

X_train_tensor = torch.tensor(padded_X_train, dtype=torch.long)
y_train_tensor = torch.tensor(padded_y_train, dtype=torch.long)

X_valid_tensor = torch.tensor(padded_X_valid, dtype=torch.long)
y_valid_tensor = torch.tensor(padded_y_valid, dtype=torch.long)

X_test_tensor = torch.tensor(padded_X_test, dtype=torch.long)
y_test_tensor = torch.tensor(padded_y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# 이건 단방향 GRU 버전
# class NERTagger(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
#         super(NERTagger, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, output_dim)

#     def forward(self, x):
#         embedded = self.embedding(x)
#         gru_out, hidden = self.gru(embedded)
#         logits = self.fc(gru_out)
#         return logits


# 이건 양방향 LSTM 버전
class NERTagger(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=2):
        super(NERTagger, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        # 양방향은 출력이 두 배로 나온다...
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        logits = self.fc(lstm_out)
        return logits


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 장치:", device)

embedding_dim = 100
hidden_dim = 256
output_dim = tag_vocab_size
learning_rate = 0.01
num_epochs = 10
num_layers = 2

model = NERTagger(vocab_size, embedding_dim, hidden_dim, output_dim, num_layers)
model.to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def calculate_accuracy(logits, labels, ignore_index=0):
    predicted = torch.argmax(logits, dim=1)
    # 패딩 무시
    mask = labels != ignore_index
    # 이 AI 코드가 더 보기 좋은데...
    # predicted = predicted[mask]
    # labels = labels[mask]
    correct = (predicted == labels).masked_select(mask).sum().item()
    # 총 갯수도 패딩 무시
    total = mask.sum().item()
    accuracy = correct / total
    return accuracy


def evaludate(model, valid_dataloader, criterion, device):
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    model.eval()
    with torch.no_grad():
        for batch_X, batch_y in valid_dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            logits = model(batch_X)
            loss = criterion(logits.view(-1, output_dim), batch_y.view(-1))
            val_loss += loss.item()
            val_correct += calculate_accuracy(
                logits.view(-1, output_dim), batch_y.view(-1)
            ) * batch_y.size(0)
            val_total += batch_y.size(0)
    val_accuracy = val_correct / val_total
    val_loss /= len(valid_dataloader)
    return val_loss, val_accuracy


# 검증 손실 최소값을 찾을거니까 일단 제일 큰 값으로 시작
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
        loss = criterion(logits.view(-1, output_dim), batch_y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_correct += calculate_accuracy(
            logits.view(-1, output_dim), batch_y.view(-1)
        ) * batch_y.size(0)
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
        torch.save(model.state_dict(), "ch14/best_seq2seq_model.pth")

# 모델 평가
model.load_state_dict(torch.load("ch14/best_seq2seq_model.pth"))
model.to(device)
val_loss, val_accuracy = evaludate(model, valid_dataloader, criterion, device)
print(
    f"Best model Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_accuracy:.4f}"
)
test_loss, test_accuracy = evaludate(model, test_dataloader, criterion, device)
print(f"Best model Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.4f}")

# 직접 넣어보기
index_to_tag = {}
for key, value in tag_to_index.items():
    index_to_tag[value] = key


def predict_labels(text, model, word_to_ix, index_to_tag, max_len=150):
    # 아니 왜 이건 다르냐?
    tokens = text.split()
    token_indices = [word_to_ix.get(token, 1) for token in tokens]
    token_indices_padded = np.zeros(max_len, dtype=int)
    token_indices_padded[: len(token_indices)] = token_indices[:max_len]
    input_tensor = torch.tensor(token_indices_padded, dtype=torch.long).to(device)

    # tokens = word_tokenize(text.lower())
    # token_indices = [word_to_ix.get(token.lower(), 1) for token in tokens]
    # 이건 왜 패딩을 안하지?
    # padded_text = pad_sequences([encoded_text], max_len)
    # tensor_text = torch.tensor(padded_text).to(torch.int64)
    # input_tensor = torch.tensor([token_indices], dtype=torch.long).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        predicted_index = torch.argmax(logits, dim=-1).squeeze(0).tolist()
        predicted_index_no_pad = predicted_index[: len(tokens)]
        predicted_tag = [index_to_tag[index] for index in predicted_index_no_pad]
    return predicted_tag


sample = " ".join(X_test[0])
print(sample)
predicted_tags = predict_labels(sample, model, word_to_index, index_to_tag)
print("예측: ", predicted_tags)
print("실제: ", y_test[0])
