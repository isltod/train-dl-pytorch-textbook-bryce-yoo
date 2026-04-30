import numpy as np
from collections import Counter
import gensim

sentences = [
    "nice great best amazing",
    "stop lies",
    "pitiful nerd",
    "excellent work",
    "supreme quality",
    "bad",
    "highly respectable",
]
y_train = [1, 0, 0, 1, 1, 0, 1]

tokenized_sentences = [sentence.split() for sentence in sentences]

word_list = []
for sentence in tokenized_sentences:
    for word in sentence:
        word_list.append(word)

word_counter = Counter(word_list)
print(len(word_counter))

vocab = sorted(word_counter, key=word_counter.get, reverse=True)
print(vocab)

word_to_index = {}
word_to_index["<pad>"] = 0
word_to_index["<unk>"] = 1
for idx, word in enumerate(vocab):
    word_to_index[word] = idx + 2

vocab_size = len(word_to_index)
print("패딩과 UNK 고려한 단어 집합 크기:", vocab_size)


def texts_to_sequences(tokenized_X_data, word_to_index):
    encoded_X_data = []
    for sentence in tokenized_X_data:
        index_sequences = []
        for word in sentence:
            try:
                index_sequences.append(word_to_index[word])
            except KeyError:
                index_sequences.append(word_to_index["<unk>"])
        encoded_X_data.append(index_sequences)
    return encoded_X_data


X_encoded = texts_to_sequences(tokenized_sentences, word_to_index)
print(X_encoded)

max_len = max(len(l) for l in X_encoded)
print(max_len)


def pad_sequences(sequences, max_len):
    features = np.zeros((len(sequences), max_len), dtype=int)
    for i, sequence in enumerate(sequences):
        if len(sequence) != 0:
            features[i, : len(sequence)] = np.array(sequence)[:max_len]
    return features


X_padded = pad_sequences(X_encoded, max_len)
print(X_padded)
y_train = np.array(y_train)

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset


# 1. 사전 훈련된 임베딩 없이 그냥 하기...뭘 하지?
class SimpleModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SimpleModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(embedding_dim * max_len, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x(배치 2, 문장 4개의 단어 ID) match (단어 ID 17, 임베딩 100) -> (배치 2, 단어 수 4, 임베딩 차원 100)
        x = self.embedding(x)
        x = self.flatten(x)
        # (배치 2, 문장 길이 4 x 임베딩 차원 100)
        x = self.fc(x)
        # (배치 2, 1)
        x = self.sigmoid(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embedding_dim = 100
simple_model = SimpleModel(vocab_size, embedding_dim).to(device)
criterion = nn.BCELoss().to(device)
optimizer = Adam(simple_model.parameters())

train_dataset = TensorDataset(
    torch.tensor(X_padded, dtype=torch.long), torch.tensor(y_train, dtype=torch.float32)
)
train_loader = DataLoader(train_dataset, batch_size=2)
print(len(train_loader))


def train_model(model, criterion, optimizer, train_loader, device, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).view(-1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")


train_model(simple_model, criterion, optimizer, train_loader, device, 10)


# 2. 사전 훈련된 임베딩 - 구글 word2vec 활용하기...
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(
    "data/GoogleNews-vectors-negative300.bin", binary=True
)
print(word2vec_model.vectors.shape)

embedding_matrix = np.zeros((vocab_size, 300))
print("embedding_matrix.shape", embedding_matrix.shape)


# 구글 word2vec의 임베딩 벡터 받기, 없으면 None
def get_embedding_vector(word):
    if word in word2vec_model:
        return word2vec_model[word]
    else:
        return None


for word, index in word_to_index.items():
    # 0번 패드와 1번 unk는 넘어가고
    if index < 2:
        continue
    vector = get_embedding_vector(word)
    if vector is not None:
        embedding_matrix[index] = vector

# word2vec_model과 embedding_matrix에서 great가 동일한지 확인하는데...이걸 왜 하지?
print(np.all(embedding_matrix[word_to_index["great"]] == word2vec_model["great"]))


class PretrainedEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(PretrainedEmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(
            # 이거 원래 embedding_matrix도 매개변수로 받아야 하는건데...그냥 위에서 한걸 쓰고 있네...
            torch.tensor(embedding_matrix, dtype=torch.float32)
        )
        self.embedding.weight.requires_grad = True
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(embedding_dim * max_len, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


pretrained_model = PretrainedEmbeddingModel(vocab_size, 300).to(device)
criterion = nn.BCELoss().to(device)
optimizer = Adam(pretrained_model.parameters())

train_model(pretrained_model, criterion, optimizer, train_loader, device, 10)
