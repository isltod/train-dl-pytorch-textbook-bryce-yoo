import torch
import torch.nn as nn

train_data = "you need to know how to code"
word_set = set(train_data.split())
vocab = {word: i + 2 for i, word in enumerate(word_set)}
vocab["<unk>"] = 0
vocab["<pad>"] = 1
print(vocab)

embedding_table = torch.FloatTensor(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.2, 0.9, 0.3],
        [0.1, 0.5, 0.7],
        [0.2, 0.1, 0.8],
        [0.4, 0.1, 0.1],
        [0.1, 0.8, 0.9],
        [0.6, 0.1, 0.1],
    ]
)
sample = "you need to run".split()
idxes = []
for word in sample:
    try:
        idxes.append(vocab[word])
    except KeyError:
        idxes.append(vocab["<unk>"])
idxes = torch.LongTensor(idxes)
lookup_result = embedding_table[idxes, :]
print(lookup_result)

embedding_layer = nn.Embedding(
    num_embeddings=len(vocab), embedding_dim=3, padding_idx=1
)
# 결국 임베딩 레이어의 가중치가 임베딩 밀집벡터...
print("임베딩 테이블", embedding_layer.weight)
lookup_result = embedding_layer(idxes)
print("샘플 임베딩 결과", lookup_result)
