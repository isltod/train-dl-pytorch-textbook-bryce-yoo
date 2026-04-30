import torch
import torch.nn as nn
import torch.optim as optim

sentence = "Repeat is the best medicine for memory".split()
# 여기서 set이 한 번 들어가서 순서가 책과 같아지지는 않는다...
vocab = list(set(sentence))
print(vocab)

word2index = {tkn: i for i, tkn in enumerate(vocab, 1)}
word2index["<unk>"] = 0
print(word2index)

index2word = {v: k for k, v in word2index.items()}
print(index2word)


# 문장을 단어 ID들의 배열로 만들고, 처음부터 마지막 - 1 단어까지가 X, 2번째부터 마지막까지가 Y
def build_data(sentence, word2index):
    encoded = [word2index[tkn] for tkn in sentence]
    input_seq, label_seq = encoded[:-1], encoded[1:]
    input_seq = torch.LongTensor(input_seq).unsqueeze(0)
    label_seq = torch.LongTensor(label_seq).unsqueeze(0)
    return input_seq, label_seq


X, Y = build_data(sentence, word2index)
print(X)
print(Y)


class Net(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size, batch_first=True):
        super(Net, self).__init__()
        # RNN 앞에 임베딩 넣기...
        self.embedding_layer = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=input_size
        )
        self.rnn_layer = nn.RNN(input_size, hidden_size, batch_first=batch_first)
        # 마지막 선형 변환에서 히든을 단어 사전 크기로....
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        # X(배치 1, 문장 안의 단어들 6), 단어들에 임베딩 룩업 붙이기...
        output = self.embedding_layer(x)
        # output(배치 1, 문장 6, 임베딩 차원 5)
        output, hidden = self.rnn_layer(output)
        # output(배치 1, 문장 6, 은닉층 20), hidden(배치 1, 마지막 단어 1, 은닉 20)
        output = self.linear(output)
        # output(배치 1, 문장 6, 단어 집합 크기 8) - unk 포함
        # 각 배치 자리의, 각 단어 위치에, 임베딩의 디코딩 정보를 원핫 벡터로 담아서 나왔는데...
        # 그걸 (배치 x 단어 위치, 단어 원핫) 벡터로 반환
        return output.view(-1, output.size(2))


vocab_size = len(word2index)
input_size = 5
hidden_size = 20

model = Net(vocab_size, input_size, hidden_size, batch_first=True)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters())

decode = lambda y: [index2word.get(x) for x in y]

for step in range(201):
    optimizer.zero_grad()
    output = model(X)
    # 애초에 Y는 X랑 같은 데이터에서 한 칸 앞 뒤로 잘라서 만들었으므로 배치 차원을 가지고 있다...
    # 그걸 없애야 X와 비교가 가능해진다...
    loss = loss_function(output, Y.view(-1))
    loss.backward()
    optimizer.step()

    if step % 40 == 0:
        print(f"Step: {step}, Loss: {loss.item()}")
        # 마지막 축에 대해서 softmax, 마지막 축에 대해서 argmax, 그리고 리스트로...
        pred = output.softmax(-1).argmax(-1).tolist()
        # 단어 ID 들의 리스트가 되고...그걸 단어로 바꿔서 Repeat 뒤로 붙이기...
        print(" ".join(["Repeat"] + decode(pred)))
        print()
