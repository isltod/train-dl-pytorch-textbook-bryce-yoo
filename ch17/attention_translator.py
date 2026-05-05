import numpy as np
import pandas as pd
import os
import re
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import unicodedata
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

num_samples = 33000


def unicode_to_ascii(s):
    # 불어 악센트 삭제...
    # 예시 : 'déjà diné' -> deja dine
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


def preprocess_sentence(sent):
    sent = unicode_to_ascii(sent.lower())

    # 단어와 구두점 사이에 공백 추가
    sent = re.sub(r"([?.!,¿])", r" \1", sent)
    # 알파벳과 구두점 외에 다 제거
    sent = re.sub(r"[^a-zA-Z?.!,?]+", r" ", sent)
    # 공백 여러 개는 한 개로
    sent = re.sub(r"\s+", " ", sent)
    return sent


# 전처리 테스트
en_sent = "Have you had dinner?"
fr_sent = "Avez-vous déjà diné?"
print("전처리 후 영어 문장", preprocess_sentence(en_sent))
print("전처리 후 프랑스어 문장", preprocess_sentence(fr_sent).encode("utf-8"))


def load_preprocessed_data():
    encoder_input, decoder_input, decoder_target = [], [], []

    with open("data/fra.txt", "r", encoding="utf-8") as lines:
        for i, line in enumerate(lines):
            # source 데이터와 target 데이터 분리, 전처리
            src_line, tar_line, _ = line.strip().split("\t")
            src_line_input = [w for w in preprocess_sentence(src_line).split()]
            tar_line = preprocess_sentence(tar_line)
            tar_line_input = [w for w in ("<sos> " + tar_line).split()]
            tar_line_target = [w for w in (tar_line + " <eos>").split()]

            encoder_input.append(src_line_input)
            decoder_input.append(tar_line_input)
            decoder_target.append(tar_line_target)

            if i == num_samples - 1:
                break

    return encoder_input, decoder_input, decoder_target


sents_en_in, sents_fra_in, sents_fra_out = load_preprocessed_data()
print("인코더의 입력 :", sents_en_in[:5])
print("디코더의 입력 :", sents_fra_in[:5])
print("디코더의 레이블 :", sents_fra_in[:5])
print(sents_fra_out[:5])


def build_vocab(sents):
    word_list = []
    for sent in sents:
        for word in sent:
            word_list.append(word)
    word_counts = Counter(word_list)
    vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    word_to_index = {}
    word_to_index["<PAD>"] = 0
    word_to_index["<UNK>"] = 1
    for index, word in enumerate(vocab):
        word_to_index[word] = index + 2
    return word_to_index


src_vocab = build_vocab(sents_en_in)
tar_vocab = build_vocab(sents_fra_in + sents_fra_out)
src_vocab_size = len(src_vocab)
tar_vocab_size = len(tar_vocab)
print("영어 단어 집합의 크기 :", src_vocab_size)
print("프랑스어 단어 집합의 크기 :", tar_vocab_size)

index_to_src = {v: k for k, v in src_vocab.items()}
index_to_tar = {v: k for k, v in tar_vocab.items()}


def texts_to_sequences(sents, word_to_index):
    encoded_X_data = []
    for sent in sents:
        index_sequences = []
        for word in sent:
            if word in word_to_index:
                index_sequences.append(word_to_index[word])
            else:
                index_sequences.append(word_to_index["<UNK>"])
        encoded_X_data.append(index_sequences)
    return encoded_X_data


encoder_input = texts_to_sequences(sents_en_in, src_vocab)
decoder_input = texts_to_sequences(sents_fra_in, tar_vocab)
decoder_target = texts_to_sequences(sents_fra_out, tar_vocab)


def pad_sequences(sentences, max_len=None):
    if max_len is None:
        max_len = max(len(sentence) for sentence in sentences)
    features = np.zeros((len(sentences), max_len), dtype=int)
    for index, sentence in enumerate(sentences):
        if len(sentence) != 0:
            features[index, : len(sentence)] = np.array(sentence)[:max_len]
    return features


encoder_input = pad_sequences(encoder_input)
decoder_input = pad_sequences(decoder_input)
decoder_target = pad_sequences(decoder_target)
print("인코더의 입력 :", encoder_input.shape)
print("디코더의 입력 :", decoder_input.shape)
print("디코더의 레이블 :", decoder_target.shape)

indices = np.arange(encoder_input.shape[0])
np.random.shuffle(indices)
encoder_input = encoder_input[indices]
decoder_input = decoder_input[indices]
decoder_target = decoder_target[indices]
print([index_to_src[word] for word in encoder_input[30997]])
print([index_to_tar[word] for word in decoder_input[30997]])
print([index_to_tar[word] for word in decoder_target[30997]])

n_of_val = int(33000 * 0.1)
print("검증 데이터의 개수 :", n_of_val)
encoder_input_train = encoder_input[:-n_of_val]
decoder_input_train = decoder_input[:-n_of_val]
decoder_target_train = decoder_target[:-n_of_val]

encoder_input_test = encoder_input[-n_of_val:]
decoder_input_test = decoder_input[-n_of_val:]
decoder_target_test = decoder_target[-n_of_val:]

encoder_input_train_tensor = torch.tensor(encoder_input_train, dtype=torch.long)
decoder_input_train_tensor = torch.tensor(decoder_input_train, dtype=torch.long)
decoder_target_train_tensor = torch.tensor(decoder_target_train, dtype=torch.long)

encoder_input_test_tensor = torch.tensor(encoder_input_test, dtype=torch.long)
decoder_input_test_tensor = torch.tensor(decoder_input_test, dtype=torch.long)
decoder_target_test_tensor = torch.tensor(decoder_target_test, dtype=torch.long)

# 배치 128을 32로, 아래 os.environ["CUDA_VISIBLE_DEVICES"] = "0" 바꿔서 돌아감...
batch_size = 32
train_dataset = TensorDataset(
    encoder_input_train_tensor,
    decoder_input_train_tensor,
    decoder_target_train_tensor,
)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

valid_dataset = TensorDataset(
    encoder_input_test_tensor,
    decoder_input_test_tensor,
    decoder_target_test_tensor,
)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

embedding_dim = 256
hidden_units = 256


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_units):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_units, batch_first=True)

    def forward(self, x):
        x = self.embedding(x)
        output, (hidden, cell) = self.lstm(x)
        # 둘 중에 뭐가 맥락 벡터냐? 인코더의 마지막 상태가 맥락이면 hidden인가?
        return output, hidden, cell


class Decoder(nn.Module):
    def __init__(self, tar_vocab_size, embedding_dim, hidden_units):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(tar_vocab_size, embedding_dim, padding_idx=0)
        # 닷프로덕트 어텐션인데...인코딩의 히든을 임베딩에 붙여서 사용...
        self.lstm = nn.LSTM(
            embedding_dim + hidden_units, hidden_units, batch_first=True
        )
        self.fc = nn.Linear(hidden_units, tar_vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, encoder_outputs, hidden, cell):
        x = self.embedding(x)
        # 닷프로덕트 어텐션, attention_scores(batch, seq_len, 1)
        # 근데 그냥 transpose(1, 2, 0) 하면 안되나? 아..그러러면 permute를 써야 한다...
        attention_scores = torch.bmm(
            encoder_outputs, hidden.transpose(0, 1).transpose(1, 2)
        )
        attention_weights = self.softmax(attention_scores)
        # context_vector(batch, 1, hidden)
        context_vector = torch.bmm(attention_weights.transpose(1, 2), encoder_outputs)
        # context_vector를 seq_len만큼 반복
        seq_len = x.shape[1]
        context_vector_repeated = context_vector.repeat(1, seq_len, 1)
        # concat context_vector와 임베딩 벡터
        x = torch.cat((x, context_vector_repeated), dim=2)

        # 디코더 LSTM에는 교사 강요를 위한 X, 맥락? hidden, 전 단어? cell이 전달되나...
        # 근데 x를 컨텍스트 벡터와 이어 붙였는데 그냥 이렇게 넣을 수 있나?
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        output = self.fc(output)
        # 일단은 output이 예측값이라는데...
        return output, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg):
        encoder_output, hidden, cell = self.encoder(src)
        # 훈련 중에는 디코더 출력 중의 output만 사용? 그럼 실전에서는?
        output, _, _ = self.decoder(trg, encoder_output, hidden, cell)
        return output


encoder = Encoder(src_vocab_size, embedding_dim, hidden_units)
decoder = Decoder(tar_vocab_size, embedding_dim, hidden_units)
model = Seq2Seq(encoder, decoder)
# 옛날 GPU 경고 메시지
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
loss_function = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters())


def evaluation(model, dataloader, loss_function, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    with torch.no_grad():
        for encoder_inputs, decoder_inputs, decoder_targets in dataloader:
            encoder_inputs, decoder_inputs, decoder_targets = (
                encoder_inputs.to(device),
                decoder_inputs.to(device),
                decoder_targets.to(device),
            )
            outputs = model(encoder_inputs, decoder_inputs)
            # (batch_size x seq_len, tar_vocab_size) 벡터로 손실 계산...
            loss = loss_function(
                outputs.view(-1, tar_vocab_size), decoder_targets.view(-1)
            )
            total_loss += loss.item()
            mask = decoder_targets != 0
            predicted = torch.argmax(outputs, dim=-1)
            total_correct += (
                (predicted == decoder_targets).masked_select(mask).sum().item()
            )
            total_count += mask.sum().item()
    return total_loss / len(dataloader), total_correct / total_count


def train(model, train_dataloader, valid_dataloader, loss_function, optimizer, device):
    num_epochs = 30
    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        model.train()

        for encoder_inputs, decoder_inputs, decoder_targets in train_dataloader:
            encoder_inputs, decoder_inputs, decoder_targets = (
                encoder_inputs.to(device),
                decoder_inputs.to(device),
                decoder_targets.to(device),
            )
            optimizer.zero_grad()
            outputs = model(encoder_inputs, decoder_inputs)
            # outputs.view(batch_size x seq_len, tar_vocab_size),
            # decoder_targets.view(batch_size x seq_len,) 정수 정답지
            loss = loss_function(
                outputs.view(-1, tar_vocab_size), decoder_targets.view(-1)
            )
            loss.backward()
            optimizer.step()

        # 손실과 정확도가 다르긴 한데, 정확도를 위해서 다시 훈련 루프를 도는게 맞냐?
        train_loss, train_acc = evaluation(
            model, train_dataloader, loss_function, device
        )
        val_loss, val_acc = evaluation(model, valid_dataloader, loss_function, device)
        print(
            f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            print(
                f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving model..."
            )
            best_val_loss = val_loss
            torch.save(model.state_dict(), "ch14/best_seq2seq_translator.pth")


# train(model, train_dataloader, valid_dataloader, loss_function, optimizer, device)
model.load_state_dict(torch.load("ch14/best_seq2seq_translator.pth"))
model.to(device)
# val_loss, val_accuracy = evaluation(model, valid_dataloader, loss_function, device)
# print(
#     f"Best model Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_accuracy:.4f}"
# )

# print("<sos> :", tar_vocab["<sos>"])
# print("<eos> :", tar_vocab["<eos>"])


# 번역기로 동작시키기
# 원문 정수 시퀀스를 텍스트 시퀀스로
def seq_to_src(input_seq):
    sentence = ""
    for encoded_word in input_seq:
        if encoded_word != 0:
            sentence += index_to_src[encoded_word] + " "
    return sentence


# 번역문 정수 시퀀스를 텍스트 시퀀스로
def seq_to_tar(input_seq):
    sentence = ""
    for encoded_word in input_seq:
        if (
            encoded_word != 0
            and encoded_word != tar_vocab["<sos>"]
            and encoded_word != tar_vocab["<eos>"]
        ):
            sentence += index_to_tar[encoded_word] + " "
    return sentence


def decode_sequence(
    input_seq,
    model,
    max_output_len,
    int_to_tar_token,
):
    # 원문 입력을 받아서 텐서로...
    encoder_inputs = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(device)
    # 인코더에서 히든과 셀을 받는데 어떤게 맥락이고 나머진 뭐냐...
    encoders_outputs, hidden, cell = model.encoder(encoder_inputs)
    # 디코더 입력을 일단 문장 시작 <sos>로 만들어놓고...
    decoder_input = (
        torch.tensor([tar_vocab["<sos>"]], dtype=torch.long).unsqueeze(0).to(device)
    )
    decoded_tokens = []
    # 문장의 각 단어자리마다 돌면서...
    for _ in range(max_output_len):
        # 처음엔 <sos>, 그 다음부턴 output.argmax 가지고 만든 token으로 모델 디코더 입력하고,
        # 더불어 hidden과 cell도 앞에 걸 받아서 계속 가는데...이게 뭐가 맥락이지?
        output, hidden, cell = model.decoder(
            decoder_input, encoders_outputs, hidden, cell
        )
        output_token = output.argmax(dim=-1).item()
        # 문장 마지막 기호를 만나면 끝이라...아예 max_output_len까지만 도는데?
        if output_token == tar_vocab["<eos>"]:
            break
        # 생산된 토큰은 문장을 만들기 위해서 누적시키고
        decoded_tokens.append(output_token)
        # 생산된 토큰을 다음 단어의 입력 자료로...
        decoder_input = (
            torch.tensor([output_token], dtype=torch.long).unsqueeze(0).to(device)
        )
    # 아마도 int_to_tar_token이 토큰을 단어로 바꾸는 모양...문장 반환...
    return " ".join([int_to_tar_token[token] for token in decoded_tokens])


for seq_index in [3, 50, 100, 300, 1001]:
    input_seq = encoder_input_train[seq_index]
    translated_text = decode_sequence(
        input_seq,
        model,
        20,
        index_to_tar,
    )
    print("입력문장 :", seq_to_src(input_seq))
    print("정답문장 :", seq_to_tar(decoder_target_train[seq_index]))
    print("번역문장 :", translated_text)
    print("-" * 50)

for seq_index in [3, 50, 100, 300, 1001]:
    input_seq = encoder_input_test[seq_index]
    translated_text = decode_sequence(
        input_seq,
        model,
        20,
        index_to_tar,
    )
    print("입력문장 :", seq_to_src(input_seq))
    print("정답문장 :", seq_to_tar(decoder_target_test[seq_index]))
    print("번역문장 :", translated_text)
    print("-" * 50)
