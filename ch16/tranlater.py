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


def load_preprocessed_data():
    encoder_input, decoder_input, decoder_target = [], [], []

    with open("data/fra.txt", "r") as lines:
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
