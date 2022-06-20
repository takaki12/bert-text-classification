import random
import glob
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import BertJapaneseTokenizer, BertForSequenceClassification
import pytorch_lightning as pl

# GPU設定
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 日本語学習モデル
MODEL_NAME = 'cl-tohoku/bert-large-japanese'
# いつもの cl-tohoku/bert-base-japanese-whole-word-masking

# データを前処理
category_list = [
    'dokujo-tsushin',
    'it-life-hack',
    'kaden-channel',
    'livedoor-homme',
    'mobie-enter',
    'peachy',
    'smax',
    'sports-watch',
    'topic-news'
]

# トークナイザ
tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)

# データの形式を整える
cnt = 0
max_length = 512
dataset_for_loader = []
for label, category in enumerate(tqdm(category_list)):
    for file_name in glob.glob(f'./data/text/{category}/{category}*'):
        file = open(file_name,'r')
        lines = file.readline().splitlines()
        text = '\n'.join(lines[3:])
        encoding = tokenizer(
            text,
            max_length = max_length,
            padding = 'max_length',
            truncation = True
        )
        encoding['labels'] = label
        for k,v in encoding.items():
            encoding = {k : torch.tensor(v)}
        dataset_for_loader.append(encoding)

    if cnt == 0:
        print(dataset_for_loader[0:3])