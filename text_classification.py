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

