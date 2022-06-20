import random
import glob
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import BertJapaneseTokenizer, BertForSequenceClassification
import pytorch_lightning as pl

# 日本語学習モデル
MODEL_NAME = 'cl-tohoku/bert-large-japanese'
