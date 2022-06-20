# ファインチューニングしたモデルのテスト

from transformers import BertJapaneseTokenizer, BertForSequenceClassification
import pytorch_lightning as pl

tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-large-japanese')
bert_sc = BertForSequenceClassification.from_pretrained(
    './model_transformers'
)