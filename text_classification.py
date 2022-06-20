import random
import glob
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import BertJapaneseTokenizer, BertForSequenceClassification
import pytorch_lightning as pl

# GPU設定
"""import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
"""

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
max_length = 126
dataset_for_loader = []
for label, category in enumerate(tqdm(category_list)):
    for file_name in glob.glob(f'./data/text/{category}/{category}*'):
        file = open(file_name,'r')
        lines = file.read().splitlines()
        text = '\n'.join(lines[3:])
        encoding = tokenizer(
            text,
            max_length = max_length,
            padding = 'max_length',
            truncation = True
        )
        encoding['labels'] = label
        for k,v in encoding.items():
            encoding[k] = torch.tensor(v)
        dataset_for_loader.append(encoding)

# データセットを分割
random.shuffle(dataset_for_loader)
n = len(dataset_for_loader)
n_train = int(0.6*n)
n_val = int(0.2*n)
dataset_train = dataset_for_loader[:n_train]
dataset_val = dataset_for_loader[n_train:n_train + n_val]
dataset_test = dataset_for_loader[n_train+n_val:]

# データローダの作成
dataloader_train = DataLoader(
    dataset_train, batch_size=32, shuffle=True
)
dataloader_val = DataLoader(dataset_val, batch_size=256)
dataloader_test = DataLoader(dataset_test, batch_size=256)


class BertForSequenceClassification_pl(pl.LightningModule):

    def __init__(self, model_name, num_labels, lr):
        """
        Args:
            model_name (str): Transformersのモデル名
            num_labels (int): ラベルの数
            lr (float): 学習率
        """

        super().__init__()
        self.save_hyperparameters()

        # bertモデルのロード
        self.bert_sc = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )

    # 損失関数
    def training_step(self, batch, batch_idx):
        output = self.bert_sc(**batch)
        loss = output.loss
        self.log('train_loss', loss)
        return loss

    # 検証データ評価指標
    def validation_step(self, batch, batch_idx):
        output = self.bert_sc(**batch)
        val_loss = output.loss
        self.log('val_loss', val_loss)

    # テストデータ評価指標
    def test_step(self, batch, batch_idx):
        output = self.bert_sc(**batch)
        test_loss = output.loss
        self.log('test_loss', test_loss)

    # オプティマイザ - Adam使う
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

checkpoint = pl.callbacks.ModelCheckpoint(
    monitor = 'val_loss', #val_lossの監視
    mode = 'min', # val_lossが小さいモデルを保存
    save_top_k = 1, # 最も小さいものにする
    save_weights_only = True, # モデルの重みのみを保存
    dirpath= 'model/' # モデルの保存ディレクトリ
)

# 学習方法を指定
trainer = pl.Trainer(
    gpus=[1],
    # 1だと個数、[1]だと番号
    max_epochs=1,
    callbacks=[checkpoint]
)

# ファインチューニング
model = BertForSequenceClassification_pl(
    MODEL_NAME,
    num_labels = 9,
    lr=1e-5
)

trainer.fit(model, dataloader_train, dataloader_val)

best_model_path = checkpoint.best_model_path
print('ベストモデルのファイル: ', checkpoint.best_model_path)
print('ベストモデルの検証データに対する損失: ', checkpoint.best_model_score)

# テストデータによる評価
test = trainer.test(test_dataloaders=dataloader_test)
print(f'Accuracy: {test[0]["accuracy"]:.sf}')

# モデルのロード
model = BertForSequenceClassification_pl.load_from_checkpoint(
    best_model_path
)

# 保存
model.bert_sc.save_pretrained('./model_transformers')