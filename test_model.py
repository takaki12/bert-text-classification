# ファインチューニングしたモデルのテスト

import torch
from torch.utils.data import DataLoader
from transformers import BertJapaneseTokenizer, BertForSequenceClassification
import pytorch_lightning as pl

# トークナイザ
tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-large-japanese')
# いつもの cl-tohoku/bert-base-japanese-whole-word-masking

MODEL_NAME = './model_transformers'

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
text = '最新の全国映画動員ランキングトップ10（6月18日・19日、興行通信社調べ）が発表され、『トップガン マーヴェリック』が土日2日間で動員35万4000人、興行収入5億9400万円をあげ、再び1位に返り咲いた。スクリーン数もさらに増えるなど勢いを増し、累計成績では動員364万人、興収56億円を突破している。'
text_label = 'mobie-enter'
label = category_list.index(text_label)

print("text: " + text)

encoding = tokenizer(
    text,
    max_length = 126,
    padding = 'max_length',
    truncation = True,
    return_tensors = 'pt'
)
print(encoding['input_ids'])

# bertモデルのロード
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME
)

output = model(**encoding)
labels_predicted = output.logits.argmax(-1)
print("Predict:" + category_list[labels_predicted])
print("Answer: " + text_label)
if category_list[labels_predicted] == text_label:
    print("True")
else:
    print("False")