# bert-text-classification
BERTでテキスト分類をする。
[bert-large-japanese](https://huggingface.co/cl-tohoku/bert-large-japanese)をファインチューニングし、livedoorのニュースコーパスの本文からそのカテゴリーを予測させる。

**データセット**  
[liveddorのニュースコーパス](https://www.rondhuit.com/download.html)  
./data/にあるtar.gzを解凍して使用する。
'$ tar -zxf ldcc-20140209'

**プログラム**  
text_classification.pyを実行すれば、ファインチューニングができる。   
./model にベストモデルが保存されている。./model_transformers を指定することでモデルを直接読み込むことができる。  

**TensorBoard**:学習時の学習データに対する損失の値の時間変化を確認できる。
'''
$ tensorboard --logdir ./
'''

**参考**  
BERTによる自然言語処理入門