# bert-text-classification
BERTでテキスト分類を試す。
[bert-large-japanese](https://huggingface.co/cl-tohoku/bert-large-japanese)をファインチューニングし、livedoorのニュースコーパスの本文からそのカテゴリーを予測する。

**データセット**  
[liveddorのニュースコーパス](https://www.rondhuit.com/download.html)  
./data/にあるtar.gzを解凍して使用する。  
`$ tar -zxf ldcc-20140209`

**プログラム**  
text_classification.pyを実行すれば、ファインチューニングができる。  
(実行後、以下の3つが生成される)  
・./lightning_logs : 学習時のログが保存される。  
・./model : ベストモデルが保存される。  
・./model_transformers : 指定することでモデルを直接読み込むことができる。

**TensorBoard**:学習時の学習データに対する損失の値の時間変化を確認できる。  
`$ tensorboard --logdir ./`  

test_model.py では、ファインチューニングしたモデルを使った推論ができる。  

**参考**  
BERTによる自然言語処理入門