# MIROSTAT で意外さをコントロールした文章生成 (NLP Hacks vol.2)
NLP勉強会の [NLP Hacks vol.2](https://note.com/elyza/n/n6437e41bc94b) での LT に使ったコードです。
Transformers の言語モデルを使って、top-k sampling, top-p sampling, MIROSTAT などのデコーディング手法を試すことができます。

## 環境
python 3.8 以上を使ってください. 。
[Poetry](https://python-poetry.org/) を使って依存ライブラリをインストールします。

```bash
poetry install
```

## 使用例
`GenerateText.ipynb` にコードの使用例があります。
