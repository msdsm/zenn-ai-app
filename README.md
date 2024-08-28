# つくりながら学ぶ！AIアプリ開発入門 - LangChain & StreamlitによるChatGPT API徹底活用

## ソース
- https://zenn.dev/ml_bear/books/d1f060a3f166a5

## 階層

## メモ
###  LangChainとは
- LLMを用いたアプリケーション開発を効率的に行うためのライブラリ
- `pip install langchain`で入る

### ベクトルデータベース(Vector Database) / ベクトルストア(Vector Store)とは
- テキスト、画像、音声などのデータをembeddingしたベクトル表現として保存するデータベースのこと
- ベクトル検索(Vector Search)ができる
  - 各ベクトル間の類似度を利用する
  - 類似度はコサイン類似度やk近傍法など
- レコメンデーション用の検索データベースや、RAG用の検索データストアとして役立つ