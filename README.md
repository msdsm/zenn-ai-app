# つくりながら学ぶ！AIアプリ開発入門 - LangChain & StreamlitによるChatGPT API徹底活用

## ソース
- https://zenn.dev/ml_bear/books/d1f060a3f166a5

## 環境構築
- `pipenv install`

## script
- `pipenv run 00` : `00_my_first_app.py`を実行
- `pipenv run 03` : `03_ai_chat_app.py`を実行
- `pipenv run 04` : `04_ai_chat_app.py`を実行

## ローカルのベクトルDB(qdrant)確認
- 10で使う
- `sqlite3 ./local_qdrant/collection/my_collection/storage.sqlite`で入る
- `.tables`でテーブル表示(pointsというテーブルができているはず)
- `select * from points;`で全部見れる

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

### pipenvの`.env`の扱い
```python
# main.py
import os
from dotenv import load_dotenv
print(os.getenv("OPENAI_API_KEY"))
load_dotenv("./.env")
print(os.getenv("OPENAI_API_KEY"))
```
- このコードを`python main.py`とグローバル環境で実行すると以下のように`load_dotenv`実行前は読み込まれない
```
None
sk...
```
- このコードを`pipenv run python main.py`とpipenv仮想環境で実行すると以下のように`.env`の内容がデフォルトで読み込まれる
```
sk...
sk...
```

### langchainメモ
#### Document Loader
- いろいろなデータ形式のものを読み込める機能
- 06ではYoutubeを読み込む
- 他にも以下読み込める
  - データ形式 : csv, html, json, pdf, excel, word, powerpoint, ...
  - サービス : youtube, twitter, slack, discord, figma, notion, google drive, arxiv, ...
  - クラウドサービス : S3, GCS, BigQuery, ...
- `load()`メソッドでソースからドキュメントを読み込む
- `load_and_split()`メソッドでソースからドキュメントを読み込みテキスト分割器を使用してチャンクサイズに分割
- 得られる`Document`は`page_content`に生のテキストデータがあり`metadata`にテキストに関するメタデータが保存されている
#### `load_summarize_chain()`
- 以下のようにして使う
```python
chain = load_summarize_chain(
    llm,  # e.g. ChatOpenAI(temperature=0)
    chain_type="stuff",
    verbose=True,
    prompt=PROMPT
)
```
- chain_typeは以下の3通り
  - `stuff` : 最も基本的なchain_typeで与えられたDocumentをそのまま処理
  - `map_reduce` : 複数のDocumentを個別に要約して、それらの結果をまとめて最後に全体の要約を生成
  - `refine` : 分割されている文書を最初から順に処理して、要約した文章と次の文章をあわせて再度要約するという方式
- chain_typeを`stuff`に指定すると上のように`prompt`引数にプロンプトを指定する
- chain_typeを`map_reduce`に指定すると下のように`map_prompt`引数に各documentを要約する時のプロンプトを与えて、`combine_prompt`に個別の要約をつなげたものを最後に要約するときのプロンプトを与える

#### RetrievalQA
- langchainのchain
- ベクトルDBからどのように検索するかをretrieverに指定
- モデルをllmに指定

### streamlitメモ
#### `st.chat_input`
- 入力を監視できる
```python
if user_input := st.chat_input("入力してね"):
  # 入力されるとここ実行
```
#### `st.session_state`
- 状態を保存できる
- key, value構造
- 今回は過去のチャット内容を保存するために使用
#### `st.spinner`
- 以下のようにしてローディング表示できる
```python
with st.spinner("ChatGPT is typing..."):
  response = 
```
#### `st.sidebar`
- サイドバー表示