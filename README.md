# つくりながら学ぶ！AIアプリ開発入門 - LangChain & StreamlitによるChatGPT API徹底活用

## ソース
- https://zenn.dev/ml_bear/books/d1f060a3f166a5

## 環境構築
- `pipenv install`

## script
- `pipenv run 00` : `00_my_first_app.py`を実行
- `pipenv run 03` : `03_ai_chat_app.py`を実行
- `pipenv run 04` : `04_ai_chat_app.py`を実行

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