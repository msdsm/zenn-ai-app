import streamlit as st
from langchain_community.callbacks.manager import get_openai_callback
from PyPDF2 import PdfReader
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

QDRANT_PATH = "./local_qdrant"
COLLECTION_NAME = "my_collection"

def init_page():
    st.set_page_config(
        page_title="Ask My PDF",
        page_icon="🤗",
    )
    st.sidebar.title("Nav")
    st.session_state.costs = []

def get_pdf_text():
    uploaded_file = st.file_uploader(
        label="Upload your PDF here",
        type="pdf", # pdfを指定
    )
    if uploaded_file:
        pdf_reader = PdfReader(uploaded_file)
        text = "\n\n".join([page.extract_text() for page in pdf_reader.pages])
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="text-embedding-ada-002",
            chunk_size=250,
            chunk_overlap=0,
        )
        return text_splitter.split_text(text)
    else:
        return None

def load_qdrant():
    client = QdrantClient(path=QDRANT_PATH)

    # コレクションすべて取得
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]

    # コレクションが存在しなければ作成
    if COLLECTION_NAME not in collection_names:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE), # embeddingの次元1536, 距離をコサイン類似度で定義
        )
        print("collection created")
    
    return Qdrant(
        client=client,
        collection_name=COLLECTION_NAME, # コレクション指定
        embeddings=OpenAIEmbeddings(), # embeddingモデル作成
    )

# ベクトルDBに追加
def build_vector_store(pdf_text):
    qdrant = load_qdrant() # インスタンス作成
    qdrant.add_texts(pdf_text) # textがembeddingに変換されてDBに追加される

def page_pdf_upload_and_build_vector_db():
    st.title("PDF Upload")
    container = st.container()
    with container:
        pdf_text = get_pdf_text()
        if pdf_text:
            with st.spinner("Loading PDF ..."):
                build_vector_store(pdf_text)

def page_ask_my_pdf():
    pass

def main():
    init_page()

    selection = st.sidebar.radio("Go to", ["PDF Upload", "Ask My PDF(s)"])
    # ページ遷移
    if selection == "PDF Upload":
        page_pdf_upload_and_build_vector_db()
    elif selection == "Ask My PDF(s)":
        page_ask_my_pdf()
    
    costs = st.session_state.get('costs', [])
    st.sidebar.markdown("## Costs")
    st.sidebar.markdown(f"**Total cost: ${sum(costs):.5f}**")
    for cost in costs:
        st.sidebar.markdown(f"- ${cost:.5f}")

if __name__ == "__main__":
    main()