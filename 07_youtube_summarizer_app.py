import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
from langchain_community.callbacks.manager import get_openai_callback
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
# from langchain.document_loaders import YoutubeLoader
from langchain_community.document_loaders import YoutubeLoader
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain

def init_page():
    st.set_page_config(
        page_title="Youtube Summarizer",
        page_icon="🤗"
    )
    st.header("Youtube Summarizer 🤗")
    st.sidebar.title("Options")
    st.session_state.costs = []


def select_model():
    model = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4"))
    if model == "GPT-3.5":
        model_name = "gpt-3.5-turbo"
    else:
        model_name = "gpt-4"

    return ChatOpenAI(temperature=0, model_name=model_name)


def get_url_input():
    url = st.text_input("Youtube URL: ", key="input")
    return url

def get_document(url):
    with st.spinner("Fetching Content ..."):
        loader = YoutubeLoader.from_youtube_url(
            url,
            add_video_info=True, # タイトルや再生数も取得
            language=['en', 'ja'] # 英語->日本語の優先順位で字幕を取得
        )
        return loader.load() # Document

# データ要約
def summarize(llm, docs):
    prompt_template = """Write a concise Japanese summary of the following transcript of Youtube Video.
    
{text}

ここから日本語で書いてね。必ず3段落以内の200文字以内で簡潔にまとめること:
"""
    PROMPT = PromptTemplate(template=prompt_template, input_variable=["text"])

    with get_openai_callback() as cb:
        chain = load_summarize_chain(
            llm,
            chain_type="stuff",
            verbose=True,
            prompt=PROMPT,
        )
        response = chain({"input_documents": docs}, return_only_outputs=True)
    return response['ouput_text'], cb.total_cost

def main():
    init_page()
    llm = select_model()

    container = st.container()
    response_container = st.container()

    with container:
        if url := get_url_input():
            document = get_document(url)
            with st.spinner("ChatGPT is typing ..."):
                output_text, cost = summarize(llm, document)
            st.session_state.costs.append(cost)
        else:
            output_text = None

    if output_text:
        with response_container:
            st.markdown("## Summary")
            st.write(output_text)
            st.markdown("---")
            st.markdown("## Original Text")
            st.write(document)

    costs = st.session_state.get('costs', [])
    st.sidebar.markdown("## Costs")
    st.sidebar.markdown(f"**Total cost: ${sum(costs):.5f}**")
    for cost in costs:
        st.sidebar.markdown(f"- ${cost:.5f}")

if __name__ == '__main__':
    main()