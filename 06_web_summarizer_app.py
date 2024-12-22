import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
from langchain_community.callbacks.manager import get_openai_callback
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

def init_page():
    st.set_page_config(
        page_title="Website Summarizer",
        page_icon="🤗"
    )
    st.header("Website Summarizer 🤗")
    st.sidebar.title("Options")

def init_messages():
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant.")
        ]
        st.session_state.costs = []

def select_model():
    model = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4"))
    if model == "GPT-3.5":
        model_name = "gpt-3.5-turbo"
    else:
        model_name = "gpt-4"

    return ChatOpenAI(temperature=0, model_name=model_name)

def get_url_input():
    url = st.text_input("URL: ", key="input")
    return url

# URLの検証
def validate_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def get_content(url):
    try:
        with st.spinner("Fetching Content ..."):
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")
            # fetch text from main
            if soup.main:
                return soup.main.get_text()
            elif soup.article:
                return soup.article.get_text()
            else:
                return soup.body.get_text()
    except:
        st.write("something wrong")
        return None

# 先頭1000文字だけ取得
def build_prompt(content, n_chars=300):
    return f""""以下はとあるWebページのコンテンツである。内容を{n_chars}程度でわかりやすく要約してください。
========

{content[:1000]}

========

日本語で書いてね！
"""

# 出力取得
def get_answer(llm, messages):
    with get_openai_callback() as cb:
        answer = llm(messages)
    return answer.content, cb.total_cost

def main():
    init_page()

    llm = select_model()
    init_messages()

    container = st.container()
    response_container = st.container()

    with container:
        url = get_url_input() # urlを入力から取得
        is_valid_url = validate_url(url) # urlを検証
        if not is_valid_url:
            st.write("Please input valid url")
            answer = None
        else:
            content = get_content(url) # text取得
            if content:
                prompt = build_prompt(content) # prompt作成
                st.session_state.messages.append(HumanMessage(content=prompt))
                with st.spinner("ChatGPT is typing ..."):
                    answer, cost = get_answer(llm, st.session_state.messages)
                st.session_state.costs.append(cost)
            else:
                answer = None
    # containerの外
    # 出力表示
    if answer:
        with response_container:
            st.markdown("## Summary")
            st.write(answer) # LLMの出力を表示
            st.markdown("---")
            st.markdown("## Original Text")
            st.write(content) # 入力の際に取得したテキスト表示
    
    # コスト表示
    costs = st.session_state.get('costs', [])
    st.sidebar.markdown("## Costs")
    st.sidebar.markdown(f"**Total cost: ${sum(costs):.5f}**")
    for cost in costs:
        st.sidebar.markdown(f"- ${cost:.5f}")
    
if __name__ == "__main__":
    main()