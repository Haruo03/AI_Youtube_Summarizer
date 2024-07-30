import streamlit as st
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback

from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from langchain.vectorstores import Qdrant
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
import hmac


st.set_page_config(
    page_title="Youtube commentaitor",
    page_icon=""
)


QDRANT_PATH = "./local.qdrant"
COLLECTION_NAME = "./my_collection"

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("Password incorrect")
    return False


if not check_password():
    st.stop()  # Do not continue if check_password is not True.


def init_page():
    st.header("Youtubeコメンテーターくん")
    st.sidebar.title("Nav")
    st.session_state.costs = []


def select_model():
    model = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4"))
    if model == "GPT-3.5":
        st.session_state.model_name = "gpt-3.5-turbo"
    else:
        st.session_state.model_name = "gpt-4"

    st.session_state.max_token = OpenAI.modelname_to_contextsize(st.session_state.model_name) - 300
    return ChatOpenAI(temperature=0, model_name=st.session_state.model_name)


def get_url_input():
    url = st.text_input("Youtube URL: ", key="input")
    return url


def get_document(url):
    with st.spinner("動画の内容を確認しています ..."):
        loader = YoutubeLoader.from_youtube_url(
            url,
            add_video_info=True,
            language=['en', 'ja']
        )
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="text-embedding-ada-002",
            chunk_size=500,
            chunk_overlap=0,
        )
        return loader.load_and_split(text_splitter=text_splitter)


def load_qdrant():
    client = QdrantClient(path=QDRANT_PATH)

    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]

    if COLLECTION_NAME not in collection_names:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        print('collection created')

    return Qdrant(
        client=client,
        collection_name=COLLECTION_NAME,
        embeddings=OpenAIEmbeddings()
    )


def build_vector_store(url_docs):
    qdrant = load_qdrant()
    texts = [doc.page_content for doc in url_docs]
    qdrant.add_texts(texts)


def build_qa_model(llm):
    qdrant = load_qdrant()
    retriever = qdrant.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        verbose=True
    )


def ask(qa, query):
    with get_openai_callback() as cb:
        answer = qa(query)

    return answer, cb.total_cost


def main():
    init_page()

    st.title("気になる動画の内容を質問してみよう！")

    llm = select_model()

    url = get_url_input()
    if url:
        url_docs = get_document(url)
        if url_docs:
            with st.spinner("ドキュメントを読み込んでいます ..."):
                build_vector_store(url_docs)

            query = st.text_input("Query: ", key="query")
            if query:
                qa = build_qa_model(llm)
                with st.spinner("解答を作成しています ..."):
                    answer, cost = ask(qa, query)
                st.session_state.costs.append(cost)

                if answer:
                    st.markdown("## Answer")
                    st.write(answer)

    costs = st.session_state.get('costs', [])
    st.sidebar.markdown("## Costs")
    st.sidebar.markdown(f"**Total cost: ${sum(costs):.5f}**")
    for cost in costs:
        st.sidebar.markdown(f"- ${cost:.5f}")


if __name__ == '__main__':
    main()
