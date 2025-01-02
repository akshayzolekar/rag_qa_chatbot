import chromadb

import streamlit as st
from streamlit_chat import message
from transformers import (pipeline, T5Tokenizer, T5ForConditionalGeneration)
import langchain
from langchain.chains import RetrievalQA
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.vectorstores import Chroma

from config import (LLM_MODEL_PATH, EMD_MODEL_PATH, MAX_CONTEXT_LENGTHS, LLM_MODEL, EMD_MODEL,
                    CHUNK_SIZE, PRE_PROCESS, TEMP_LLM, CHAR_IN_TOKEN, PERCENT_CONTEXT_LLM)
from ingest import persist_dir

# langchain.debug = True

st.set_page_config(layout="wide")

tokenizer = T5Tokenizer.from_pretrained(LLM_MODEL_PATH, legacy=False)
base_model = T5ForConditionalGeneration.from_pretrained(
    LLM_MODEL_PATH,
    device_map="auto",
)


@st.cache_resource
def llm_pipeline():
    pipe = pipeline(
        "text2text-generation",
        model=base_model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        # truncation=True,
        do_sample=True,
        temperature=TEMP_LLM,
        top_p=0.95
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)

    return local_llm


@st.cache_resource
def qa_llm(db_dir, chunk_size):
    llm = llm_pipeline()
    embeddings = SentenceTransformerEmbeddings(
        model_name=EMD_MODEL_PATH
    )
    client = chromadb.PersistentClient(path=db_dir)
    vector_db = Chroma(client=client, embedding_function=embeddings)

    # calculate k for the retriever
    max_new_token = MAX_CONTEXT_LENGTHS[LLM_MODEL]
    max_new_characters = max_new_token * CHAR_IN_TOKEN * PERCENT_CONTEXT_LLM
    k = max_new_characters / chunk_size
    k = int(k)
    if k < 1:
        raise ValueError

    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        verbose=False,
    )

    return qa, k


def process_answer(query: dict, db_dir: str, chunk_size: int):
    qa, k = qa_llm(db_dir, chunk_size)
    generated_text = qa(query)
    answer = generated_text["result"]

    return answer


def display_conversation(history):
    """display conversation history"""
    for i in range(len(history["generated"])):
        message(history["past"][i], is_user=True, key=str(i) + "_user")
        message(history["generated"][i], key=str(i))


def main(db_dir: str, chunk_size: int):
    st.markdown(
        "<h1 style='text-align: center; color: blue;'>Q&A Chatbot 📄</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<h3 style='text-align: center; color: grey;'>Built by Catalyst with ❤️</h3>",
        unsafe_allow_html=True
    )

    with st.expander("About the Chatbot"):
        st.markdown(
            """
            This is a generative AI powered Chatbot that responds to the questions about PDF 
            files.
            """
        )

    user_input = st.text_input("chat here", key="input")

    # initialize the session state for generated responses and past messages
    if "past" not in st.session_state:
        st.session_state["past"] = ["Hello there!"]
    if "generated" not in st.session_state:
        st.session_state["generated"] = ["I am ready to help you"]

    # search the database for a response based on user input and update session state
    if user_input:
        answer = process_answer({"query": user_input}, db_dir, chunk_size)
        st.session_state["past"].append(user_input)
        st.session_state["generated"].append(answer)

    # display conversation history using streamlit messages
    if st.session_state["generated"]:
        display_conversation(st.session_state)


if __name__ == "__main__":
    db_dire = persist_dir(model=EMD_MODEL, chunk=CHUNK_SIZE, pre_process=PRE_PROCESS)
    main(db_dire, CHUNK_SIZE)
