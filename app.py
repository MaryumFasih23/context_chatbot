import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# -------- Load local flan-t5-base model --------
model_id = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
llm = HuggingFacePipeline(pipeline=pipe)

# -------- Embedding model (use same every time) --------
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# -------- Always rebuild vectorstore to avoid mismatch --------
loader = TextLoader("data/my_notes.txt")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

db = FAISS.from_documents(docs, embedding_model)

# -------- LangChain with memory --------
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=db.as_retriever(),
    memory=memory
)

# -------- Streamlit Chat UI --------
st.set_page_config(page_title="Local Chatbot", layout="wide")
st.title("💬 Context-Aware Chatbot (flan‑t5‑base, Local)")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Ask me anything from the document...")
if user_input:
    result = qa_chain({"question": user_input})
    answer = result["answer"]
    st.session_state.chat_history.append((user_input, answer))

for user_q, bot_a in st.session_state.chat_history:
    st.markdown(f"**You:** {user_q}")
    st.markdown(f"**Bot:** {bot_a}")
