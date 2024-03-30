import streamlit as st
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import re
from preprocessing import receive_main_text

# Load OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Load web page
loader = WebBaseLoader("https://www.neformat.com.ua/ua/articles/moyi-12-rokiv-na-neformatcomua-i-ne-tilki.html")

# Preprocess content before creating embeddings
main_body = receive_main_text(loader.load())

# Split preprocessed text into chunks
r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=25,
    separators=["\n\n", "\n", "(?<=\. )", " ", ""],
    length_function=len
)

splits = r_splitter.split_documents(main_body)

# Initialize OpenAI embedding model
embeddings = OpenAIEmbeddings()

# Convert all splits into vectors embeddings using OpenAI embedding model
# Store all vectors in FAISS index and save locally to 'faiss_index'
db = FAISS.from_documents(splits, embeddings)
db.save_local("faiss_index")

print('Local FAISS index has been successfully saved.')
