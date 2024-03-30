import streamlit as st
import openai
from langchain.document_loaders import NotionDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import re

def receive_main_text(documents):
    doc_main_body = []
    for doc in documents:
        header_start = doc.page_content.split('Підтримати')[-1].split('Neformat.com.ua ©')[0]
        up_to_site_mention = re.sub(r'\xa0|&a|quot;|lt;|amp;', '\n', header_start).strip()
        up_to_site_mention = up_to_site_mention.replace("\n ", "").replace(" \n", "")
        up_to_site_mention = re.sub(r'[\t\r\f]+', ' ', up_to_site_mention)
        normalised = re.sub(r'\n{2,}', '\n\n', up_to_site_mention)
        doc.page_content = normalised
        doc_main_body.append(doc)
    return doc_main_body

# Load OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

loader = WebBaseLoader("https://www.neformat.com.ua/ua/articles/moyi-12-rokiv-na-neformatcomua-i-ne-tilki.html")

content = loader.load()

main_body = receive_main_text(content)

# Split Notion content into smaller chunks
r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=25,
    separators=["\n\n", "\n", "(?<=\. )", " ", ""],
    length_function=len
)

splits = r_splitter.split_documents(main_body)

# Initialize OpenAI embedding model
embeddings = OpenAIEmbeddings()

# Convert all chunks into vectors embeddings using OpenAI embedding model
# Store all vectors in FAISS index and save locally to 'faiss_index'
db = FAISS.from_documents(splits, embeddings)
db.save_local("faiss_index")

print('Local FAISS index has been successfully saved.')