import streamlit as st
import openai
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate

openai.api_key = st.secrets["OPENAI_API_KEY"]

@st.cache_resource
def load_chain():
    """
    The `load_chain()` function initializes and configures a conversational retrieval chain for
    answering user questions.
    :return: The `load_chain()` function returns a ConversationalRetrievalChain object.
    """

    # Load OpenAI embedding model
    embeddings = OpenAIEmbeddings()

    # Load OpenAI chat model
    llm = ChatOpenAI(temperature=0)

    # Load local FAISS index as a retriever
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Create memory 'chat_history'
    memory = ConversationBufferWindowMemory(k=3,memory_key="chat_history")

    # Create the Conversational Chain
    chain = ConversationalRetrievalChain.from_llm(llm,
                                                  retriever=retriever,
                                                  memory=memory,
                                                  get_chat_history=lambda h : h,
                                                  verbose=True)

    # Create system prompt
    template = """Ти асистент, який допомагає користувачам дізнатись більше про видання Neformat.com.ua. Використовуй подані тексти як контекст, що відповісти на запитання. 
    Якщо ти не знаєш відповіді, відповідай 'На жаль, я не знаю відповіді 😔'. Не вигадуй інформацію. Використовуй максимум три речення. 
    Ти можеш допомагати користувачу відповідями на питання тільки про Neformat.com.ua та його історію. 
    Якщо питання не за цими темами, відповідай 'Я не можу Вам допомогти з цим запитом. Запитайте мене щось про історію видання Neformat.com.ua'
    Будь ввічливим.
    {context}
    Запитання: {question}
    Змістовна відповідь:"""

    # Add system prompt to chain
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template)
    chain.combine_docs_chain.llm_chain.prompt.messages[0] = SystemMessagePromptTemplate(prompt=QA_CHAIN_PROMPT)

    return chain
