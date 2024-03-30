import time
import streamlit as st
from utils import load_chain

# Configure streamlit page
st.set_page_config(
    page_title="Чатбот, що допоможе дізнатися більше про Neformat.com.ua"
)

# Initialize LLM chain in session_state
if 'chain' not in st.session_state:
    # To avoid repeated loading attempts
    st.session_state['chain_loaded'] = True
    st.session_state['chain'] = load_chain()

# Initialize chat history
if 'messages' not in st.session_state:
    # Start with first message from assistant
    st.session_state['messages'] = [{"role": "assistant",
                                     "content": "Привіт! Мене створили, щоб я допомагав тобі краще зрозуміти статтю про історію видання Neformat - твого гайду локальною сценою."}]

# Display chat messages from history on app rerun
for message in st.session_state.get('messages', []):
    if message["role"] == 'assistant':
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    else:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat logic
if query := st.chat_input("Запитай мене про Neformat.com.ua"):
    # Add user message to chat history
    st.session_state.setdefault('messages', []).append({"role": "user", "content": query})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        # Send user's question to chain
        result = st.session_state['chain']({"question": query})
        response = result['answer']
        full_response = ""

        # Simulate stream of response with milliseconds delay
        for chunk in response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    # Add assistant message to chat history
    st.session_state.setdefault('messages', []).append({"role": "assistant", "content": response})
