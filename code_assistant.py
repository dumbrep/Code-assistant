import streamlit as st
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from streamlit import session_state as ss
import os

load_dotenv()
llm = ChatOllama(model = "codingo")

if 'chat_history' not in ss:
    ss.chat_history = []
if "store" not in ss:
    ss.store = {}
if "documents" not in ss:
    ss.documents = None  # To store processed documents
if "response" not in ss:
    ss.response = None


prompt_message = (
    "you are a helpfull code assistant"
    "User will give you questions related to coding"
    "he may say to generate code in specific language"
    "fullfill his requirements from your full potential"
    "\n\n"
)


prompt_template = ChatPromptTemplate.from_messages(
    [
    ("system", prompt_message),
    MessagesPlaceholder("chat_history"),
    ("user","{question}")
    ]
)

chain = prompt_template|llm

def get_session_history(session_id : str) -> BaseChatMessageHistory:
    if session_id not in ss.store:
        ss.store[session_id] = ChatMessageHistory()

    return ss.store[session_id]


conversational_chain = RunnableWithMessageHistory(
    chain,
    get_session_history=get_session_history,
    input_messages_key="question",
    output_messages_key="output",
    history_messages_key="chat_history"
)

st.set_page_config(page_title="Code Assistant", page_icon="💻")
st.title("Let's make coding easy, ask your question....")

session_id = st.sidebar.text_input("Enter session id")

if(session_id):
    user_input = st.text_input("Ask your question")
    if user_input:
        with st.spinner("Thinking........"):
            try:
                 # Invoke the chain and store the response in session state
                ss.response = conversational_chain.invoke(
                    {"question": user_input},
                    config={"configurable": {"session_id": session_id}}
                )

                # Debugging: Print the entire response to understand its structure
                content = ss.response.content
                lines = content.splitlines()
                for line in lines:
                    st.write(line)  # Print each line in              

            except Exception as e:  
                st.error(f"An error occurred: {e}")

