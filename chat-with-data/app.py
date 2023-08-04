import streamlit as st
from streamlit_chat import message
from langchain.document_loaders import UnstructuredCSVLoader
from langchain.chat_models import ChatOpenAI
from langchain.indexes import VectorstoreIndexCreator


llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
data = UnstructuredCSVLoader('final.csv')
index = VectorstoreIndexCreator().from_loaders([data])


def chat():
    question = st.text_input('Enter your query:')
    api_key = st.sidebar.text_input('API Key')

    if question != None and api_key != None:
        if api_key.startswith('sk-'):
            response = index.query(llm=llm, question=question, chain_type="stuff", api_key=api_key)
            message("Bot:", response)
        else:
            st.error('API key must start with "sk-"')


st.title('Chat your data')

chat()
