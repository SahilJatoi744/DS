# type: ignore
import os
from typing import TextIO
import streamlit as st

import openai
import pandas as pd
import streamlit as st
from langchain.agents import create_csv_agent, create_pandas_dataframe_agent
from langchain.llms import OpenAI

#OPENAI_API_KEY = "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"

os.environ["OPENAI_API_KEY"] = "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"
def get_answer_csv(file: TextIO, query: str) -> str:
    """
    Returns the answer to the given query by querying a CSV file.

    Args:
    - file (str): the file path to the CSV file to query.
    - query (str): the question to ask the agent.

    Returns:
    - answer (str): the answer to the query from the CSV file.
    """
    # Load the CSV file as a Pandas dataframe
    # df = pd.read_csv(file)
    #df = pd.read_csv("titanic.csv")

    # Create an agent using OpenAI and the Pandas dataframe
    agent = create_csv_agent(OpenAI(temperature=0), file, verbose=False)
    #agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=False)

    # Run the agent on the given query and return the answer
    #query = "whats the square root of the average age?"
    answer = agent.run(query)
    return answer


st.header("Chat with any CSV")
uploaded_file = st.file_uploader("Upload a csv file", type=["csv"])

if uploaded_file is not None:
    query = st.text_area("Ask any question related to the document")
    button = st.button("Submit")
    if button:
        st.write(get_answer_csv(uploaded_file, query))


