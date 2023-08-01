import os
from typing import TextIO

import openai
import pandas as pd
import streamlit as st
from langchain.agents import create_csv_agent, create_pandas_dataframe_agent
from langchain.llms import OpenAI

# Set your OpenAI API key here (replace YOUR_OPENAI_API_KEY with the actual key)
openai.api_key = "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"

def get_answer_csv(df1: pd.DataFrame, df2: pd.DataFrame, query: str) -> str:
    """
    Returns the answer to the given query by querying two CSV files.

    Args:
    - df1 (pd.DataFrame): the first CSV file as a Pandas dataframe to query.
    - df2 (pd.DataFrame): the second CSV file as a Pandas dataframe to query.
    - query (str): the question to ask the agent.

    Returns:
    - answer (str): the answer to the query from the CSV files.
    """
    # Create agents using OpenAI and the Pandas dataframes
    agent1 = create_csv_agent(OpenAI(temperature=0), df1, verbose=False)
    agent2 = create_csv_agent(OpenAI(temperature=0), df2, verbose=False)

    # Run the agents on the given query and return the answers
    answer1 = agent1.run(query)
    answer2 = agent2.run(query)

    # Combine the answers from both CSV files
    answer = f"Answer from CSV File 1: {answer1}\nAnswer from CSV File 2: {answer2}"
    return answer

st.header("Chat with CSV Data")

# Replace "path_to_csv_file_1" and "path_to_csv_file_2" with the actual paths of your CSV files
df1 = pd.read_csv("final.csv")
df2 = pd.read_csv("final_1.csv")

query = st.text_area("Ask any question related to the CSV data")
button = st.button("Submit")
if button:
    result = get_answer_csv(df1, df2, query)
    st.write(result)
