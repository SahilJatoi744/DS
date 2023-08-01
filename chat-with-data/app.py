# type: ignore
import streamlit as st
import pandas as pd

from utils import get_answer_csv

st.header("Chat with CSV Data")

df1 = pd.read_csv("final.csv")
df2 = pd.read_csv("final_1.csv")

query = st.text_area("Ask any question related to the CSV data")
button = st.button("Submit")
if button:
    result = get_answer_csv(df1, df2, query)
    st.write(result)
