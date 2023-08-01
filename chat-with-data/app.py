import streamlit as st
import pandas as pd
import openai

# Set your OpenAI API key here (replace YOUR_OPENAI_API_KEY with the actual key)
openai.api_key = "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"

def get_answer_csv(df: pd.DataFrame, query: str) -> str:
    """
    Returns the answer to the given query by querying the CSV data.

    Args:
    - df (pd.DataFrame): the CSV data as a Pandas dataframe to query.
    - query (str): the question to ask the agent.

    Returns:
    - answer (str): the answer to the query from the CSV data.
    """
    # Convert the CSV data to text format
    csv_text = df.to_string(index=False, header=False)
    
    # Call the OpenAI API to get the answer
    response = openai.Completion.create(
        engine="text-davinci-002",  # Use appropriate engine like text-davinci-002, gpt-3.5-turbo, etc.
        prompt=f"{query}\nCSV data:\n{csv_text}",
        max_tokens=100
    )

    # Extract the answer from the API response
    answer = response['choices'][0]['text'].strip()
    return answer

st.header("Chat with CSV Data")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded CSV Data")
    st.write(df.head())

    query = st.text_area("Ask any question related to the data")
    button = st.button("Submit")
    if button:
        result = get_answer_csv(df, query)
        st.write(result)
