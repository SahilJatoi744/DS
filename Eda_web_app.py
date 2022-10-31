import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

# web app ka title 
st.markdown(''' # **Exploratory Data Analysis Web Application**
This app is developed by **babaAammar** ''')

# how to upload a file from pc 

with st.sidebar.header("Upload your dataset (.csv)"):
    upload_file=st.sidebar.file_uploader("Upload your dataset",type=["csv"])
    df=sns.load_dataset('titanic')
    st.sidebar.markdown("[Example CSV file](https://raw.githubusercontent.com/babaAammar/Eda_web_app/master/titanic.csv)")


# profiling report 

if upload_file is not None:
    @st.cache
    def load_csv():
        df=pd.read_csv(upload_file)
        return df
    df=load_csv()
    pr=ProfileReport(df,explorative=True)
    st.subheader("Input Data Profile")
    st.write(df)
    st.write('---')
    st.header("Profiling Report with Pandas")
    st_profile_report(pr)
else:
    st.info("Please upload a csv file")
    if st.button("Press to use Example Dataset"):
        @st.cache
        def load_dataset():
            a=pd.DataFrame(np.random.randn(100,5),columns=['a_col','b_col','c_col','d_col','e_col'])
            return a
        df=load_dataset()
        pr=ProfileReport(df,explorative=True)
        st.subheader("Input Data Profile")
        st.write(df)
        st.write('---')
        st.header("Profiling Report with Pandas")
        st_profile_report(pr)
    


