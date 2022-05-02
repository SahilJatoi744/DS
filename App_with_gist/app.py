import streamlit as st
from streamlit_embedcode import github_gist

link = "https://gist.github.com/SahilJatoi744/084f510ff3d39669f3c06fe1013ca401"

st.write("Embed github gist: ")

github_gist(link)