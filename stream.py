
#importing required libraries

import streamlit as st
import streamlit.components.v1 as comp
import matplotlib.pyplot as plt
import numpy as np
import mpld3

st.set_page_config(layout="wide")
st.title('Lottery Ticket Inspector')


#st.session_state.Experiment = '71'

def plot():
    st.write(st.session_state['Experiment'])

st.radio(
    label="Experiment",
    options=("exp1", "exp2"),
    index=0,
    on_change=plot,
    key="Experiment",
)

col1, col2, col3 = st.columns(3)

with col1:
   st.header("A cat")
   st.image("https://static.streamlit.io/examples/cat.jpg")

with col2:
   st.header("A dog")
   st.image("https://static.streamlit.io/examples/dog.jpg")

with col3:
   st.header("An owl")
   st.image("https://static.streamlit.io/examples/owl.jpg")


# #plotting the figure
# st.pyplot(fig, )

# fig_html = mpld3.fig_to_html(fig)
# comp.html(fig_html, height=600)