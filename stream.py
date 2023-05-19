"""This script runs the streamlit webapp. It can be executed from the command line with:
streamlit run stream.py
"""
import streamlit as st
import pathlib
import os
import analysis.figures as mf
import pickle


# Globals
datapath = pathlib.Path('../open_lth_data')


@st.cache_data()
def get_img(path):
   """Function that uses caching for faster reloading"""
   with open(path, "rb") as f:
      fig = pickle.load(f)
   return fig

def fetch(selected):
   """Function that populates the columns of the dashboard after a change."""
   selection = {key: experiments[key] for key in selected}

   for col, (name, experiment) in zip(
      st.columns(len(selected)),
      selection.items()
      ):

      # move the hparams to the side
      container = st.sidebar.container()
      container.markdown(f'## {name.replace("_", " ")}')
      _expander = container.expander('Hyperparameters', expanded=False)
      _expander.json(experiment.get_hparams())

      col.header(name)
      for name, path in experiment.png_generator():
         expander = col.expander(name.replace("_", " "))
         expander.image(path.as_posix(), output_format="png")


st.set_page_config(layout="wide")
st.title('Lottery Ticket Inspector')


# find all available experiments, distplay in multiselect
paths = [f for f in datapath.iterdir() if f.is_dir() if f.name.startswith("experiment")]
names = [ex.name.replace("experiment_", "") for ex in paths]

# create the experiment objects used to obtain figures.
experiments = {
    name : mf.MNIST_LENET_300_100_Experiment(os.path.join(path, "replicate_1"))
    for name, path in zip(names, paths)
}

selected = st.sidebar.multiselect("Experiments to see", names)

if not selected:
   st.text("Select an Experiment in the Sidebar.")
   st.stop()

selection = {key: experiments[key] for key in selected}


if st.sidebar.button(
   label="Fetch Plots for selection."
):
   fetch(selected)
else:
   for col, (name, experiment) in zip(
      st.columns(len(selected)),
      selection.items()
      ):

      col.header(name)
      expander = col.expander(f'Hyperparameters', expanded=False)
      expander.json(experiment.get_hparams())