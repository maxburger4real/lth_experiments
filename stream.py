
#importing required libraries

import streamlit as st
import matplotlib
import pathlib
import os
import my_figures as mf
import pickle


st.set_page_config(layout="wide")
st.title('Lottery Ticket Inspector')

#@st.cache(hash_funcs={matplotlib.figure.Figure: lambda _: None})
@st.cache_data()
def get_fig(path):
   st.image(path)
   return fig

@st.cache_data()
def get_img(path):
   with open(path, "rb") as f:
      fig = pickle.load(f)
   return fig

def plot():
    st.write(st.session_state['Experiment'])

def fetch(selected):

   selection = {key: experiments[key] for key in selected}

   for col, (name, experiment) in zip(
      st.columns(len(selected)),
      selection.items()
      ):

      # move the hparams to the side
      container = st.sidebar.container()
      container.markdown(f'## {name.replace("_", " ")}')
      _expander = container.expander('Script', expanded=False)
      _expander.text(experiment.get_script())
      _expander = container.expander('Hyperparameters', expanded=False)
      _expander.json(experiment.get_hparams())

      col.header(name)
      for name, path in experiment.png_generator():
      #for name, path in experiment.unpickle_generator():
         
         #fig = get_fig(path)
         #fig.suptitle("")
         # fig.set_size_inches(16, 9)
         expander = col.expander(name.replace("_", " "))
         #expander.pyplot(fig)
         expander.image(path.as_posix(), output_format="png")



# find all available experiments, distplay in multiselect
p = pathlib.Path('./open_lth_data')
paths = [f for f in p.iterdir() if f.is_dir() if f.name.startswith("experiment")]
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
      expander = col.expander(f'Script', expanded=False)
      expander.text(experiment.get_script())
      expander = col.expander(f'Hyperparameters', expanded=False)
      expander.json(experiment.get_hparams())

      
# #plotting the figure
# st.pyplot(fig, )

# fig_html = mpld3.fig_to_html(fig)
# comp.html(fig_html, height=600)