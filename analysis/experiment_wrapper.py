"""This module consists a Class that wraps a lottery ticket experiment."""

import os 
import json
import torch
import pickle
import pathlib

import pandas as pd

from utils import std_label

class MNIST_LENET_300_100_Experiment:
    """A Wrapper for an Experiment with the MNIST LENET 300 100 Model."""
    def __init__(self, experiment_path, layermap=None):
        self.path = experiment_path

        # count the number of directories, levels in the experiment
        subdirs = next(os.walk(self.path))[1]
        self.num_levels = len([x for x in subdirs if x.startswith("level")])
        self.state_dict = None
        self.current_path = None

        # the standard layermap of the mnist_lenet_300_100 experiment. 
        # these are the names in the .pth file
        if layermap is None:
            self.layermap = (
                "fc_layers.0",
                "fc_layers.1",
                "fc",
            )
        else:
            self.layermap = layermap

        self.num_layers = len(self.layermap)

        # parse out the number of iterations
        path = os.path.join(self.path, f'level_0')
        path = os.path.join(path, 'main')

        names = [name for name in os.listdir(path) if "model" in name]
        assert len(names) == 2, "Not yet possible for more than 2 .pth files automagically."
        for name in names:

            name = name.split(".")[0]
            _,ep,it = name.split("_")
            ep = ep.split("ep")[1]
            it = it.split("it")[1]
            if ep != 0:
                self.ep = int(ep)
            if it != 0:
                self.it = int(it)
        
        self.hparams = self.get_hparams()
    
    def unpickle_generator(self):
        """Returns all figure paths and their names in alphabetical order lazily."""

        path = pathlib.Path(os.path.join(self.path, "pickle"))

        # get paths and names for all the plots
        paths = [f for f in path.iterdir() if f.is_file()]
        names = [ex.name.replace(".pickle", "") for ex in paths]

        # create s sorted dict, not pretty but works.
        d = dict(sorted({name:path for name, path in zip(names, paths)}.items()))
        
        # go through all plots, load them and yield lazily
        for name, path in d.items():
            yield name, path

    def png_generator(self):
        """Returns all figure paths and their names in alphabetical order lazily."""

        path = pathlib.Path(os.path.join(self.path, "png"))

        # get paths and names for all the plots
        paths = [f for f in path.iterdir() if f.is_file()]
        names = [ex.name.replace(".png", "") for ex in paths]

        # create s sorted dict, not pretty but works.
        d = dict(sorted({name:path for name, path in zip(names, paths)}.items()))
        
        # go through all plots, load them and yield lazily
        for name, path in d.items():
            yield name, path

    def pickle(self, fig):
        """pickle a figure in the experiment folder."""
        if not fig._suptitle.get_text(): raise
        # convert figure suptitle to filename
        figname = "_".join(
            fig._suptitle.get_text()
            .replace(".","")
            .replace(",","")
            .split(" ")
            )
        figname += ".pickle"

        # create pickle folder in the experiment
        path = os.path.join(self.path, "pickle")
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)

        # dump the pickle
        path = os.path.join(path, figname)        
        with open(path, 'wb') as f: # should be 'wb' rather than 'w'
            pickle.dump(fig, f) 
        self.to_png(fig)
  
    def to_png(self, fig):
        if not fig._suptitle.get_text(): raise
        # convert figure suptitle to filename
        figname = "_".join(
            fig._suptitle.get_text()
            .replace(".","")
            .replace(",","")
            .split(" ")
            )
        path = os.path.join(self.path, "png")
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        path = os.path.join(path, figname)
        fig.savefig(path)

    def get_state_dict(self, level, ep=0, it=0):
        """Obtain the state dict of the experiment
          given the level and epochs iterations."""
        
        if level > self.num_levels:
            raise ValueError(f"There are only {self.num_levels} levels")

        level_name = f'level_{level}'
        path = os.path.join(self.path, level_name)
        path = os.path.join(path, 'main')
        path = os.path.join(path, f'model_ep{ep}_it{it}.pth')

        if self.current_path != path:
            self.state_dict = torch.load(path)

        return self.state_dict
    
    def get_metrics(self, level):
        """Returns the Metrics of the level. 
        index, loss, accuracy."""

        if level > self.num_levels:
            raise ValueError(f"There are only {self.num_levels} levels")

        level_name = f'level_{level}'
        path = os.path.join(self.path, level_name)
        path = os.path.join(path, 'main')
        path = os.path.join(path, f'logger')
        
        df = pd.read_csv(path, names=["metric", "it","value"])
        dff = df.pivot_table(
            index=['it'], 
            columns=['metric'],
            values='value'
        )
        # not useful.
        dff.drop(columns=["test_examples"])

        return dff.index.to_numpy(), dff["test_loss"].to_numpy(), dff["test_accuracy"].to_numpy()

    def get_sparsity(self, level):
        """Get the current sparsity in the level,
          calculated by sparsity report."""

        level_name = f'level_{level}'
        path = os.path.join(self.path, level_name)
        path = os.path.join(path, 'main')
        path = os.path.join(path, f'sparsity_report.json')
        with open(path, "r") as f:
            d = json.load(f)
        return 1 - d['unpruned'] / d['total'] 

    def map_layer_name(self, layer: int, bias=False):
        """Map a numeric value that indicates layer
        index starting at 0 (input layer) to name in state dict"""
        try:
            if bias:
                return f"{self.layermap[layer]}.bias"
            else:
                return f"{self.layermap[layer]}.weight"

        except:
            raise ValueError(f"There are only {len(self.layermap)} layers, not {layer}")
    
    def weights(self, 
            level: int, # the pruning level
            layer: int, # the layer of the Network
            it: int = 0,    # the iteration during the level
            ep: int = 0     # the epoch during level
            ):
        """Get the weights of a level in a layer 
        at certain point in level."""

        layer_name = self.map_layer_name(layer)
        state_dict = self.get_state_dict(level, ep, it)
        return state_dict[layer_name].cpu()
    
    def biases(self, 
            level: int, # the pruning level
            layer: int, # the layer of the Network
            it: int = 0,    # the iteration during the level
            ep: int = 0     # the epoch during level
            ):
        """Get the weights of a level in 
        a layer at certain point in level."""

        layer_name = self.map_layer_name(layer, bias=True)
        state_dict = self.get_state_dict(level, ep, it)
        return state_dict[layer_name].cpu()

    def mask(self, 
            level: int, # the pruning level
            layer: int, # the layer of the Network
            ):
        """ Obtain the mask of the current level. 
        Note: that the mask used in level 1 is the
        mask resulting from level 0.
        """
       
        level_name = f'level_{level}'
        path = os.path.join(self.path, level_name)
        path = os.path.join(path, 'main')
        path = os.path.join(path, f'mask.pth')
        mask = torch.load(path)

        layer_name = self.map_layer_name(layer)
        return mask[layer_name].cpu() == 1

    def get_hparams(self):
        """Obtain the hyperparams as a dictionary."""
        path = os.path.join(self.path, f'level_{0}')
        path = os.path.join(path, 'main')
        path = os.path.join(path, f'hparams.log')
        hparams = {
            "levels" : self.num_levels,
        }

        with open(path, "r") as f:
            for line in f.readlines():
                parts = line.split("=>")
                if len(parts) > 1:
                    l, r = parts
                    key = l.split("*")[1].strip()
                    val = r.strip()
                    hparams[key] = val

        hparams["metrics"] = [std_label(level, self) for level in range(self.num_levels)],


        return hparams

    def get_script(self):
        raise NotImplementedError("This function does not work properly yet.")

        parent = pathlib.Path(self.path).parent
        scripts = [f for f in os.listdir(parent) if f.endswith('.sh')]
        if len(scripts) != 1:
            raise ValueError('should be only one script file in the current directory')

        script = scripts[0]

        with open(script, "r") as f:
            return f.read()
     