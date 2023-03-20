import torch
import torch.nn.functional as F
import torchvision
import pathlib
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Polygon
import numpy as np
from scipy.spatial import ConvexHull
import pandas as pd
import json
import pickle


cmap_mnist = mpl.cm.get_cmap("coolwarm").reversed()
cmap_mnist.set_bad("black")


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
        hparams = {}
        with open(path, "r") as f:
            for line in f.readlines():
                parts = line.split("=>")
                if len(parts) > 1:
                    l, r = parts
                    key = l.split("*")[1].strip()
                    val = r.strip()
                    hparams[key] = val

        return hparams

    def get_script(self):
        parent = pathlib.Path(self.path).parent
        scripts = [f for f in os.listdir(parent) if f.endswith('.sh')]
        if len(scripts) != 1:
            raise ValueError('should be only one script file in the current directory')

        script = scripts[0]

        with open(script, "r") as f:
            return f.read()
        
class direct_access_cmap():
    """
    A abstraction for a discrete colormap to make it easier to access color.
    The color is spaced evenly between 0 and n.
    usage:
        cmap = direct_access_cmap(10)
        color = cmap(5)
        color = cmap(10000) # saturated
        color = cmap(-1000) # saturated
    """
    def __init__(self, n, style="cool"):
        self.n = n
        self.cmap = mpl.cm.get_cmap(style, n)    # a discrete colormap
    def __call__(self, idx):
        return self.cmap(idx/self.n)
    
    @property
    def scalarmappable(self):
        return mpl.cm.ScalarMappable(cmap=self.cmap)

# UTILITIES

def get_pruned_and_remaining_weights(x_t, m_t_plus_1)-> tuple:
    """Returns the newly pruned weights with the new mask."""
    X=x_t
    m=m_t_plus_1
    removed_weights_idx = X[~m].nonzero(as_tuple=False)
    removed_weights = X[~m][removed_weights_idx].squeeze()
    #removed_weights.size(dim=0)
    assert removed_weights.all(), "all newly pruned weights must be nonzero! there might be coincidences where a unpruned weights is actually 0. but highly unlikely"
    return removed_weights, X[m]

def find_dead_features(X, shape, ax , title="", cmap=None):
    X = X.reshape(-1, 28,28)
    N = X.shape[0]
    nonzero = torch.count_nonzero(X, dim=0)

    if cmap == None:
        cmap = mpl.colormaps.get_cmap("turbo").copy()
    cmap.set_bad(color="k")

    ax.set_title(f"{title}, \n {range_ltx(nonzero)}")
    masked_array = np.ma.array (nonzero, mask= (nonzero==0))
    im = ax.imshow(masked_array, interpolation="nearest", cmap=cmap, vmin=0, vmax=N)

    return im

def find_dead_neurons(X, ax,title="", cmap=None):
    nonzero = torch.count_nonzero(X, dim=1)

    if cmap == None:
        cmap = mpl.colormaps.get_cmap("turbo").copy()
    cmap.set_bad(color="k")

    ax.set_title(f"{title}, \n {range_ltx(nonzero)}")
    valrange = nonzero.max() - nonzero.min() + 1
    ax.hist(nonzero, bins=valrange)

def range_ltx(x):
    return f"values $\in [{x.min().item()}, {x.max().item()}]$"

def plot_weight_distribution(x, ax, label="", title="", include_zeros=False, curves=False, density=True, color=None):
    #frequency, bins = torch.histogram(x, bins=10)
    
    x = torch.flatten(x)
    x = torch.where(x != 0, x, torch.nan)
    ax.title.set_text(title)
    if curves: # a lot faster
        x = x[np.isfinite(x)]
        counts, bins = np.histogram(x, bins=64, density=density)
        ax.plot(bins[:-1], counts, label=label, color=color)
        ax.legend()
    else:
        print("not implemented....")
        #ax.hist(x, bins=64, histtype="step")
    
    # ax.plot(data[0])
    
def morph_distributions(distlist, density=True, ax=None, fig=None, cmap=None, title=""):
    """Create a Plot overlappinng a list of densities normalized on a new fig if not provided."""
    cmap = direct_access_cmap(len(distlist))
    # a discrete colormap

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6,6), constrained_layout=True)

    curves=True
    for i, (dist, label) in enumerate(distlist):
        plot_weight_distribution(
            x=dist,
            ax=ax,
            label=label,
            title=title,
            include_zeros=False,
            curves=curves,
            density=density, # unnormalized makes no sense
            color=cmap(i+1) 
        )
    fig.suptitle(f'Morph Distributions {title}', fontsize=16)

def points_to_convvex_hull_polygon(points):
    """Turn a array of points, shape (N,2) into a a polygon of its complex hull."""
    points = np.array(points)
    hull = ConvexHull(points)
    cent = np.mean(points, axis=0)
    pts = points[hull.vertices]
    k = 1
    return Polygon(
        k*(pts - cent) + cent, # for adding some padding around it. 
        closed=True,
        capstyle='round',
        facecolor="none",
        edgecolor="red",
        linewidth=2,
        alpha=0.2)

def get_outer_shell_2d_matrix(input, pad=1):
    """Get the outer shell of a 2d matrix including batch dimension.
    (N,X,Y) -> shell = (N*pad*pad), core = N*(X-pad)*(Y-pad))
    """
    import torch.nn.functional as F
    N, x, y = input.shape
    source = torch.zeros((x-2*pad,y-2*pad), dtype=torch.bool)
    mask = F.pad(input=source, pad=(pad,pad,pad,pad), mode='constant', value=True)
    mask = mask.reshape(1,x,y)
    mask = np.tile(mask.T, N).T

    assert mask.shape == input.shape

    return input[mask], input[~mask]

def std_label(level, e):
    return f"lvl:{level} remain:{1-e.get_sparsity(level):.2f} acc:{e.get_metrics(level)[2][-1]:.2f}"

def get_remain_and_prune_idx(x, m)-> tuple:
    """Returns the newly pruned weights with the new mask."""


    remain = (x * m).nonzero(as_tuple=True)
    prune = (x * ~m).nonzero(as_tuple=True)

    assert x.nonzero(as_tuple=True)[0].numel() == remain[0].numel() + prune[0].numel(), (x.nonzero().numel(),  remain[0].numel(), prune[0].numel())
    return remain, prune, x[remain], x[prune]

def get_std_mnist_mean_mnist():
    try:
        std_mnist = torch.load('std_mnist.pt')
        mean_mnist = torch.load('mean_mnist.pt')

    except:
        train_set = torchvision.datasets.MNIST(
            train=True,
            root=os.path.join("open_lth_datasets", 'mnist'),
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=[0.1307], std=[0.3081])
                ]
            )
        )

        trainloader = torch.utils.data.DataLoader(
            train_set,
            batch_size=60000,
            shuffle=False,
            num_workers=2
        )

        # get the first batch, aka all images.
        mnist = next(iter(trainloader))[0].squeeze()    

        # calculate mean and stddev for each feature. replace 0 with nan.
        std_mnist, mean_mnist  = torch.std_mean(mnist, axis=0)
        std_mnist[std_mnist == 0] = torch.nan
        
        # save because it is unchanging and takes ~15 seconds to compute.
        torch.save(std_mnist, 'std_mnist.pt')
        torch.save(mean_mnist, 'mean_mnist.pt')
    return std_mnist, mean_mnist

def plotgrid(n_plots=1, n_cols=None, individual_plot_size=(6,6)):
    if n_cols is None or n_cols > n_plots: n_cols = n_plots
    n_rows = n_plots // n_cols + (0 if n_plots % n_cols == 0 else 1)
    x_size, y_size = individual_plot_size
    fig, ax = plt.subplots( 
        n_rows, n_cols,
        figsize=(n_cols*x_size,n_rows*y_size), 
        constrained_layout=True
    )

    def ax_generator():
        for i in range(n_plots):
            row, col = i // n_cols , i % n_cols 
            if n_rows == 1:
                yield ax[col]
            else:
                yield ax[row, col]

    

    return fig, ax_generator(), ax

""" 
- - - - FIGURES - - - - 
These are functions that receive an Experiment 
Object as input and output a Figure object.
"""
def fig_remaining_connections_of_neurons(e: MNIST_LENET_300_100_Experiment, n_cols=3):

    fig, ax_generator, axs = plotgrid(e.num_levels, n_cols, (4,4))

    # colormap
    cmap = mpl.colormaps.get_cmap("turbo").copy().reversed()
    cmap.set_bad(color="k", alpha=1)

    # populate grid with plots
    for level, ax in enumerate(ax_generator):
        find_dead_neurons(
            e.weights(level, layer=0),
            ax,
            title=std_label(level, e),
            cmap=cmap
        )



    return fig

def fig_weights_evolution_both_layers(
        e: MNIST_LENET_300_100_Experiment,
        layer: int
):
    fig, ax = plt.subplots(
        2, 1,
        figsize=(16,9),
        constrained_layout=True,
        sharey=True,
        sharex=True)

    density = True

    morph_distributions(
        distlist=[ (e.weights(level=i, layer=layer),std_label(i, e)) for i in range(e.num_levels-1)],
        ax=ax[0],
        fig=fig,
        title=f"remaining initial weights over all levels {'normalized' if density else 'unnormalized'}",
        density=density
    )
    morph_distributions(
        distlist=[ (e.weights(level=i, layer=layer, it=e.it, ep=e.ep), std_label(i, e)) for i in range(e.num_levels-1)],
        ax=ax[1],     
        fig=fig,
        title=f"trained weights over all levels {'normalized' if density else 'unnormalized'}",
        density=density
    )

    # View twice the range as min and max from weights at init
    _ = e.weights(level=0, layer=layer)
    ax[0].set_xlim(_.min()*2,_.max()*2)

    return fig

def fig_grid_input_features_connections_remaining(
        e: MNIST_LENET_300_100_Experiment,
        ncols = 3
):
    fig, ax_generator, axs = plotgrid(e.num_levels, ncols, (4,4))

    # colormap
    cmap = mpl.colormaps.get_cmap("turbo").copy().reversed()
    cmap.set_bad(color="k", alpha=1)

    # populate grid with plots
    for level, ax in enumerate(ax_generator):
        im = find_dead_features(
            e.weights(level=level, layer=0),
            (28,28), 
            ax,
            title=std_label(level, e),
            cmap=cmap
        )

    fig.colorbar(
            im,
            ax=axs,
            shrink=1.0,
            location="top"
        )

    return fig

def fig_histogram_of_padding_input_weights_vs_nonpadding_weights(
        e, pad=1, density=True, sharey=False):
    
    fig, ax_generator, axs = plotgrid(e.num_levels, 4, (4,4))

    for i, ax in enumerate(ax_generator):
        x = e.weights(level=i, layer=0).reshape(300,28,28)
        shell, core = get_outer_shell_2d_matrix(x, pad=pad)

        for x, name in zip([shell, core], ["shell", "core"]):

            plot_weight_distribution(x, ax, name, std_label(i,e), False, True, density)

    return fig

def fig_morph_distributions_shell_vs_core_over_levels(e, pad=1, sharey=False, density=True):
    fig, ax = plt.subplots(
    1, 2,
    figsize=(16,9), 
    constrained_layout=True, 
    sharey=sharey,               # you can see the differences in amount of weights, but the shape is lost
    )

    morph_distributions(
        [
        (get_outer_shell_2d_matrix(e.weights(level=i, layer=0).reshape(300,28,28), pad=pad)[0]
        , std_label(i, e)) 
        for i in range(e.num_levels)
        ],
        ax=ax[0],
        fig=fig,
        title="Shell of Input Layer Features",
        density=density
    )
    morph_distributions(
        [
        (get_outer_shell_2d_matrix(e.weights(level=i, layer=0).reshape(300,28,28), pad=pad)[1]
        , std_label(i, e)) 
        for i in range(e.num_levels)
        ],
        ax=ax[1],
        fig=fig,
        title="Core of Input Layer Features",
        density=density
    )
    return fig

def fig_scatter_of_survivors_in_before_after_weight_space_(
        e, 
        cmap="Set3",
        which=None, 
        show_pruned=True, 
        show_survived=True, 
        n_cols=None,
        skip_last=True
        ):

    fig, ax_generator, axs = plotgrid(e.num_levels, n_cols=n_cols)
    layer=0

    # initial trailing weights
    Y_ = e.weights(0, layer, ep=e.ep, it=e.it)

    #for level in range(0, e.num_levels-1):
    for level, ax in enumerate(ax_generator):

        if level == e.num_levels-1 and skip_last:
            continue # skip last level.

        X = e.weights(level, layer)
        Y = e.weights(level, layer, ep=e.ep, it=e.it)
        m = e.mask(level+1, layer)

        # used to index same shape as X
        _, _, X_remain, X_prune = get_remain_and_prune_idx(X, m)
        remain_idx, prune_idx, Y_remain, Y_prune = get_remain_and_prune_idx(Y, m)

        ax.set_title(f"level: {level}")

        # set all default to none. For each coloring, specify everything explicitly
        cr = mr = cp = cr = None

        xr=X_remain.flatten()
        yr=Y_remain.flatten()
        xp=X_prune.flatten()
        yp=Y_prune.flatten()

        if which == "absolute_movement":
            cr = np.abs(X_remain-Y_remain).flatten()
            mr = ","
            cp = "black"
            mp = "x"
        elif which == "previous_y":
            if level==0: 
                xr = X.flatten()
                yr = Y.flatten()
                cr = Y.flatten()
                cp = None
            else:
                cr = np.abs(Y_[remain_idx]).flatten()
                cp = "black"
                mr = ","
                mp = "x"

        elif which == "std_mnist":
            std_mnist = get_std_mnist_mean_mnist()[0]
            like_X = torch.tile(std_mnist.reshape(1,-1), dims=(300,1))
            
            # sort remaining weights
            cr = like_X[remain_idx].flatten()
            cr, idx = torch.sort(cr, descending=True)
            xr,yr = xr[idx], yr[idx]

            # sort pruned weights
            cp = like_X[prune_idx].flatten()
            cp, idx = torch.sort(cp, descending=True)
            xp,yp = xp[idx], yp[idx]

            mr = "o"
            mp = "o"
            cmap = cmap_mnist

        elif which == "color_previous_level":
            if level==0: 
                xr = X.flatten()
                yr = Y.flatten()
                cr = np.ones_like(X).flatten()
                cp = None
            else:
                cr = np.abs(Y_[remain_idx]-X_remain).flatten()
                mr = ","
                cp = "black"
                mp = "x"
        elif which == "only_pruned_previous_y":
            if level==0: continue
            cp = np.abs(Y_[prune_idx]).flatten()
            mp = ","
        else:
            print("illegal coloring scheme.")
            return

        if cp is not None and show_pruned:
            im = ax.scatter(
                x=xp,
                y=yp,
                marker=mp,
                c=cp,
                cmap=cmap,
                #alpha=0.5
                s=0.7
                )
            ax.set_ylim(Y_prune.min()*1.1, Y_prune.max()*1.1)
            ax.set_xlim(X.min()*1.1, X.max()*1.1)

        if cr is not None and show_survived:
            im = ax.scatter(
                x=xr,
                y=yr,
                marker=mr,
                c=cr,
                cmap=cmap,
                # alpha=0.5,
                s=0.7
                )
            ax.set_ylim(Y_remain.min()*1.1, Y_remain.max()*1.1)
            ax.set_xlim(X_remain.min()*1.1, X_remain.max()*1.1)
        

        # prune brorders
        y_prune_lower_bound, y_prune_upper_bound = Y_prune.min(),Y_prune.max()
        ax.axhline(y_prune_lower_bound, c="red")
        ax.axhline(y_prune_upper_bound, c="red")
        ax.axline((0, 0), (1, 1), linewidth=1, color='r')
        ax.set_title(std_label(level, e))
        ax.set_xlabel("initial")
        ax.set_ylabel("final")
        ax.grid()

        Y_ =Y

    fig.colorbar(
        im, 
        ax=axs,
        location="top"
    )

    return fig

def fig_average_statistics_overlap(e):

    std_mnist = get_std_mnist_mean_mnist()[0]
    like_X = torch.tile(std_mnist.reshape(1,-1), dims=(300,1)).nan_to_num()

    metrics = {
        "stddev" : {},
        "avg_movement" : {},
        "avg_magnitude_init" : {},
        "avg_magnitude_trained" : {},
        "avg_value_init" : {},
        "avg_value_trained" : {},
    }
    for metric in metrics.keys():
        metrics[metric] = {"survive":[], "dead":[]}

    fig, ax_generator, axs = plotgrid(len(metrics), 2, (4,4))
    layer = 0

    for level in range(e.num_levels-1):
        X = e.weights(level, layer)
        Y = e.weights(level, layer, ep=e.ep, it=e.it)
        m = e.mask(level+1, layer)
        _, _, X_remain, X_prune = get_remain_and_prune_idx(X, m)
        remain_idx, prune_idx, Y_remain, Y_prune = get_remain_and_prune_idx(Y, m)

        Y_prune = Y_prune.nan_to_num()
        Y_remain = Y_remain.nan_to_num()
        metrics["stddev"]["survive"] += [torch.mean(like_X[remain_idx])]
        metrics["stddev"]["dead"] += [torch.mean(like_X[prune_idx])]

        metrics["avg_movement"]["survive"] += [torch.mean(torch.abs(X_remain- Y_remain))]
        metrics["avg_movement"]["dead"] += [torch.mean(torch.abs(X_prune- Y_prune))]

        metrics["avg_magnitude_init"]["survive"] += [torch.mean(torch.abs(X_remain))]
        metrics["avg_magnitude_init"]["dead"] += [torch.mean(torch.abs(X_prune))]

        metrics["avg_magnitude_trained"]["survive"] += [torch.mean(torch.abs(Y_remain))]
        metrics["avg_magnitude_trained"]["dead"] += [torch.mean(torch.abs(Y_prune))]

        metrics["avg_value_init"]["survive"] += [torch.mean((X_remain))]
        metrics["avg_value_init"]["dead"] += [torch.mean((X_prune))]

        metrics["avg_value_trained"]["survive"] += [torch.mean((Y_remain))]
        metrics["avg_value_trained"]["dead"] += [torch.mean((Y_prune))]

    for ax, (title, x) in zip(ax_generator, metrics.items()):

        ax.plot(x["survive"], label="survive")
        ax.plot(x["dead"], label="dead")
        ax.set_xlabel("pruning levels")
        ax.set_title(title)
        ax.legend()

    return fig

def fig_convex_hull(experiment):
    fig, ax = plt.subplots(1,1,figsize=(16,9))

    cmap = direct_access_cmap(experiment.num_levels)    # a discrete colormap
    layer=0

    X = experiment.weights(experiment.num_levels-1, layer)
    Y = experiment.weights(experiment.num_levels-1, layer, ep=experiment.ep, it=experiment.it)
    ax.scatter(X,Y)

    for level in range(experiment.num_levels-1):
        
        X = experiment.weights(level, layer)
        Y = experiment.weights(level, layer, ep=experiment.ep, it=experiment.it)
        m = experiment.mask(level+1, layer)

        x_prune, _ = get_pruned_and_remaining_weights(X, m)
        y_prune, _ = get_pruned_and_remaining_weights(Y, m)

        points = torch.stack((x_prune, y_prune), dim=1)
        poly = points_to_convvex_hull_polygon(np.array(points))

        # prune brorders
        # x_up, x_low = x_prune.min(),x_prune.max()
        # y_up, y_low = y_prune.min(),y_prune.max()
        poly.set_edgecolor(cmap(level))
        poly.set_label(std_label(level, experiment))
        poly.set_fill("none")
        poly.set_linewidth(1)
        poly.set_alpha(1)

        ax.add_patch(
            poly
        )

        ax.set_xlabel("initial")
        ax.set_ylabel("final")

        ax.grid()
        ax.legend()
    return fig

def fig_scatter_of_survivors_in_before_after_weight_space__example_with_movement_discrete(e):
    fig, ax = plt.subplots(
        1, 1,
        figsize=(24,18), 
        constrained_layout=True, 
        sharey=False,               # you can see the differences in amount of weights, but the shape is lost
        )

    level=0
    layer=0

    X = e.weights(level, layer)
    Y = e.weights(level, layer, ep=e.ep, it=e.it)
    m = e.mask(level+1, layer)

    x_prune, x_remain = get_pruned_and_remaining_weights(X, m)
    y_prune, y_remain = get_pruned_and_remaining_weights(Y, m)

    

    im = ax.scatter(
        x=x_remain,
        y=y_remain,
        alpha=0.5,
        marker=",",
        c=np.abs(y_remain-x_remain),
        cmap="Set3"
        )
    
    ax.scatter(
        x=x_prune,
        y=y_prune,
        marker="x",
        c="black"
        )
    
    fig.colorbar(
        im, 
        ax=ax,
    )

    # prune brorders
    x_prune_lower_bound, x_prune_upper_bound = x_prune.min(),x_prune.max()
    y_prune_lower_bound, y_prune_upper_bound = y_prune.min(),y_prune.max()
    ax.axhline(y_prune_lower_bound, c="red")
    ax.axhline(y_prune_upper_bound, c="red")

    ax.set_xlabel("initial")
    ax.set_ylabel("final")
    ax.set_ylim(y_remain.min()*1.1, y_remain.max()*1.1)
    ax.set_xlim(x_remain.min()*1.1, x_remain.max()*1.1)
    ax.grid()

    return fig

def fig_plot_scatter_fan_in_fan_out(e):

        # display them next to each other
    fig, ax = plt.subplots(
        1, 1,
        figsize=(20,20), 
        constrained_layout=True, 
        sharey=True,               # you can see the differences in amount of weights, but the shape is lost
        sharex=True
        )


    for level in range(1, e.num_levels):

        l0 = e.weights(level, layer=0)
        l1 = e.weights(level, layer=1)
        
        ax.scatter(
            torch.count_nonzero(l0, dim=1),
            torch.count_nonzero(l1, dim=0)
            )
    ax.set_xlabel("fan_in")
    ax.set_ylabel("fan_out")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid()

    return fig

def fig_plot_distribution_fan_in_fan_out(e):

        # display them next to each other
    fig, ax = plt.subplots(
        2,1,
        figsize=(20,10), 
        constrained_layout=True, 
        sharey=True,               # you can see the differences in amount of weights, but the shape is lost
        )

    cmap = direct_access_cmap(e.num_levels-1)

    for level in range(1, e.num_levels):
        l0 = e.weights(level, layer=0)
        l1 = e.weights(level, layer=1)
        fan_in = torch.count_nonzero(l0, dim=1)
        ax[0].hist(
            fan_in,
            bins=np.arange(-0.5, 784.5, 1),
            density=False,
            color=cmap(level),
            alpha=0.5,
            label=std_label(level, e)
            )
        ax[0].set_title("Fan in Counts for every Neuron")

        ax[1].hist(
            torch.count_nonzero(l1, dim=0),
            density=False,
            bins=np.arange(-0.5, 100.5, 1),
            color=cmap(level),
            #alpha=0.3
            histtype="step",
            label=std_label(level, e)
            )
        ax[1].set_title("Fan in Counts for every Neuron")

    ax[0].legend()
    ax[1].legend()

    return fig

def fig_mnist_mean_stddev_heatmap():

    std_mnist, mean_mnist = get_std_mnist_mean_mnist()

    # "dead" input features are black.
    fig, ax = plt.subplots(1,2)
    cmap = cmap_mnist

    im = ax[0].imshow(std_mnist, cmap=cmap)
    ax[0].set_title("MNIST Standard deviation")
    fig.colorbar(
        im,
        ax=ax[0]
    )
    im= ax[1].imshow(mean_mnist, cmap=cmap)
    ax[1].set_title("MNIST mean")
    fig.colorbar(
        im,
        ax=ax[1]
    )
    return fig


if __name__ == "__main__":
    # run all the tests

    # Experiment class test
    e = MNIST_LENET_300_100_Experiment(
        pathlib.Path('open_lth_data/lottery_exp1/replicate_1')
    )
    assert e.weights(1,0)[~e.mask(1,0)].any() == False, "mask must not cover any non-zero"
    assert e.weights(0,0)[e.mask(0,0)].all() == True
    assert e.weights(2,2)[e.mask(2,2)].all() == True, "mask must cover all zeros"
    assert e.weights(2,2)[e.mask(1,2)].all() == False, "earliear mask does not cover all removed elements"

    e.hparams

    # shell core test
    pad = 1
    source = torch.zeros((26,26), dtype=torch.bool)
    test = F.pad(input=source, pad=(pad,pad,pad,pad), mode='constant', value=True)
    test = test.reshape(1,28,28)
    test = np.tile(test.T, 300).T
    shell, core = get_outer_shell_2d_matrix(test, pad=1)
    assert np.all(shell) == True, "The outer shell must be true!"
    assert np.any(core) == False, "The inner core must be false!"

    # test get pruned and remaining weights
    for level in range(e.num_levels-1):
        X = e.weights(level,0) # current weights
        m = e.mask(level+1,0) # next mask
        _m = e.mask(level,0) # current mask

        # get the weights that are pruned and the ones that remain
        pruned, remaining = get_pruned_and_remaining_weights(X,m)
        assert pruned.size(0) +remaining.size(0) == X[_m].numel(), (pruned.size(0), remaining.size(0), " summed must equal ", X[_m].numel())
