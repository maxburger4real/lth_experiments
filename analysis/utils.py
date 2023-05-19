"""This module contains utility functions for experiments and plots."""

import os
import torch
import numpy as np
import matplotlib as mpl
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
import torchvision
import matplotlib.pyplot as plt
   
class direct_access_cmap():
    """
    A abstraction for a discrete colormap to make it easier to access color.
    The color is spaced evenly between 0 and n.
    usage example:
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
    """Retrieve a heatmap of the number of remaining weights per input feature."""
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
    """Retrieve a histogram of the number of weights connected to a neuron."""

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
    """Plot the distribution of given weights."""
    
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
    """Generate a label that includes valuable information for plotting."""
    return f"lvl:{level} remain:{1-e.get_sparsity(level):.2f} acc:{e.get_metrics(level)[2][-1]:.2f}"

def get_remain_and_prune_idx(x, m)-> tuple:
    """Returns the newly pruned weights with the new mask."""


    remain = (x * m).nonzero(as_tuple=True)
    prune = (x * ~m).nonzero(as_tuple=True)

    assert x.nonzero(as_tuple=True)[0].numel() == remain[0].numel() + prune[0].numel(), (x.nonzero().numel(),  remain[0].numel(), prune[0].numel())
    return remain, prune, x[remain], x[prune]

def get_std_mnist_mean_mnist():
    """Create pixelwise statistics of MNIST Dataset."""

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
    """Create a Plotgrid defined by number of plots and columns, accessible via a generator over the axes."""
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
