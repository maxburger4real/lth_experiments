""" 
- - - - FIGURES - - - - 
These are functions that receive an Experiment 
Object as input and output a Figure object.
"""

import torch
import torch.nn.functional as F
import pathlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from experiment_wrapper import MNIST_LENET_300_100_Experiment
from utils import *

cmap_mnist = mpl.cm.get_cmap("coolwarm").reversed()
cmap_mnist.set_bad("black")

def fig_remaining_connections_of_neurons(e: MNIST_LENET_300_100_Experiment, layer, n_cols=3):

    fig, ax_generator, axs = plotgrid(e.num_levels, n_cols, (4,4))

    # colormap
    cmap = mpl.colormaps.get_cmap("turbo").copy().reversed()
    cmap.set_bad(color="k", alpha=1)

    # populate grid with plots
    for level, ax in enumerate(ax_generator):
        find_dead_neurons(
            e.weights(level, layer),
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
        layer,
        cmap="Set3",
        which=None, 
        show_pruned=True, 
        show_survived=True, 
        n_cols=None,
        skip_last=True
        ):

    fig, ax_generator, axs = plotgrid(e.num_levels, n_cols=n_cols)

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
            assert layer==0, "This only works for input layer"
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
        ax.axhline(y_prune_lower_bound, c="black", linestyle="-")
        ax.axhline(y_prune_upper_bound, c="black", linestyle="-")
        ax.axline((0, 0), (1, 1), linewidth=1, color='black', linestyle="-")
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

def fig_convex_hull(experiment, layer):
    fig, ax = plt.subplots(1,1,figsize=(16,9))

    cmap = direct_access_cmap(experiment.num_levels)    # a discrete colormap

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
