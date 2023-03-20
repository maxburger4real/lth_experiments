import pathlib
import my_figures as mf
import os

def main():

    p = pathlib.Path('./open_lth_data')
    paths = [f for f in p.iterdir() if f.is_dir() if f.name.startswith("experiment")]
    names = [ex.name.replace("experiment_", "") for ex in paths]

    # create the experiment objects used to obtain figures.
    experiments = {
        name : mf.MNIST_LENET_300_100_Experiment(os.path.join(path, "replicate_1"))
        for name, path in zip(names, paths)
    }
    for experiment in experiments.values():

        for layer in [0,1]:
            fig = mf.fig_weights_evolution_both_layers(experiment, layer)
            fig.suptitle(f'Evolution of initial Weight distribution before and after training of Hidden Layer {layer}', fontsize=16)
            experiment.pickle(fig) 

        fig = mf.fig_remaining_connections_of_neurons(experiment)
        fig.suptitle('number of remaining weights for each Neuron Histogram.', fontsize=16)
        experiment.pickle(fig) 

        fig = mf.fig_grid_input_features_connections_remaining(experiment)
        fig.suptitle('Fraction of Pruned weights without weight updates', fontsize=16)
        experiment.pickle(fig) 

        fig = mf.fig_histogram_of_padding_input_weights_vs_nonpadding_weights(experiment, pad=2, density=True)
        fig.suptitle('Fraction of Pruned weights in the outer padded area', fontsize=16)
        experiment.pickle(fig) 

        fig = mf.fig_morph_distributions_shell_vs_core_over_levels(experiment, pad=2, density=True)
        fig.suptitle('Outer Shell versus Core of the MNIST features morph layer 0', fontsize=16)
        experiment.pickle(fig) 

        fig = mf.fig_scatter_of_survivors_in_before_after_weight_space_(
            experiment, 
            "viridis",
            which="absolute_movement",
            n_cols=2,
            skip_last=True
            )
        fig.suptitle("Survivors, color indicates absolute movement during training level", fontsize=30)
        experiment.pickle(fig) 

        fig = mf.fig_scatter_of_survivors_in_before_after_weight_space_(
            experiment, 
            "viridis", 
            which="color_previous_level")
        fig.suptitle("Survivors, color is absolute movement from previous level", fontsize=30)

        fig = mf.fig_scatter_of_survivors_in_before_after_weight_space_(
            experiment, 
            "viridis", 
            which="previous_y",
            n_cols=2,
            skip_last=True
        )
        fig.suptitle("Survivors, color is absolute value of y from previous level", fontsize=30)
        experiment.pickle(fig) 

        fig = mf.fig_scatter_of_survivors_in_before_after_weight_space_(
            experiment,
            "viridis", 
            which="only_pruned_previous_y",
            n_cols=2,
            skip_last=True
        )
        fig.suptitle("Survivors, color is absolute value of y from previous level", fontsize=30)
        experiment.pickle(fig) 


        fig = mf.fig_convex_hull(experiment)
        fig.suptitle("Convex Hull of the pruned weights and the weights remaining at last level.")
        experiment.pickle(fig) 


        fig = mf.fig_plot_scatter_fan_in_fan_out(experiment)
        fig.suptitle("Fan in vs Fan out of Neurons over levels.", fontsize=32)
        experiment.pickle(fig) 


        fig = mf.fig_plot_distribution_fan_in_fan_out(experiment)
        fig.suptitle("Fan in fan out distribution", fontsize=32)
        experiment.pickle(fig) 


        fig = mf.fig_mnist_mean_stddev_heatmap()
        fig.suptitle("MNIST Featurewise information, black is exactly 0", fontsize=12)
        experiment.pickle(fig) 


        fig = mf.fig_scatter_of_survivors_in_before_after_weight_space__example_with_movement_discrete(experiment)
        fig.suptitle("Intuition about the Weights, pruning and movement")
        experiment.pickle(fig) 


        fig = mf.fig_scatter_of_survivors_in_before_after_weight_space_(
            experiment, 
            cmap="viridis",
            which="std_mnist", 
            show_survived=False, 
            n_cols=3)
        fig.suptitle("Weights that are pruned, colored by variance of the input feature they observe.", fontsize=30)
        experiment.pickle(fig) 


        fig = mf.fig_scatter_of_survivors_in_before_after_weight_space_(
            experiment, 
            "viridis", 
            which="std_mnist", 
            show_pruned=False, 
            n_cols=3
            )
        fig.suptitle("Weights that survived, colored by variance of the input feature they observe.", fontsize=30)
        experiment.pickle(fig) 

        # Save figure handle to disk
        fig = mf.fig_average_statistics_overlap(experiment)
        fig.suptitle("Metrics")
        experiment.pickle(fig) 


if __name__ == "__main__":
    main()