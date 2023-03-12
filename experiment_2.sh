#!/bin/zsh
python open_lth/open_lth.py lottery \
--levels=10 \
--dataset_name=mnist \
--batch_size=128 \
--batchnorm_init=uniform \
--model_name=mnist_lenet_300_100 \
--model_init=kaiming_normal \
--optimizer_name=adam \
--lr=1.2e-3 \
--training_steps=2ep \
--pruning_strategy=sparse_global \
--pruning_fraction=0.333 \
--data_order_seed=42