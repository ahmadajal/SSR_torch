# SSR_torch

Implementation of Smooth Surface Regularization in pytorch.

- `smooth_loss.py`: implementation of the SSR loss.
- `train_ssr.py`: script with the configuration to train the netwrok.
- `demo_train_ssr`: the scripts that calls the train function in `train_ssr.py` and modifies the configs for training. The results of each epoch are saved in a log file.
- `ssr_trained_ACC.ipynb`: notebook containing the accuracy of the SSR trained network together with its explanation for a single example.
