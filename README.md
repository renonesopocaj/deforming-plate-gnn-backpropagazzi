# deforming-plate-gnn-backpropagazzi

1. Add a folder `raw_data` under `src` with the dataset
2. Run `python main_data.py` to generate processed training data. See `dataconfig.yaml`.
3. Run `python main_valid_data.py` to generate validation data transformed with normalization fitted on the training data
4. Run `python train_egnn.py` to train egnn. See `config_egnn.yaml`
5. Run `python train_gunet.py` to train egnn. See `config_gunet.yaml`
6. Run `data_exploration.py` after having generated data to explore them
7. Run `visualization_simulation.py` to run rollout on the different models EGNN or Graph-U-Net