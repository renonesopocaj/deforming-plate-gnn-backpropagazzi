# deforming-plate-gnn-backpropagazzi

1. Add a folder `raw_data` under `src` with the dataset. So it should be `src/raw_data/train.tfrecord` `src/raw_data/validation.tfrecord`
2. Run `python main_data.py` to generate processed training data. See `dataconfig.yaml`.
3. Run `python main_valid_data.py` to generate validation data transformed with normalization fitted on the training data
4. Run `python train_egnn.py` to train egnn. See `config_egnn.yaml`
5. Run `python train_gunet.py` to train egnn. See `config_gunet.yaml`
6. Run `data_exploration.py` after having generated data to explore them
7. Run `visualization_simulation.py` to run rollout on the different models EGNN or Graph-U-Net

Where to download data: `https://www.swisstransfer.com/d/2346dac9-18c4-4073-aeb3-c56ebaeb2217`