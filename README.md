# EvolveURE
This is an implementation of EvolveURE. Following this code, you can easily reproduce EvolveURE and conduct detailed further ablation experiments. This link is currently anonymous and will be made public if the paper is accepted. At that time, more detailed questions about this work will be welcomed.

# Environment
- python 3.10.4
- torch 1.12.1
- numpy 1.23.1
- tqdm 4.64.0

# Dataset
The datasets used in our paper are NYTaxi (https://opendata.cityofnewyork.us/) and ChicagoTaxi (https://data.cityofchicago.org/). The preprocessed data have been uploaded to Baidu Netdisk (Link：https://pan.baidu.com/s/1dkkMjjXN5CHJyS3wqWExbQ Code：1234), which can be downloaded and directly used as input.

# Pre-training (Stage 1)
- Train on ChicagoTaxi:
   ```python train_phase1.py --data=ChicagoTaxi```
- Train on NYTaxi:
   ```python train_phase1.py --data=NYTaxi```

# Downstream (Stage 2)
- Crime Prediction (Taking ChicagoTaxi as an example):
   ```python train_phase1.py --emb_path=./output/phase1/results/ChicagoTaxi_EvolveURE.pkl --data=Chicago```
- Travel time Estimation (Taking ChicagoTaxi as an example):
   ```python travel_time_estimation.py --emb_path=./output/phase1/results/ChicagoTaxi_EvolveURE.pkl --data=ChicagoTaxi```
