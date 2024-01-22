import logging
import time
import os
import sys
import argparse

import pandas as pd
import torch
import numpy as np
import pickle
from pathlib import Path
import random
import tqdm
import shutil
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error

config = {
    "NYC": {
        "crime_data_path": "./data/NYTaxi/2012_crime_byday.csv",
        "od_path": "./data/NYTaxi/od_matrix.npy",
        "emb_path": "./output/phase1/results/NYTaxi_EvolveURE.pkl",
        "start_day": 121,
        "slices": 48,
        "step": 1,
        "train_day": 42,
        "val_day": 7,
        "test_day": 14,
        "n_nodes": 180,
        "batch_size": 5
    },
    "Chicago": {
        "crime_data_path": "./data/ChicagoTaxi/2016_crime_byday.csv",
        "od_path": "./data/ChicagoTaxi/od_matrix.npy",
        "emb_path": "./output/phase1/results/ChicagoTaxi_EvolveURE.pkl",
        "start_day": 121,
        "slices": 48,
        "step": 1,
        "train_day": 42,
        "val_day": 7,
        "test_day": 14,
        "n_nodes": 77,
        "batch_size": 5
    }
}

### Argument and global variables
parser = argparse.ArgumentParser('dynamic task prediction')
parser.add_argument('--data', type=str, help='Dataset name (eg. NYTaxi or BJMetro)',
                    default='Chicago')
parser.add_argument('--type', type=str, default="crime", help='task type（eg. crime)')
parser.add_argument('--emb_path', type=str, help='Embedding Dir',
                    default='./output/phase1/results/ChicagoTaxi_EvolveURE.pkl')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--suffix', type=str, default='EvolveURE', help='Suffix to name the checkpoints')
parser.add_argument('--best', type=str, default='', help='Path of the best model')
parser.add_argument('--n_epoch', type=int, default=1000, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--patience', type=int, default=20, help='Patience for early stopping')
parser.add_argument('--device', type=str, default="cuda:0", help='Idx for the gpu to use: cpu, cuda:0, etc.')
parser.add_argument('--task', type=str, default="time", help='what task (eg. static/time/random')
parser.add_argument('--loss', type=str, default="mse", help='Loss function')
parser.add_argument('--emb_dim', type=int, default="144", help='Embedding dimension')

class EarlyStopMonitor(object):
    def __init__(self, max_round=3, higher_better=False, tolerance=1e-10):
        self.max_round = max_round
        self.num_round = 0

        self.epoch_count = 0
        self.best_epoch = 0

        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check(self, curr_val, epoch):
        self.epoch_count = epoch
        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1

        return self.num_round >= self.max_round, (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance

class PredictionLayer(nn.Module):
    def __init__(self, embedding_dim, n_nodes, embeddings, step, device):
        super(PredictionLayer, self).__init__()
        # (days, n_node, emb_dim)
        self.embeddings = embeddings
        self.step = step
        self.device = device
        self.n_nodes = n_nodes
        self.mlp = nn.Sequential(
            nn.Linear(self.step * embedding_dim, (self.step * embedding_dim) // 2),
            # drop
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Linear((self.step * embedding_dim) // 2, (self.step * embedding_dim) // 4),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Linear((self.step * embedding_dim) // 4, (self.step * embedding_dim) // 8),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Linear((self.step * embedding_dim) // 8, 1),
        )

    def forward(self, batch_data):
        """
        day || crime_counts
        :param batch_data: (batch, 1 + n_nodes)
        :return:
        """
        batch_data = batch_data.numpy()
        days = batch_data[:, 0].astype(np.int32)
        source_embeddings = []
        for day in days:
            # (n_node, steps, emb_dim)
            emb = self.embeddings[day - self.step: day, ...].transpose(1, 0, 2).reshape(self.n_nodes, -1)
            source_embeddings.append(emb)
        source_embeddings = np.stack(source_embeddings, axis=0).squeeze()
        source_embeddings = torch.Tensor(source_embeddings).to(self.device)
        if len(source_embeddings.shape) == 2:
            source_embeddings = source_embeddings.unsqueeze(0)
        return self.mlp(source_embeddings).squeeze(axis=-1)


def get_od_data(config):
    slices = config['slices']
    step = config['step']
    start_day = config['start_day']
    train_day = config["train_day"]
    val_day = config["val_day"]
    test_day = config["test_day"]
    n_nodes = config["n_nodes"]

    full_days = train_day + val_day + test_day

    embeddings = pickle.load(open(config["emb_path"], "rb"))
    embeddings = embeddings["embeddings"]
    emb = np.zeros((full_days, embeddings.shape[1], embeddings.shape[2]))
    # agg by day
    for day in range(full_days):
        next_day = day + 1
        emb[day] = np.mean(embeddings[day * slices: next_day * slices, ...], axis=0)

    full_data = pd.read_csv(config[f"{config['type']}_data_path"]).iloc[start_day:full_days + start_day]

    date = pd.to_datetime(full_data['date'], format='%Y-%m-%d', errors='coerce').dt.weekday.values

    assert date.shape[0] == emb.shape[0], 'the dimension between embedding and data is unaligned'
    assert (full_data.shape[1] - 1) == emb.shape[1], 'the num of regions is not equal'

    emb = np.concatenate((emb, date[:, np.newaxis, np.newaxis] * np.ones((1, n_nodes, 1))), axis=2)

    embedding_dim = emb.shape[2]

    full_data = full_data.values

    full_data = np.concatenate((np.arange(full_days)[:, np.newaxis], full_data[:, 1:]), axis=1).astype(np.float32)

    train_set = full_data[step: train_day, :]
    val_set = full_data[train_day: train_day + val_day, :]
    test_set = full_data[train_day + val_day:, :]

    print("train_set type", type(train_set))
    print("train_set shape", train_set.shape)

    data_loaders = {"train": DataLoader(train_set, shuffle=True, batch_size=config["batch_size"], drop_last=False),
                    "val": DataLoader(val_set, shuffle=False, batch_size=config["batch_size"], drop_last=False),
                    "test": DataLoader(test_set, shuffle=False, batch_size=config["batch_size"], drop_last=False)}

    return n_nodes, embedding_dim, step, emb, data_loaders


def load_embedding(embedding_dir):
    # Get all embedding
    embedding_files = [f for f in os.listdir(embedding_dir) if f.endswith('.npy')]

    # sort by timestamp
    embedding_files.sort()

    embeddings = []
    # Iterate through each .npy file and load them
    for embedding_file_name in embedding_files:
        embedding_file_path = os.path.join(embedding_dir, embedding_file_name)
        data = np.load(embedding_file_path)
        embeddings.append(data)
    return np.array(embeddings)

def calculate_metrics(stacked_prediction, stacked_label):
    stacked_prediction[stacked_prediction < 0] = 0
    reshaped_prediction = stacked_prediction.reshape(-1)
    reshaped_label = stacked_label.reshape(-1)
    mse = mean_squared_error(reshaped_prediction, reshaped_label)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(reshaped_prediction, reshaped_label)
    pcc = np.corrcoef(reshaped_prediction, reshaped_label)[0][1]
    smape = np.mean(2 * np.abs(reshaped_prediction - reshaped_label) / (
            np.abs(reshaped_prediction) + np.abs(reshaped_label) + 1))
    return (mse, rmse, mae, pcc, smape)

def calculate_above_average_metrics(stacked_prediction, stacked_label):
    stacked_prediction[stacked_prediction < 0] = 0
    reshaped_prediction = stacked_prediction.reshape(-1)
    reshaped_label = stacked_label.reshape(-1)
    truth_mean = np.mean(reshaped_label)
    print("truth_mean:", truth_mean)
    mask = reshaped_prediction >= truth_mean
    if len(reshaped_prediction[mask]) >0:
        mse = mean_squared_error(reshaped_prediction[mask], reshaped_label[mask])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(reshaped_prediction[mask], reshaped_label[mask])
        pcc = np.corrcoef(reshaped_prediction[mask], reshaped_label[mask])[0][1]
        smape = np.mean(2 * np.abs(reshaped_prediction[mask] - reshaped_label[mask]) / (np.abs(reshaped_prediction[mask]) + np.abs(reshaped_label[mask]) + 1))
        return (mse, rmse, mae, pcc, smape)
    else:
        return 0

def predict_dynamic_task(args):
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    NUM_EPOCH = args.n_epoch
    device = args.device
    DATA = args.data
    LEARNING_RATE = args.lr

    Path(f"./dynamic_task_prediction/{args.task}/saved_models/").mkdir(parents=True, exist_ok=True)
    Path(f"./dynamic_task_prediction/{args.task}/saved_checkpoints/").mkdir(parents=True, exist_ok=True)
    MODEL_SAVE_PATH = f"./dynamic_task_prediction/{args.task}/saved_models/{args.data}_{args.suffix}.pth"
    get_checkpoint_path = lambda epoch: f"./dynamic_task_prediction/{args.task}/saved_checkpoints/{args.data}_{args.suffix}_{epoch}.pth"
    results_path = f"./dynamic_task_prediction/{args.task}/results/{args.data}_{args.suffix}.pkl"
    Path(f"./dynamic_task_prediction/{args.task}/results/").mkdir(parents=True, exist_ok=True)

    ### set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    Path(f"./dynamic_task_prediction/{args.data}_{args.suffix}_{args.task}/log/").mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(f"./dynamic_task_prediction/{args.data}_{args.suffix}_{args.task}/log/{str(time.time())}_{args.data}_{args.suffix}.log")
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(args)

    config[DATA]['data'] = DATA
    config[DATA]["task"] = args.task
    config[DATA]['emb_dim'] = args.emb_dim
    config[DATA]["type"] = args.type

    if args.emb_path != "":
        print("assigned embedding")
        config[DATA]["emb_path"] = args.emb_path
    # Extract data for training, validation and testing
    n_nodes, embedding_dim, step, emb, data_loaders = get_od_data(config[DATA])

    model = PredictionLayer(embedding_dim=embedding_dim, n_nodes=n_nodes, embeddings=emb,
                            step=step, device=device)

    if args.loss == "mse":
        criterion = torch.nn.MSELoss()
    else:
        criterion = torch.nn.MSELoss()

    model = model.to(device)

    val_mses = []
    epoch_times = []
    total_epoch_times = []
    train_mses = []
    if args.best == "":
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        early_stopper = EarlyStopMonitor(max_round=args.patience, higher_better=False)
        ifstop = False
        for epoch in range(NUM_EPOCH):
            print("================================Epoch: %d================================" % epoch)
            start_epoch = time.time()
            logger.info('start {} epoch'.format(epoch))
            for phase in ["train", "val"]:
                label, prediction = [], []
                if phase == "train":
                    model = model.train()
                else:
                    model = model.eval()
                batch_range = tqdm.tqdm(data_loaders[phase])
                for batch_data in batch_range:
                    real_data = batch_data[:, 1:].numpy().astype("float")
                    predicted_data = model(batch_data)
                    if phase == "train":
                        optimizer.zero_grad()
                        loss = criterion(predicted_data, torch.Tensor(real_data).to(device))
                        loss.backward()
                        optimizer.step()
                        batch_range.set_description(f"train_loss: {loss.item()};")
                    label.append(real_data)
                    prediction.append(predicted_data.cpu().detach().numpy())
                concated_label = np.concatenate(label)
                concated_prediction = np.concatenate(prediction)
                metrics = calculate_metrics(concated_prediction, concated_label)
                logger.info(
                    'Epoch {} {} metric: mse, rmse, mae, pcc, smape, {}, {}, {}, {}, {}'.format(epoch, phase, *metrics))
                if phase == "train":
                    train_mses.append(metrics[0])
                elif phase == "val":
                    val_mses.append(metrics[0])
                    # Early stopping
                    ifstop, ifimprove = early_stopper.early_stop_check(metrics[0], epoch)
                    if ifstop:
                        logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
                        logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
                        logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
                    else:
                        logger.info('No improvement over {} epochs'.format(early_stopper.num_round))
                        torch.save(
                            {"statedict": model.state_dict()},
                            get_checkpoint_path(epoch))
            if ifstop:
                break
            total_epoch_time = time.time() - start_epoch
            total_epoch_times.append(total_epoch_time)
            logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))

        # Save temporary results
        pickle.dump({
            "val_mses": val_mses,
            "train_losses": train_mses,
            "epoch_times": epoch_times,
            "total_epoch_times": total_epoch_times
        }, open(results_path, "wb"))

        logger.info('Saving dynamic_task_prediction model')
        shutil.copy(get_checkpoint_path(early_stopper.best_epoch), MODEL_SAVE_PATH)
        logger.info('dynamic_task_prediction model saved')
        best_model_param = torch.load(get_checkpoint_path(early_stopper.best_epoch))
    else:
        best_model_param = torch.load(args.best)

    # load model parameters, memories from best epoch on val dataset
    model.load_state_dict(best_model_param["statedict"])
    # Test
    print("================================Test================================")

    model = model.eval()
    batch_range = tqdm.tqdm(data_loaders["test"])
    label, prediction = [], []
    for batch_data in batch_range:
        predicted_data = model(batch_data)
        real_data = (batch_data[:, 1:]).numpy().astype("float")
        label.append(real_data)
        prediction.append(predicted_data.cpu().detach().numpy())
    concated_label = np.concatenate(label)
    concated_prediction = np.concatenate(prediction)

    test_metrics = calculate_metrics(concated_prediction, concated_label)
    above_average_metrics = calculate_above_average_metrics(concated_prediction, concated_label)

    logger.info(
        'Test statistics:-- mse: {}, rmse: {}, mae: {}, pcc: {}, smape:{}'.format(*test_metrics))

    logger.info(
        'Above Average Test statistics:-- mse: {}, rmse: {}, mae: {}, pcc: {}, smape:{}'.format(*above_average_metrics))
    # Save results for this run
    pickle.dump({
        "val_mses": val_mses,
        "test_mse": test_metrics[0],
        "test_rmse": test_metrics[1],
        "test_mae": test_metrics[2],
        "test_pcc": test_metrics[3],
        "test_smape": test_metrics[4],
        "epoch_times": epoch_times,
        "train_losses": train_mses,
        "total_epoch_times": total_epoch_times,
        "label": concated_label,
        "prediction": concated_prediction
    }, open(results_path, "wb"))


if __name__ == '__main__':
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    if args.suffix == '':
        args.suffix = args.task

    predict_dynamic_task(args)

