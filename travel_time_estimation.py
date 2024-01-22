import logging
import os
import time
import sys
import argparse
import torch
import numpy as np
import pickle
from pathlib import Path
import random
import tqdm
import shutil
from utils.utils import EarlyStopMonitor
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error

config = {
    "NYTaxi": {
        "data_path": "./data/NYTaxi/time_dataset.npy",
        "emd_path": "./output/phase1/results/NYTaxi_EvolveURE.pkl",
        "matrix_path": "./data/NYTaxi/od_matrix.npy",
        "day_cycle": 86400,
        "slice": 1800,
        "train_day": 42,
        "val_day": 7,
        "test_day": 14,
        "day_start": 0,
        "day_end": 48,
        "n_nodes": 180,
        "batch_size": 4096,
        "emb_dim": 144
    },
    "ChicagoTaxi": {
        "data_path": "./data/ChicagoTaxi/time_dataset.npy",
        "emd_path": "./output/phase1/results/ChicagoTaxi_EvolveURE.pkl",
        "matrix_path": "./data/ChicagoTaxi/od_matrix.npy",
        "day_cycle": 86400,
        "slice": 1800,
        "train_day": 42,
        "val_day": 7,
        "test_day": 14,
        "day_start": 0,
        "day_end": 48,
        "n_nodes": 77,
        "batch_size": 4096,
        "emb_dim": 144
    }
}

### Argument and global variables
parser = argparse.ArgumentParser('EvolveURE training')
parser.add_argument('--data', type=str, help='Dataset name (eg. NYTaxi or ChicagoTaxi)',
                    default='ChicagoTaxi')
parser.add_argument('--emd_path', type=str, help='Embedding Path',
                    default='./output/phase1/results/ChicagoTaxi_EvolveURE.pkl')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--suffix', type=str, default='EvolveURE', help='Suffix to name the checkpoints')
parser.add_argument('--best', type=str, default='', help='Path of the best model')
parser.add_argument('--n_epoch', type=int, default=2000, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--patience', type=int, default=20, help='Patience for early stopping')
parser.add_argument('--device', type=str, default="cuda:2", help='Idx for the gpu to use: cpu, cuda:0, etc.')
parser.add_argument('--task', type=str, default="time", help='what task')
parser.add_argument('--loss', type=str, default="mse", help='Loss function')
parser.add_argument('--needPrompt', type=str, default="false", help='need prompt')


class PredictionLayer(nn.Module):
    def __init__(self, embedding_dim, n_nodes, embeddings, slice, device):
        super(PredictionLayer, self).__init__()
        self.embeddings = embeddings
        self.slice = slice
        self.device = device
        self.n_nodes = n_nodes
        self.w = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 1, embedding_dim),
            nn.LeakyReLU(),
            nn.Linear(embedding_dim, int(embedding_dim / 2)),
            nn.LeakyReLU(),
            nn.Linear(int(embedding_dim / 2), 1),
        )

    def forward(self, batch_data):
        batch_data = batch_data.numpy()
        source_embeddings = self.embeddings[
            (batch_data[:, 2] // self.slice - 1).astype("int"), batch_data[:, 0].astype("int")]
        target_embeddings = self.embeddings[
            (batch_data[:, 2] // self.slice - 1).astype("int"), batch_data[:, 1].astype("int")]
        batch_embeddings = torch.tensor(
            np.concatenate([source_embeddings, target_embeddings, batch_data[:, 4:5]], axis=1), dtype=torch.float32).to(self.device)

        return self.w(batch_embeddings).squeeze()

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


def get_data(config):
    day_cycle = config["day_cycle"]
    train_day = config["train_day"]
    val_day = config["val_day"]
    test_day = config["test_day"]
    n_nodes = config["n_nodes"]
    embeddings = pickle.load(open(config["emd_path"], "rb"))
    if "embeddings" in embeddings.keys():
        embeddings = embeddings["embeddings"]
    else:
        embeddings = embeddings["embeddints"]

    full_steps, n_nodes, embedding_dim = embeddings.shape
    #  'src', 'dst', 'pu_timestamp', 'do_timestamp', 'travel_dis', 'travel_time' (only for chicago)
    full_data = np.load(config["data_path"])
    # filter row with nan
    full_data = full_data[~np.isnan(full_data).any(axis=1)]

    travel_time = full_data[:, 3] - full_data[:, 2] if "NY" in config['data'] else full_data[:, -1]

    full_data = full_data[np.logical_and(travel_time <= 2227, travel_time >= 86)]
    # full_data = full_data[travel_time >=86]
    start_time = full_data[:, 2]
    train_set = full_data[np.logical_and(start_time > config["slice"], start_time < train_day * day_cycle)]
    val_set = full_data[
        np.logical_and(start_time >= train_day * day_cycle, start_time < (train_day + val_day) * day_cycle)]
    test_set = full_data[np.logical_and(start_time >= (train_day + val_day) * day_cycle,
                                        start_time < (train_day + val_day + test_day) * day_cycle)]
    print("train_set type", type(train_set))
    print("train_set shape", train_set.shape)

    data_loaders = {"train": DataLoader(train_set, shuffle=True, batch_size=config["batch_size"], drop_last=False),
                    "val": DataLoader(val_set, shuffle=False, batch_size=config["batch_size"], drop_last=False),
                    "test": DataLoader(test_set, shuffle=False, batch_size=config["batch_size"], drop_last=False)}

    return n_nodes, embedding_dim, embeddings, data_loaders


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


def TTE(args):
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    NUM_EPOCH = args.n_epoch
    device = args.device
    DATA = args.data
    LEARNING_RATE = args.lr

    Path(f"./output/{args.task}/saved_models/").mkdir(parents=True, exist_ok=True)
    Path(f"./output/{args.task}/saved_checkpoints/").mkdir(parents=True, exist_ok=True)
    MODEL_SAVE_PATH = f"./output/{args.task}/saved_models/{args.data}_{args.suffix}.pth"
    get_checkpoint_path = lambda epoch: f"./output/{args.task}/saved_checkpoints/{args.data}_{args.suffix}_{epoch}.pth"
    results_path = f"./output/{args.task}/results/{args.data}_{args.suffix}.pkl"
    Path(f"./output/{args.task}/results/").mkdir(parents=True, exist_ok=True)

    ### set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    Path(f"./output/{args.task}/log/").mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(f"./output/{args.task}/log/{str(time.time())}_{args.data}_{args.suffix}.log")
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(args)

    if args.emd_path != "":
        print("assigned embedding")
        config[DATA]["emd_path"] = args.emd_path

    config[DATA]["task"] = args.task
    config[DATA]["data"] = DATA

    # Extract data for training, validation and testing
    n_nodes, embedding_dim, embeddings, data_loaders = get_data(config[DATA])

    model = PredictionLayer(embedding_dim=embedding_dim, n_nodes=n_nodes, embeddings=embeddings,
                            slice=config[DATA]["slice"], device=device)

    if args.loss == "mse":
        criterion = torch.nn.MSELoss()
    else:
        raise NotImplementedError

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
                    real_data = (batch_data[:, 3] - batch_data[:, 2] if "NY" in DATA else batch_data[:, -1]).numpy().astype("float")
                    predicted_data = model(batch_data)
                    if phase == "train":
                        optimizer.zero_grad()
                        loss = criterion(predicted_data, torch.Tensor(real_data).to(device))
                        loss.backward()
                        optimizer.step()
                        # m_loss.append(loss.item())
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
                    ifstop, ifimprove = early_stopper.early_stop_check(metrics[0])
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

        logger.info('Saving EvolveURE model')
        shutil.copy(get_checkpoint_path(early_stopper.best_epoch), MODEL_SAVE_PATH)
        logger.info('EvolveURE model saved')
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
        real_data = (batch_data[:, 3] - batch_data[:, 2]).numpy().astype("float")
        label.append(real_data)
        prediction.append(predicted_data.cpu().detach().numpy())
    concated_label = np.concatenate(label)
    concated_prediction = np.concatenate(prediction)
    test_metrics = calculate_metrics(concated_prediction, concated_label)

    logger.info(
        'Test statistics:-- mse: {}, rmse: {}, mae: {}, pcc: {}, smape:{}'.format(*test_metrics))
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
    TTE(args)
