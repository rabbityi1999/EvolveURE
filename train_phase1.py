import logging
import time
import sys
import argparse
import torch
import numpy as np
import pickle
from pathlib import Path
import random
from tqdm import trange
import shutil
from evaluation.evaluation_phase1 import eval_phase1_prediction
from model.model import EvolveURE,Encoder,MLP_Predictor
from utils.utils import EarlyStopMonitor
from utils.data_processing import get_data
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler

config = {
    "NYTaxi": {
        "data_path": "./data/NYTaxi/NYCTaxi.npy",
        "matrix_path": "./data/NYTaxi/od_matrix.npy",
        "point_path": "./data/NYTaxi/back_points.npy",
        "input_len": 1800,
        "output_len": 1800,
        "day_cycle": 86400,
        "train_day": 42,
        "val_day": 7,
        "test_day": 14,
        "day_start": 0,
        "day_end": 86400,
        "sample": 1,
        "n_nodes": 180
    },
    "ChicagoTaxi": {
        "data_path": "./data/ChicagoTaxi/ChicagoTaxi.npy",
        "matrix_path": "./data/ChicagoTaxi/od_matrix.npy",
        "point_path": "./data/ChicagoTaxi/back_points.npy",
        "input_len": 1800,
        "output_len": 1800,
        "day_cycle": 86400,
        "train_day": 42,
        "val_day": 7,
        "test_day": 14,
        "day_start": 0,
        "day_end": 86400,
        "sample": 1,
        "n_nodes": 77
    }
}
### Argument and global variables
parser = argparse.ArgumentParser('EvolveURE training')
parser.add_argument('--data', type=str, help='Dataset name (eg. NYTaxi or ChicagoTaxi)',
                    default='NYTaxi')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--suffix', type=str, default='EvolveURE', help='Suffix to name the checkpoints')
parser.add_argument('--best', type=str, default='', help='Path of the best model')
parser.add_argument('--n_epoch', type=int, default=20000, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--patience', type=int, default=20, help='Patience for early stopping')
parser.add_argument('--device', type=str, default="cuda:1", help='Idx for the gpu to use: cpu, cuda:0, etc.')
parser.add_argument('--ratio', type=float, default=0.3, help='augment ratio')
parser.add_argument('--aug_method', type=str, default="scale", help='add sub time scale')
parser.add_argument('--message_dim', type=int, default=144, help='Dimensions of the messages')
parser.add_argument('--memory_dim', type=int, default=144, help='Dimensions of the memory for '
                                                               'each node')
parser.add_argument('--beta', type=float, default=0.999, help='ema ratio')
parser.add_argument('--lambs', type=float, nargs="+", default=[1.0], help='Lamb of different time scales')


def timestampaugment(head, tail , input_len, device, timestamps_batch, ratio):
    off = np.random.randn(len(timestamps_batch))
    new_timestamps = timestamps_batch + ratio * off * input_len
    return head, tail, new_timestamps

def get_embedding(args):
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    NUM_EPOCH = args.n_epoch
    device = args.device
    DATA = args.data
    LEARNING_RATE = args.lr
    MESSAGE_DIM = args.message_dim
    MEMORY_DIM = args.memory_dim

    input_len = config[DATA]["input_len"]
    output_len = config[DATA]["output_len"]
    day_cycle = config[DATA]["day_cycle"]
    day_start = config[DATA]["day_start"]
    day_end = config[DATA]["day_end"]

    Path("./output/phase1/saved_models/").mkdir(parents=True, exist_ok=True)
    Path("./output/phase1/saved_checkpoints/").mkdir(parents=True, exist_ok=True)
    MODEL_SAVE_PATH = f'./output/phase1/saved_models/{args.data}_{args.suffix}.pth'
    get_checkpoint_path = lambda epoch: f'./output/phase1/saved_checkpoints/{args.data}_{args.suffix}_{epoch}.pth'
    results_path = f"./output/phase1/results/{args.data}_{args.suffix}.pkl"
    Path("./output/phase1/results/").mkdir(parents=True, exist_ok=True)

    ### set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    Path("./output/phase1/log/").mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(f"./output/phase1/log/{str(time.time())}_{args.data}_{args.suffix}.log")
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(args)

    ### Extract data for training, validation and testing
    n_nodes, node_features, full_data, val_time, test_time, all_time, od_matrix_30, back_points = get_data(config[DATA])

    model = EvolveURE(device=device, n_nodes=n_nodes, node_features=node_features,
                      message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
                      output=output_len, lambs=args.lambs).to(device)

    for p in model.target_encoder.parameters():
        p.requires_grad = False

    val_losses = []
    total_epoch_times = []
    train_losses = []
    optimizer = torch.optim.Adam(list(model.online_encoder.parameters()) + list(model.online_predictor.parameters()) + list(model.mobility_reconstruct.parameters()) + list(model.reconstruct_mob.parameters()), lr=LEARNING_RATE)

    beta = args.beta
    aug_method = args.aug_method

    if args.best == "":
        early_stopper = EarlyStopMonitor(max_round=args.patience, higher_better=False)
        num_batch = val_time // input_len
        for epoch in range(NUM_EPOCH):
            print("================================Epoch: %d================================" % epoch)
            start_epoch = time.time()
            logger.info('start {} epoch'.format(epoch))
            m_loss = []

            model.online_encoder.init_memory()
            model.target_encoder.init_memory()
            model.online_encoder = model.online_encoder.train()
            batch_range = trange(num_batch)

            for j in batch_range:
                mm = 1 - (1 - beta) * (np.cos(np.pi * (j + num_batch * epoch) / (NUM_EPOCH * num_batch)) + 1) / 2.0

                ### Training
                now_time = j * input_len + input_len
                if now_time % day_cycle < day_start or now_time % day_cycle >= day_end:
                    continue

                head, tail = back_points[j], back_points[j+1]
                if head == tail:
                    continue
                sources_batch, destinations_batch = full_data.sources[head:tail], full_data.destinations[head:tail]
                timestamps_batch = torch.Tensor(full_data.timestamps[head:tail]).to(device)
                time_diffs_batch = torch.Tensor(- full_data.timestamps[head:tail] + now_time).to(device)
                edge_idxs_batch = torch.Tensor(full_data.edge_idxs[head:tail]).to(device)
                if now_time % day_cycle >= day_end:
                    predict_IND = False
                else:
                    predict_IND = True

                add_number = int(len(sources_batch) * (1 + args.ratio))
                if (j >= 2):
                    head_now, tail_now = back_points[j - 1], back_points[j + 2]#3
                else:
                    head_now, tail_now = back_points[j], back_points[j + 2]

                source_batch2, destination_batch2 = full_data.sources[head_now:tail_now], full_data.destinations[
                                                                                          head_now:tail_now]
                timestamps_batch2 =full_data.timestamps[head_now: tail_now]
                choice_index = torch.randperm(len(source_batch2))[:add_number]
                source_batch2 = source_batch2[choice_index]
                destination_batch2 = destination_batch2[choice_index]
                timestamps_batch2 = timestamps_batch2[choice_index]

                sorted_indices = np.argsort(timestamps_batch2)
                timestamps_batch2 = timestamps_batch2[sorted_indices]
                timestamps_batch2 = torch.Tensor(timestamps_batch2).to(device)
                source_batch2 = source_batch2[sorted_indices]
                destination_batch2 = destination_batch2[sorted_indices]

                loss, _, __ = model.compute_train_loss(source_batch2, destination_batch2, timestamps_batch2,
                                                       sources_batch, destinations_batch,
                                                       timestamps_batch, now_time,
                                                       time_diffs_batch,
                                                       edge_idxs_batch, od_matrix_30[j],
                                                       predict_IND=predict_IND)


                if predict_IND:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    m_loss.append(loss.item())

                batch_range.set_description(f"train_loss: {m_loss[-1]} ;")
                for param_q, param_k in zip(model.online_encoder.parameters(), model.target_encoder.parameters()):
                    param_k.data.mul_(mm).add_(param_q.data, alpha=1. - mm)
                model.target_encoder.restore_memory(model.online_encoder.backup_memory())

            ### Validation
            print("================================Val================================")
            val_metrics, embeddings, new_node_features = eval_phase1_prediction(model=model, data=full_data, back_points=back_points,
                                                          st=val_time, ed=test_time, device=device, config=config[DATA],
                                                          optimizer=optimizer, od_matrix=od_matrix_30)

            val_losses.append(val_metrics)
            train_losses.append(np.mean(m_loss))
            total_epoch_time = time.time() - start_epoch
            total_epoch_times.append(total_epoch_time)

            # Save temporary results
            pickle.dump({
                "val_losses": val_losses,
                "train_losses": train_losses,
                "total_epoch_times": total_epoch_times
            }, open(results_path, "wb"))

            logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
            logger.info('Epoch mean train loss: {}'.format(np.mean(m_loss)))
            logger.info('Epoch val loss: loss: {}'.format(val_metrics))
            # Early stopping
            ifstop, ifimprove = early_stopper.early_stop_check(val_metrics)
            if ifstop:
                logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
                logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
                logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
                break
            else:
                torch.save(
                    {"statedict": model.state_dict(), "memory": model.backup_memory()}, get_checkpoint_path(epoch))
        logger.info('Saving EvolveURE model')
        shutil.copy(get_checkpoint_path(early_stopper.best_epoch), MODEL_SAVE_PATH)
        logger.info('EvolveURE model saved')
        best_model_param = torch.load(get_checkpoint_path(early_stopper.best_epoch))
    else:
        best_model_param = torch.load(args.best)

    # load model parameters, memories from best epoch on val dataset
    model.load_state_dict(best_model_param["statedict"])
    model.restore_memory(best_model_param["memory"])
    model.init_memory()
    # Test
    print("================================Test================================")
    test_metrics, embeddings, new_node_features = eval_phase1_prediction(model=model, data=full_data,
                                                   back_points=back_points, st=0, ed=all_time,
                                                   device=device, config=config[DATA], optimizer=optimizer,
                                                   od_matrix=od_matrix_30)

    logger.info(
        'Test statistics:-- loss: {}'.format(test_metrics))
    # Save results for this run
    pickle.dump({
        "val_losses": val_losses,
        "test_metrics": test_metrics,
        "train_losses": train_losses,
        "total_epoch_times": total_epoch_times,
        "embeddings": embeddings,
        "node_features": new_node_features
    }, open(results_path, "wb"))


if __name__ == '__main__':
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    get_embedding(args)
