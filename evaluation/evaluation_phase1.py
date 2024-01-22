from tqdm import tqdm
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error


def eval_phase1_prediction(model, data, back_points, st, ed, device, config, optimizer, od_matrix):
    input_len = config["input_len"]
    output_len = config["output_len"]
    day_cycle = config["day_cycle"]
    day_start = config["day_start"]
    day_end = config["day_end"]
    embeddings_list = []
    node_features = []
    loss_list = []

    with torch.no_grad():
        model = model.eval()
        # num_test_batch = (ed - st - output_len) // input_len
        num_test_batch = (ed - st) // input_len
        for j in tqdm(range(num_test_batch)):
            # optimizer.zero_grad()
            begin_time = j * input_len + st
            now_time = j * input_len + input_len + st
            if now_time % day_cycle < day_start or now_time % day_cycle >= day_end:
                continue
            head, tail = back_points[begin_time // input_len], back_points[now_time // input_len]
            # [head,tail1) nowtime [tail1,tail2) nowtime+τ

            if head == tail:
                embeddings_list.append(embeddings_list[-1])
                continue

            if now_time % day_cycle >= day_end:
                predict_IND = False
            else:
                predict_IND = True

            sources_batch, destinations_batch = data.sources[head:tail], data.destinations[head:tail]
            timestamps_batch_torch = torch.Tensor(data.timestamps[head:tail]).to(device)
            time_diffs_batch_torch = torch.Tensor(- data.timestamps[head:tail] + now_time).to(device)
            edge_idxs_batch = torch.Tensor(data.edge_idxs[head:tail]).to(device)
            loss, embeddings, node_feature = model.compute_IND(sources_batch, destinations_batch,
                                                                         timestamps_batch_torch, now_time,
                                                                         time_diffs_batch_torch,
                                                                         edge_idxs_batch, od_matrix[begin_time // input_len],
                                                                         predict_IND=predict_IND)
            if predict_IND:
                embeddings_list.append(embeddings.cpu().detach().numpy())
                node_features.append(node_feature.cpu().detach().numpy())
                loss_list.append(loss.item())
        stacked_embeddings = np.stack(embeddings_list)
        stacked_node_features = np.stack(node_features)
    return np.mean(loss_list), stacked_embeddings, stacked_node_features
