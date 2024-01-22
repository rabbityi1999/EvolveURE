import numpy as np


class Data:
    def __init__(self, sources, destinations, timestamps, edge_idxs, n_nodes):
        self.sources = sources
        self.destinations = destinations
        self.timestamps = timestamps
        self.edge_idxs = edge_idxs
        self.n_interactions = len(sources)
        self.unique_nodes = set(list(range(n_nodes)))
        self.n_unique_nodes = n_nodes


def get_data(config):
    n_nodes = config["n_nodes"]
    whole_data = np.load(config["data_path"])
    od_matrix = np.load(config["matrix_path"])
    back_points = np.load(config["point_path"])
    # print("data loaded")
    all_time = (config["train_day"] + config["val_day"] + config["test_day"]) * config["day_cycle"]
    # val_time, test_time = (config["train_day"]) * config["day_cycle"] - config["input_len"], (
    #         config["train_day"] + config["val_day"]) * config["day_cycle"] - config["input_len"]
    val_time, test_time = (config["train_day"]) * config["day_cycle"], (
            config["train_day"] + config["val_day"]) * config["day_cycle"]

    sources = whole_data[::config["sample"], 0].astype("int")
    destinations = whole_data[::config["sample"], 1].astype("int")
    timestamps = whole_data[::config["sample"], 2].astype("int")
    if whole_data.shape[1] > 3:
        edge_idxs = whole_data[::config["sample"], 3]
        print("self edge index")
    else:
        edge_idxs = np.arange(whole_data.shape[0] // config["sample"])
    node_features = np.diag(np.ones(n_nodes))
    full_data = Data(sources, destinations, timestamps, edge_idxs, n_nodes)
    print("The dataset has {} interactions, involving {} different nodes".format(full_data.n_interactions,
                                                                                 full_data.n_unique_nodes))

    return n_nodes, node_features, full_data, val_time, test_time, all_time, od_matrix, back_points

