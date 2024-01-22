import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.memory import ExpMemory_lambs
from modules.message_aggregator import get_message_aggregator
from modules.message_function import get_message_function
from modules.memory_updater import get_memory_updater
from modules.embedding_module import get_embedding_module
from torch.nn.functional import cosine_similarity

class MLP_Predictor(nn.Module):
    r"""MLP used for predictor. The MLP has one hidden layer.
    Args:
        input_size (int): Size of input features.
        output_size (int): Size of output features.
        hidden_size (int, optional): Size of hidden layer. (default: :obj:`4096`).
    """

    def __init__(self, input_size, output_size, hidden_size=512):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=True),
            nn.PReLU(1),
            nn.Linear(hidden_size, output_size, bias=True)
        )
        self.reset_parameters()

    def forward(self, x):
        return self.net(x)

    def reset_parameters(self):
        # kaiming_uniform
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()

class Encoder(nn.Module):
    def __init__(self, n_nodes, node_features, embedding_dimension, memory_dimension, message_dimension, lambs, device,
                 output, init_lamb=0.5):
        super(Encoder, self).__init__()
        self.n_nodes = n_nodes
        # dynamic node_features
        # self.node_features = nn.Parameter(torch.Tensor(node_features), requires_grad=True)
        self.node_features = node_features
        self.embedding_dimension = embedding_dimension
        self.output = output
        self.memory_dimension = memory_dimension
        self.message_dimension = message_dimension
        self.device = device
        self.lambs = torch.Tensor(lambs).to(self.device) * self.output
        self.lamb_len = self.lambs.shape[0]
        self.queue_length = 48

        # add source
        raw_message_dimension = 2 * self.memory_dimension * self.lamb_len + self.node_features.shape[1] + 1

        self.memory = ExpMemory_lambs(n_nodes=self.n_nodes,
                                      memory_dimension=self.memory_dimension,
                                      lambs=self.lambs,
                                      device=self.device)  # (3, nodes, raw_message_dim)
        self.message_aggregator = get_message_aggregator(aggregator_type="exp_lambs", device=self.device,
                                                         embedding_dimension=memory_dimension)
        self.message_function = get_message_function(module_type="mlp",
                                                     raw_message_dimension=raw_message_dimension,
                                                     message_dimension=self.message_dimension)
        self.memory_updater = get_memory_updater(memory=self.memory,
                                                 message_dimension=self.message_dimension,
                                                 memory_dimension=self.memory_dimension,
                                                 device=self.device)
        self.embedding_memory = torch.zeros([self.n_nodes, self.queue_length, self.memory_dimension]).to(self.device)
        self.exp_embedding = get_embedding_module(module_type="identity",
                                                  n_node_features=self.memory_dimension)
        self.iden_embedding = get_embedding_module(module_type="identity",
                                                   n_node_features=self.memory_dimension)
        self.theta = nn.Parameter(torch.Tensor([init_lamb]), requires_grad=True)
        self.static_embedding = nn.Embedding(self.n_nodes, self.embedding_dimension)
        self.embedding_transform = torch.nn.Sequential(
            torch.nn.Linear(memory_dimension * self.lamb_len, memory_dimension, bias=True),
            torch.nn.LeakyReLU()
        )
        self.spatial_transform = torch.nn.Sequential(
            torch.nn.Linear(memory_dimension, embedding_dimension, bias=True),  # changed
            torch.nn.LeakyReLU()
        )

    def forward(self, source_nodes, target_nodes, timestamps_batch_torch, now_time,od_matrix, predict_IND):
        # 1. Get node messages from last updated memories
        memory = self.memory.get_memory()
        source_embeddings = self.exp_embedding.compute_embedding(memory=memory,
                                                                 nodes=source_nodes)  # (nodes, l * memory)
        target_embeddings = self.exp_embedding.compute_embedding(memory=memory,
                                                                 nodes=target_nodes)  # (nodes, l * memory)
        source_embeddings = source_embeddings.squeeze()
        target_embeddings = target_embeddings.squeeze()
        # Compute node_level messages
        raw_messages = self.get_raw_messages(source_nodes,
                                             source_embeddings,
                                             target_embeddings,
                                             self.node_features[target_nodes],
                                             timestamps_batch_torch)  # (nodes, 2 * l * memory + feature)

        unique_nodes, unique_raw_messages, unique_timestamps = self.message_aggregator.aggregate(source_nodes,
                                                                                                 raw_messages,
                                                                                                 self.lambs)  # unique_raw_messages: (nodes, l, raw_message_dim)

        unique_messages = self.message_function.compute_message(unique_raw_messages)

        updated_memory, updated_last_update = self.memory_updater.get_updated_memory(unique_nodes,
                                                                                     unique_messages,
                                                                                     timestamps=unique_timestamps)
        recent_node_embeddings = self.exp_embedding.compute_embedding(memory=updated_memory,
                                                                      nodes=list(range(self.n_nodes))).reshape(
            [self.n_nodes, -1])

        dynamic_embeddings = recent_node_embeddings
        embeddings = self.theta * self.static_embedding.weight + (1 - self.theta) * self.spatial_transform(
            self.embedding_transform(
                dynamic_embeddings).reshape([1, self.n_nodes, self.memory_dimension]).permute([1, 0, 2]).reshape(
                [self.n_nodes, -1]))

        return embeddings, self.node_features

    def get_raw_messages(self, source_nodes, source_embeddings, target_embeddings, node_features, edge_times):
        # add source
        source_message = torch.cat(
            [source_embeddings, target_embeddings, node_features,
             torch.ones([target_embeddings.shape[0], 1]).to(self.device)], dim=1)
        messages = dict()
        unique_nodes = np.unique(source_nodes)
        for node_i in unique_nodes:
            ind = np.arange(source_message.shape[0])[source_nodes == node_i]
            messages[node_i] = [source_message[ind], edge_times[ind]]
        return messages

    def init_memory(self):
        self.memory.__init_memory__()

    def backup_memory(self):
        return [self.memory.backup_memory()]

    def restore_memory(self, memory):
        self.memory.restore_memory(memory[0])

    def detach_memory(self):
        self.memory.detach_memory()

class Decoder(nn.Module):
    def __init__(self, embedding_dim, n_nodes, device):
        super(Decoder, self).__init__()
        self.device = device
        self.n_nodes = n_nodes
        self.linear1 = nn.Linear(in_features=1, out_features=1)
        self.linear2 = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LeakyReLU(),
            nn.Linear(embedding_dim, int(embedding_dim / 2)),
            nn.LeakyReLU(),
            nn.Linear(int(embedding_dim / 2), 1)
        )

    def forward(self, embeddings, o_nodes, d_nodes, time_diffs, edge_ind, od_matrix, output_len):
        pos_value = edge_ind
        pos_pairs = torch.cat([embeddings[o_nodes], embeddings[d_nodes]], dim=-1) * self.linear1(
            time_diffs.unsqueeze(-1) / output_len)
        od_matrix[od_matrix == 0] = 1

        neg_mask = np.ones([self.n_nodes, self.n_nodes])
        neg_mask[o_nodes, d_nodes] = 0
        neg_mask = neg_mask.astype(bool)
        neg_o = np.arange(self.n_nodes).repeat([self.n_nodes])[neg_mask.reshape(-1)]
        neg_d = np.arange(self.n_nodes).repeat(self.n_nodes).reshape([self.n_nodes, self.n_nodes]).transpose(1,
                                                                                                             0).reshape(
            -1)[neg_mask.reshape(-1)]
        neg_pairs = torch.cat([embeddings[neg_o], embeddings[neg_d]],
                              dim=-1)

        normalizer = torch.Tensor(od_matrix[np.concatenate([o_nodes, neg_o]), np.concatenate([d_nodes, neg_d])]).to(
            self.device)

        out = self.linear2(torch.cat([pos_pairs, neg_pairs], dim=0)).reshape(-1)
        truth = torch.cat([pos_value, torch.zeros(np.sum(neg_mask)).to(self.device)])
        nll = torch.sum(torch.pow(out - truth, 2) / normalizer) / (self.n_nodes * self.n_nodes)
        return nll


class Mob_Decoder(nn.Module):
    def __init__(self, embedding_dim, n_nodes, device):
        super(Mob_Decoder, self).__init__()
        self.device = device
        self.n_nodes = n_nodes
        self.o_linear = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.d_linear = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)

    def forward(self, embeddings, o_nodes, d_nodes, od_matrix):
        """
        :param embeddings: n * d, the embedding for n nodes
        :param o_nodes: origin nodes
        :param d_nodes: destination nodes
        :param od_matrix: n * n, the od during this slice
        :return: mob loss
        """

        mobility = od_matrix.copy()
        mobility = mobility / np.mean(mobility)

        mob = torch.tensor(mobility, dtype=torch.float32).to(self.device)

        o_emb = self.o_linear(embeddings)
        d_emb = self.d_linear(embeddings)

        inner_prod = torch.mm(o_emb, d_emb.T)
        ps_hat = F.softmax(inner_prod, dim=-1)

        loss = torch.sum(-torch.mul(mob, torch.log(ps_hat)))

        return loss


class EvolveURE(nn.Module):
    def __init__(self, device,
                 n_nodes=268, node_features=None,
                 message_dimension=64, memory_dimension=64, lambs=None,
                 output=30):
        super(EvolveURE, self).__init__()
        if lambs is None:
            lambs = [1]
        self.logger = logging.getLogger(__name__)
        self.device = device
        node_raw_features = torch.from_numpy(node_features.astype(np.float32)).to(device)
        embedding_dimension = memory_dimension * 1
        self.online_encoder = Encoder(n_nodes, node_raw_features, embedding_dimension, memory_dimension,
                               message_dimension, lambs, device,
                               output, init_lamb=0.1)
        self.target_encoder = Encoder(n_nodes, node_raw_features, embedding_dimension, memory_dimension,
                                message_dimension, lambs, device,
                                output, init_lamb=0.1)
        self.output = output
        # od_loss
        self.mobility_reconstruct = Decoder(embedding_dimension, n_nodes, device)
        # mob_loss
        self.reconstruct_mob = Mob_Decoder(embedding_dimension, n_nodes, device)
        self.lambda_param = 0.005
        self.online_predictor = MLP_Predictor(memory_dimension , memory_dimension , memory_dimension ).to(device)

    def embedding_reconstruct(self, predicted_embeddings, target_embeddings):
        cos = cosine_similarity(predicted_embeddings,target_embeddings.detach(),dim=-1).mean()
        return 1 - cos

    def compute_IND(self, o_nodes, d_nodes, timestamps_batch_torch, now_time, time_diffs, edge_index, od_matrix,
                    predict_IND=True):
        embeddings, node_features = self.online_encoder(o_nodes, d_nodes, timestamps_batch_torch, now_time,od_matrix, predict_IND)
        nll = None
        if predict_IND:
            nll = self.mobility_reconstruct(embeddings, o_nodes, d_nodes, time_diffs,
                                            edge_index, od_matrix, self.output)

        return nll, embeddings, node_features

    def compute_train_loss(self, o_nodes2, d_nodes2, timestamps_batch_torch2,
                           o_nodes, d_nodes, timestamps_batch_torch, now_time, time_diffs, edge_index, od_matrix,
                           predict_IND=True):
        embeddings, node_features = self.online_encoder(o_nodes, d_nodes, timestamps_batch_torch, now_time,od_matrix, predict_IND)
        predicted_embeddings = self.online_predictor(embeddings)
        embeddings2, node_features2 = self.target_encoder(o_nodes2, d_nodes2, timestamps_batch_torch2, now_time,od_matrix, predict_IND)
        nll = None
        if predict_IND:
            nll = self.mobility_reconstruct(embeddings, o_nodes, d_nodes, time_diffs, edge_index, od_matrix, self.output)
            BYOL_loss = self.embedding_reconstruct(predicted_embeddings=predicted_embeddings, target_embeddings=embeddings2)
            nll = BYOL_loss + nll
        return nll, embeddings, node_features

    def init_memory(self):
        self.online_encoder.init_memory()
        self.target_encoder.init_memory()

    def backup_memory(self):
        return self.online_encoder.backup_memory()

    def restore_memory(self, memories):
        self.online_encoder.restore_memory(memories)

    def detach_memory(self):
        self.online_encoder.detach_memory()
        self.target_encoder.detach_memory()
