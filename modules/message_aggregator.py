import torch
import numpy as np


class ExpLambsMessageAggregator(torch.nn.Module):
    def __init__(self, device, embedding_dimension):
        super(ExpLambsMessageAggregator, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.device = device

    def aggregate(self, node_ids, messages, lambs):
        unique_node_ids = np.unique(node_ids)
        unique_messages = []
        unique_timestamps = []
        to_update_node_ids = []
        for node_id in unique_node_ids:
            if len(messages[node_id]) > 0:
                to_update_node_ids.append(node_id)
                unique_timestamps.append(messages[node_id][1][-1])
                A = messages[node_id][0].repeat(lambs.shape[0], 1, 1).permute([1, 0, 2])
                B = torch.exp(
                        (messages[node_id][1] - messages[node_id][1][-1]).repeat(lambs.shape[0],
                                                                                 1).T * lambs).unsqueeze(-1)
                # unique_messages.append(torch.sum( A * B, dim=0))
                # unique_messages.append(torch.sum(
                #     messages[node_id][0].repeat(lambs.shape[0], 1, 1).permute([1, 0, 2]) * torch.exp(
                #         (messages[node_id][1] - messages[node_id][1][-1]).repeat(lambs.shape[0],
                #                                                                  1).T * lambs).unsqueeze(-1), dim=0))

                # unique_messages.append(torch.mean(messages[node_id][0], dim=0).unsqueeze(   0) )
                unique_messages.append(torch.mean(A * B, dim=0))
        unique_messages = torch.stack(unique_messages) if len(to_update_node_ids) > 0 else []
        unique_timestamps = torch.stack(unique_timestamps) if len(to_update_node_ids) > 0 else []

        return to_update_node_ids, unique_messages, unique_timestamps


def get_message_aggregator(aggregator_type, device, embedding_dimension=0):
    if aggregator_type == "exp_lambs":
        return ExpLambsMessageAggregator(device=device, embedding_dimension=embedding_dimension)
    else:
        raise ValueError("Message aggregator {} not implemented".format(aggregator_type))
