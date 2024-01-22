from torch import nn
import torch


class ExpLambsMemoryUpdater(nn.Module):
    def __init__(self, memory, message_dimension, lambs, device):
        super(ExpLambsMemoryUpdater, self).__init__()
        self.memory = memory
        self.lambs = lambs
        self.lamb_len = lambs.shape[0]
        self.message_dimension = message_dimension
        self.device = device

    def update_memory(self, unique_node_ids, unique_messages, timestamps=None):
        if len(unique_node_ids) <= 0:
            return
        memory = self.memory.get_memory()[unique_node_ids]
        time_delta = self.memory.get_last_update(unique_node_ids) - timestamps
        # nl nld
        updated_memory = unique_messages + torch.exp(time_delta.repeat(self.lamb_len, 1).T * self.lambs).unsqueeze(
            -1) * memory
        if timestamps is not None:
            assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), "Trying to " \
                                                                                              "update memory to time in the past"
            self.memory.last_update[unique_node_ids] = timestamps

        self.memory.set_memory(unique_node_ids, updated_memory.detach())

    def get_updated_memory(self, unique_node_ids, unique_messages, timestamps=None, memory=None):
        if len(unique_node_ids) <= 0:
            return self.memory.memory.data.clone(), self.memory.last_update.data.clone()
        if memory is None:
            updated_memory = self.memory.memory.data.clone()
        else:
            updated_memory = memory

        updated_last_update = self.memory.last_update.data.clone()
        time_delta = updated_last_update[unique_node_ids] - timestamps
        updated_memory[unique_node_ids] = unique_messages + torch.exp(
            time_delta.repeat(self.lamb_len, 1).T * self.lambs).unsqueeze(-1) * updated_memory[unique_node_ids]

        if timestamps is not None:
            assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), "Trying to " \
                                                                                              "update memory to time in the past"
            updated_last_update[unique_node_ids] = timestamps

        return updated_memory, updated_last_update


def get_memory_updater(module_type, memory, message_dimension, memory_dimension, device):
    if module_type == "exp_lambs":
        return ExpLambsMemoryUpdater(memory, message_dimension, memory_dimension, device)
    else:
        ValueError("Memory updater {} not implemented".format(module_type))
