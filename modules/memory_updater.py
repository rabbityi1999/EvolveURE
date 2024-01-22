from torch import nn

class GRUMemoryUpdater(nn.Module):
    def __init__(self, memory, message_dimension, memory_dimension, device):
        super(GRUMemoryUpdater, self).__init__()
        self.memory = memory
        self.message_dimension = message_dimension
        self.device = device
        print("message_dimension:",message_dimension,"memory_dimension:",memory_dimension)
        self.memory_updater = nn.GRUCell(input_size=message_dimension,
                                         hidden_size=memory_dimension)
    def get_updated_memory(self, unique_node_ids, unique_messages, timestamps, memory=None):
        if len(unique_node_ids) <= 0:
            return self.memory.memory.data.clone(), self.memory.last_update.data.clone()

        updated_memory = self.memory.memory.data.clone()
        updated_memory = updated_memory.squeeze()
        unique_messages = unique_messages.squeeze()
        updated_memory[unique_node_ids] = self.memory_updater(unique_messages, updated_memory[unique_node_ids])
        updated_last_update = self.memory.last_update.data.clone()
        updated_last_update[unique_node_ids] = timestamps
        updated_memory = updated_memory.unsqueeze(1)
        local_nodes_updated_memory = updated_memory[unique_node_ids]
        self.memory.set_memory(unique_node_ids, local_nodes_updated_memory.detach())
        return updated_memory, updated_last_update

def get_memory_updater(memory, message_dimension, memory_dimension, device ):
    return GRUMemoryUpdater(memory, message_dimension, memory_dimension, device)
