from torch import nn


class EmbeddingModule(nn.Module):
    def __init__(self, n_node_features):
        super(EmbeddingModule, self).__init__()
        self.n_node_features = n_node_features

    def compute_embedding(self, memory, nodes):
        pass


class IdentityEmbedding(EmbeddingModule):
    def compute_embedding(self, memory, nodes):
        return memory[nodes, :]


class ExpLambsEmbedding(EmbeddingModule):
    def __init__(self, n_node_features):
        super(ExpLambsEmbedding, self).__init__(n_node_features)

    def compute_embedding(self, memory, nodes):
        embeddings = (memory[nodes, :, :-1] / memory[nodes, :, -1:]).reshape([len(nodes), -1])
        return embeddings


def get_embedding_module(module_type, n_node_features):
    if module_type == "identity":
        return IdentityEmbedding(n_node_features=n_node_features)
    elif module_type == "exp_lambs":
        return ExpLambsEmbedding(n_node_features=n_node_features)
    else:
        raise ValueError("Embedding Module {} not supported".format(module_type))
