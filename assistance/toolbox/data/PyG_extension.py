from typing import List, Tuple

import torch
from torch_geometric.data import Data

from toolbox.data.DataSchema import RelationalTripletData
from toolbox.data.functional import with_inverse_relations


def triple_to_Data(triple: List[Tuple[int, int, int]]) -> Data:
    triple_tensor = torch.tensor(triple, dtype=torch.long)
    return Data(edge_index=triple_tensor[:, [0, 2]].T, edge_attr=triple_tensor[:, 1])


def train_view(data: RelationalTripletData, max_relation_id: int) -> Data:
    train_triples, train_edge_idx, train_edge_type = with_inverse_relations(data.train_triples_ids, max_relation_id)
    train_view_of_graph = Data(edge_index=train_edge_idx, edge_type=train_edge_type, num_nodes=data.entity_count)
    train_view_of_graph.node_id = torch.arange(train_view_of_graph.num_nodes)
    return train_view_of_graph
