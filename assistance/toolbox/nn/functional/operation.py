import torch


def rankOf(vector, values):
    """Returns the indices of the first occurrences of values in a tensor.

    Args:
        tensor (tensor): ranking tensor, shape [len(tensor)]
        values (tensor): values to be ranked, shape [len(values)]

    Returns:
        tensor: indices of the first occurrences of values in a tensor, shape [len(values)]
    """
    return torch.nonzero(vector.view(-1)[..., None] == values)[:, 0]


def indexOf(tensor: torch.Tensor, values: torch.Tensor):
    origin_shape = tensor.shape
    assert len(values.shape) == 1
    return torch.nonzero(tensor.view(-1)[..., None] == values)[:, 1].view(*origin_shape)


def subgraph_by_anchors(anchor_ids: torch.LongTensor, edge_index: torch.LongTensor, edge_type=None):
    selector: torch.LongTensor = anchor_ids.view(-1)
    head, tail = edge_index[0, :].view(-1), edge_index[1, :].view(-1)
    head_index_selected: torch.LongTensor = torch.nonzero(head[..., None] == selector)[:, 0].view(-1)
    tail_index_selected: torch.LongTensor = torch.nonzero(tail[..., None] == selector)[:, 0].view(-1)
    edge_index_selected: torch.LongTensor = torch.nonzero(head_index_selected[..., None] == tail_index_selected)[:, 1]
    subgraph_edge_index = edge_index[:, edge_index_selected]
    if edge_type is not None:
        subgraph_edge_type = edge_type[edge_index_selected]
        return subgraph_edge_index, subgraph_edge_type
    else:
        return subgraph_edge_index
