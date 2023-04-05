import torch


def indexOf(tensor: torch.Tensor, values: torch.Tensor):
    origin_shape = tensor.shape
    assert len(values.shape) == 1
    # print("tensor shape:", tensor.shape, "values shape:", values.shape, "nonzero", torch.nonzero(tensor.view(-1)[..., None] == values).shape)
    # print("tensor shape:", tensor.shape, "values shape:", values.shape, "nonzero",  (tensor.view(-1)[..., None] == values).shape)
    # print("max", tensor.max(), "min", tensor.min(), "values max", values.max(), "values min", values.min())
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
