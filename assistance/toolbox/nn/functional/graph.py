from functools import reduce

import torch


def edge_match(edge_index, query_index):
    # O((n + q)logn) time
    # O(n) memory
    # edge_index: big underlying graph (3, n)
    # query_index: edges to match (3, q)

    base = edge_index.max(dim=1)[0] + 1
    # we will map edges to long ints, so we need to make sure the maximum product is less than MAX_LONG_INT
    # idea: max number of edges = num_nodes * num_relations
    # e.g. for a graph of 10 nodes / 5 relations, edge IDs 0...9 mean all possible outgoing edge types from node 0
    # given a tuple (h, r), we will search for all other existing edges starting from head h
    assert reduce(int.__mul__, base.tolist()) < torch.iinfo(torch.long).max
    scale = base.cumprod(0)  # (num_nodes, num_nodes * num_nodes, num_nodes * num_nodes * num_relations)
    scale = scale[-1] // scale  # (num_nodes * num_relations, num_relations, 1)

    # hash both the original edge index and the query index to unique integers
    # hash(edge_idx=(h, t, r)) = h * num_nodes * num_relations + t * num_relations + r
    edge_hash = (edge_index * scale.unsqueeze(-1)).sum(dim=0)  # vector, len=edge_index.shape[1]
    edge_hash, order = edge_hash.sort()
    query_hash = (query_index * scale.unsqueeze(-1)).sum(dim=0)  # vector, len=query_index.shape[1]

    # matched ranges: [start[i], end[i]) 可能有重复边，所以是个区间
    start = torch.bucketize(query_hash, edge_hash)  # [edge_hash.index(i) for i in query_hash], vector, len=query_index.shape[1]
    end = torch.bucketize(query_hash, edge_hash, right=True)  # [edge_hash.index(i, right=True) for i in query_hash], vector, len=query_index.shape[1]
    # num_match shows how many edges satisfy the (h, r) pattern for each query in the batch
    num_match = end - start  # 差分数组，num_match[i]=所匹配到的每个边的重复个数-1，最小为0表示只有一条边

    # generate the corresponding ranges
    offset = num_match.cumsum(0) - num_match  # [0] + num_match.cumsum(dim=0)[:-1], vector, len=query_index.shape[1]
    arange = torch.arange(num_match.sum(), device=edge_index.device)  # num_match.sum() = 匹配到的边数（重复边也计入）
    arange = arange + (start - offset).repeat_interleave(num_match)
    # start =     [s[0], s[1], ..., s[n-1]], n=query_index.shape[1]
    # num_match = [d[0], d[1], ..., d[n-1]], d[i]=end[i]-start[i]为区间长度
    # offset =    [0, d[0], d[0]+d[1], ..., d[0]+...+d[n-1]]
    # (start - offset).repeat_interleave(num_match)
    #       = [s[0], s[0],   ..., s[0];        s[1]-d[0], s[1]-d[0], ..., s[1]-d[0];   ...; s[n-1]-sum_{k=0}^{n-2} d[k], ...,           s[n-1]-sum_{k=0}^{n-2} d[k] ]
    # arange= [0,    1,      ..., d[0]-1;      d[0],      d[0]+1,    ..., d[0]+d[1]-1; ...; sum_{k=0}^{n-2} d[k],        ...,           sum_{k=0}^{n-1} d[k]+1      ]
    # +.    = [s[0], s[0]+1, ..., s[0]+d[0]-1; s[1],      s[1]+1,    ..., s[1]+d[1]-1; ...; s[n-1],                      s[n-1]+1, ..., s[n-1]+d[n-1]-1             ]

    return order[arange], num_match


def negative_sampling(data, batch, num_negative, strict=True):
    batch_size = len(batch)
    pos_h_index, pos_t_index, pos_r_index = batch.t()

    # strict negative sampling vs random negative sampling
    if strict:
        # 严格负采样，真负例
        t_mask, h_mask = strict_negative_mask(data, batch)
        t_mask = t_mask[:batch_size // 2]
        neg_t_candidate = t_mask.nonzero()[:, 1]
        num_t_candidate = t_mask.sum(dim=-1)
        # draw samples for negative tails
        rand = torch.rand(len(t_mask), num_negative, device=batch.device)
        index = (rand * num_t_candidate.unsqueeze(-1)).long()
        index = index + (num_t_candidate.cumsum(0) - num_t_candidate).unsqueeze(-1)
        neg_t_index = neg_t_candidate[index]

        h_mask = h_mask[batch_size // 2:]
        neg_h_candidate = h_mask.nonzero()[:, 1]
        num_h_candidate = h_mask.sum(dim=-1)
        # draw samples for negative heads
        rand = torch.rand(len(h_mask), num_negative, device=batch.device)
        index = (rand * num_h_candidate.unsqueeze(-1)).long()
        index = index + (num_h_candidate.cumsum(0) - num_h_candidate).unsqueeze(-1)
        neg_h_index = neg_h_candidate[index]
    else:
        # 随机负采样，假负例
        neg_index = torch.randint(data.num_nodes, (batch_size, num_negative), device=batch.device)
        neg_t_index, neg_h_index = neg_index[:batch_size // 2], neg_index[batch_size // 2:]

    h_index = pos_h_index.unsqueeze(-1).repeat(1, num_negative + 1)
    t_index = pos_t_index.unsqueeze(-1).repeat(1, num_negative + 1)
    r_index = pos_r_index.unsqueeze(-1).repeat(1, num_negative + 1)  # r 没翻转
    t_index[:batch_size // 2, 1:] = neg_t_index  # batch 上半是 t 负例
    h_index[batch_size // 2:, 1:] = neg_h_index  # batch 下半是 h 负例，但 r 没翻转

    return torch.stack([h_index, t_index, r_index], dim=-1)


def all_negative(data, batch):
    pos_h_index, pos_t_index, pos_r_index = batch.t()
    r_index = pos_r_index.unsqueeze(-1).expand(-1, data.num_nodes)
    # generate all negative tails for this batch
    all_index = torch.arange(data.num_nodes, device=batch.device)
    h_index, t_index = torch.meshgrid(pos_h_index, all_index)
    t_batch = torch.stack([h_index, t_index, r_index], dim=-1)
    # generate all negative heads for this batch
    all_index = torch.arange(data.num_nodes, device=batch.device)
    t_index, h_index = torch.meshgrid(pos_t_index, all_index)
    h_batch = torch.stack([h_index, t_index, r_index], dim=-1)

    return t_batch, h_batch


def strict_negative_mask(data, batch):
    # this function makes sure that for a given (h, r) batch we will NOT sample true tails as random negatives
    # similarly, for a given (t, r) we will NOT sample existing true heads as random negatives

    pos_h_index, pos_t_index, pos_r_index = batch.t()

    # part I: sample hard negative tails
    # edge index of all (head, relation) edges from the underlying graph
    edge_index = torch.stack([data.edge_index[0], data.edge_type])
    # edge index of current batch (head, relation) for which we will sample negatives
    query_index = torch.stack([pos_h_index, pos_r_index])
    # search for all true tails for the given (h, r) batch
    edge_id, num_t_truth = edge_match(edge_index, query_index)
    # build an index from the found edges
    t_truth_index = data.edge_index[1, edge_id]
    sample_id = torch.arange(len(num_t_truth), device=batch.device).repeat_interleave(num_t_truth)
    t_mask = torch.ones(len(num_t_truth), data.num_nodes, dtype=torch.bool, device=batch.device)
    # assign 0s to the mask with the found true tails
    t_mask[sample_id, t_truth_index] = 0
    t_mask.scatter_(1, pos_t_index.unsqueeze(-1), 0)

    # part II: sample hard negative heads
    # edge_index[1] denotes tails, so the edge index becomes (t, r)
    edge_index = torch.stack([data.edge_index[1], data.edge_type])
    # edge index of current batch (tail, relation) for which we will sample heads
    query_index = torch.stack([pos_t_index, pos_r_index])
    # search for all true heads for the given (t, r) batch
    edge_id, num_h_truth = edge_match(edge_index, query_index)
    # build an index from the found edges
    h_truth_index = data.edge_index[0, edge_id]
    sample_id = torch.arange(len(num_h_truth), device=batch.device).repeat_interleave(num_h_truth)
    h_mask = torch.ones(len(num_h_truth), data.num_nodes, dtype=torch.bool, device=batch.device)
    # assign 0s to the mask with the found true heads
    h_mask[sample_id, h_truth_index] = 0
    h_mask.scatter_(1, pos_h_index.unsqueeze(-1), 0)

    return t_mask, h_mask


def compute_ranking(pred, target, mask=None):
    pos_pred = pred.gather(-1, target.unsqueeze(-1))
    if mask is not None:
        # filtered ranking
        ranking = torch.sum((pos_pred <= pred) & mask, dim=-1) + 1
    else:
        # unfiltered ranking
        ranking = torch.sum(pos_pred <= pred, dim=-1) + 1
    return ranking


def index_to_mask(index, size):
    index = index.view(-1)
    size = int(index.max()) + 1 if size is None else size
    mask = index.new_zeros(size, dtype=torch.bool)
    mask[index] = True
    return mask
