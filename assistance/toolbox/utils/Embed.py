from typing import List

import torch


def get_vec(entities_embedding, id_list: List[int], embedding_dim=200, device="cuda"):
    tensor = torch.LongTensor(id_list).view(-1, 1).to(device)
    return entities_embedding(tensor).view(-1, embedding_dim).cpu().detach().numpy()


def get_vec2(entities_embedding, id_list: List[int], embedding_dim=200, device="cuda"):
    all_entity_ids = torch.LongTensor(id_list).view(-1).to(device)
    all_entity_vec = torch.index_select(
        entities_embedding,
        dim=0,
        index=all_entity_ids
    ).view(-1, embedding_dim).cpu().detach().numpy()
    return all_entity_vec


def get_vec3(entities_embedding, orth: torch.Tensor, id_list: List[int], device="cuda"):
    all_entity_ids = torch.LongTensor(id_list).view(-1).to(device)
    all_entity_vec = torch.index_select(
        entities_embedding,
        dim=0,
        index=all_entity_ids
    ).view(-1, 200)
    all_entity_vec = all_entity_vec.matmul(orth.transpose(0, 1))
    return all_entity_vec.cpu().detach().numpy()

