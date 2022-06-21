import torch

from train_CQE_TFLEX import FLEX

max_id = 20
entity_count = max_id
relation_count = max_id
timestamp_count = max_id
hidden_dim = 10
gamma = 10
center_reg = 0.02
test_batch_size = 1
input_dropout = 0.1
model = FLEX(
    nentity=entity_count,
    nrelation=relation_count,
    ntimestamp=timestamp_count,
    hidden_dim=hidden_dim,
    gamma=gamma,
    center_reg=center_reg,
    test_batch_size=test_batch_size,
    drop=input_dropout,
)
B = 8
query_args = ["e1", "r1", "t1", "e2", "r2", "t2", "r3", "t3"]
query_structure = "Pe_e2u"
query_tensor = torch.randint(0, max_id, (B, len(query_args)))
predict = model.single_predict(query_structure, query_tensor)
print(predict)
