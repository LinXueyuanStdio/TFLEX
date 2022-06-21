# FLEX

code for "TFLEX: Temporal Feature-Logic Embedding Framework for CompleX Knowledge Graph Reasoning"

$$q=V_{?},\exists t:criticize(China, Japan, t) \land (visit(Xi Jinping, V_{?}, t'>t) \land \lnot visit(Obama, V_{?}, t'>t))$$

## Environment

- PyTorch 1.8.1 + cuda 10.2

## Get Started

```shell
# FB15k-237
CUDA_VISIBLE_DEVICES=0 python train_CQE_TFLEX.py --name="TFLEX_base" --dataset="ICEWS14"

# FB15k
CUDA_VISIBLE_DEVICES=0 python train_CQE_TFLEX.py --name="TFLEX_base" --dataset="ICEWS05_15"

# NELL
CUDA_VISIBLE_DEVICES=0 python train_CQE_TFLEX.py --name="TFLEX_base" --dataset="GDELT"
```

## Dataset

The KG data (FB15k, FB15k-237, NELL995) mentioned in the BetaE paper and the Query2box paper can be downloaded [here](http://snap.stanford.edu/betae/KG_data.zip). Note the two use the same training queries, but the difference is that the valid/test queries in BetaE paper have a maximum number of answers, making it more realistic.

Each folder in the data represents a KG, including the following files.
- `train.txt/valid.txt/test.txt`: KG edges
- `id2rel/rel2id/ent2id/id2ent.pkl`: KG entity relation dicts
- `train-queries/valid-queries/test-queries.pkl`: `defaultdict(set)`, each key represents a query structure, and the value represents the instantiated queries
- `train-answers.pkl`: `defaultdict(set)`, each key represents a query, and the value represents the answers obtained in the training graph (edges in `train.txt`)
- `valid-easy-answers/test-easy-answers.pkl`: `defaultdict(set)`, each key represents a query, and the value represents the answers obtained in the training graph (edges in `train.txt`) / valid graph (edges in `train.txt`+`valid.txt`)
- `valid-hard-answers/test-hard-answers.pkl`: `defaultdict(set)`, each key represents a query, and the value represents the **additional** answers obtained in the validation graph (edges in `train.txt`+`valid.txt`) / test graph (edges in `train.txt`+`valid.txt`+`test.txt`)
