# FLEX

code for "TFLEX: Temporal Feature-Logic Embedding Framework for CompleX Knowledge Graph Reasoning"

$$q=V_{?},\exists t:criticize(China, Japan, t) \land (visit(Xi Jinping, V_{?}, t'>t) \land \lnot visit(Obama, V_{?}, t'>t))$$

## Environment

- PyTorch 1.8.1 + cuda 10.2

## Get Started

```shell
# ICEWS14
CUDA_VISIBLE_DEVICES=0 python train_CQE_TFLEX.py --name="TFLEX" --dataset="ICEWS14"

# ICEWS05-15
CUDA_VISIBLE_DEVICES=0 python train_CQE_TFLEX.py --name="TFLEX" --dataset="ICEWS05_15"

# GDELT
CUDA_VISIBLE_DEVICES=0 python train_CQE_TFLEX.py --name="TFLEX" --dataset="GDELT"
```

## Dataset

To generate dataset, please run `python sampling_CQE.py`.

## Interpreter

To launch an interactive interpreter, please run `python interpreter.py`
