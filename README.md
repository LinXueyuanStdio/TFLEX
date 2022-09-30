# TFLEX

code for "TFLEX: Temporal Feature-Logic Embedding Framework for Complex Reasoning over Temporal Knowledge Graph"

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

## Citation
```
@misc{https://doi.org/10.48550/arxiv.2205.14307,
  doi = {10.48550/ARXIV.2205.14307},
  url = {https://arxiv.org/abs/2205.14307},
  author = {Lin, Xueyuan and Xu, Chengjin and E, Haihong and Su, Fenglong and Zhou, Gengxian and Hu, Tianyi and Li, Ningyuan and Sun, Mingzhi and Luo, Haoran},
  keywords = {Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {TFLEX: Temporal Feature-Logic Embedding Framework for Complex Reasoning over Temporal Knowledge Graph},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution Non Commercial Share Alike 4.0 International}
}
```
