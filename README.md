# TFLEX

code for "TFLEX: Temporal Feature-Logic Embedding Framework for Complex Reasoning over Temporal Knowledge Graph"

$$q=V_{?},\exists t:criticize(China, Japan, t) \land (visit(Xi Jinping, V_{?}, t'>t) \land \lnot visit(Obama, V_{?}, t'>t))$$

## Environment

- PyTorch 1.8.1 + cuda 10.2

```
pip install -r requirements
cd assistence
pip install -e .
```

## Get Started

```shell
# ICEWS14
CUDA_VISIBLE_DEVICES=0 python train_TCQE_TFLEX.py --name="TFLEX_dim800_gamma15" --hidden_dim=800 --test_batch_size=32 --every_test_step=10000 --dataset="ICEWS14"
CUDA_VISIBLE_DEVICES=0 python train_TCQE_X+ConE.py --name="X+ConE_dim800_gamma15" --hidden_dim=800 --test_batch_size=32 --every_test_step=10000 --dataset="ICEWS14"
CUDA_VISIBLE_DEVICES=0 python train_TCQE_X-1F.py --name="X-1F_dim800_gamma15" --hidden_dim=800 --test_batch_size=32 --every_test_step=10000 --dataset="ICEWS14"
CUDA_VISIBLE_DEVICES=0 python train_TCQE_X_without_entity_logic.py --name="X_without_entity_logic_dim800_gamma15" --hidden_dim=800 --test_batch_size=32 --every_test_step=10000 --dataset="ICEWS14"
CUDA_VISIBLE_DEVICES=0 python train_TCQE_X_without_time_logic.py --name="X_without_time_logic_dim800_gamma15" --hidden_dim=800 --test_batch_size=32 --every_test_step=10000 --dataset="ICEWS14"
CUDA_VISIBLE_DEVICES=0 python train_TCQE_X_without_logic.py --name="X_without_logic_dim800_gamma15" --hidden_dim=800 --test_batch_size=32 --every_test_step=10000 --dataset="ICEWS14"
CUDA_VISIBLE_DEVICES=0 python train_TCQE_Query2box.py --name="Query2box_dim800_gamma15" --hidden_dim=800 --test_batch_size=32 --every_test_step=10000 --dataset="ICEWS14"
CUDA_VISIBLE_DEVICES=0 python train_TCQE_BetaE.py --name="BetaE_dim800_gamma15" --hidden_dim=800 --test_batch_size=32 --every_test_step=10000 --dataset="ICEWS14"
CUDA_VISIBLE_DEVICES=0 python train_TCQE_ConE.py --name="ConE_dim800_gamma15" --hidden_dim=800 --test_batch_size=32 --every_test_step=10000 --dataset="ICEWS14"

CUDA_VISIBLE_DEVICES=0 python train_TCQE_Query2box.py --name="Query2box_dim800_gamma15" --hidden_dim=800 --test_batch_size=32 --every_test_step=10000 --dataset="ICEWS14" --resume=True --eval_tasks="Pe,Pe2,Pe3,e2i,e3i"
CUDA_VISIBLE_DEVICES=0 python train_TCQE_BetaE.py --name="BetaE_dim800_gamma15" --hidden_dim=800 --test_batch_size=32 --every_test_step=10000 --dataset="ICEWS14" --resume=True --eval_tasks="Pe,Pe2,Pe3,e2i,e3i,e2i_N,e3i_N,Pe_e2i_Pe_NPe,e2i_PeN,e2i_NPe,e2u,Pe_e2u"
CUDA_VISIBLE_DEVICES=0 python train_TCQE_ConE.py --name="ConE_dim800_gamma15" --hidden_dim=800 --test_batch_size=16 --every_test_step=10000 --dataset="ICEWS14" --resume=True --eval_tasks="Pe,Pe2,Pe3,e2i,e3i,e2i_N,e3i_N,Pe_e2i_Pe_NPe,e2i_PeN,e2i_NPe,e2u,Pe_e2u"

# ICEWS05-15
CUDA_VISIBLE_DEVICES=0 python train_TCQE_TFLEX.py --name="TFLEX_dim800_gamma15" --hidden_dim=800 --test_batch_size=16 --every_test_step=10000 --dataset="ICEWS05_15"
CUDA_VISIBLE_DEVICES=0 python train_TCQE_X+ConE.py --name="X+ConE_dim800_gamma15" --hidden_dim=800 --test_batch_size=16 --every_test_step=10000 --dataset="ICEWS05_15"
CUDA_VISIBLE_DEVICES=0 python train_TCQE_X-1F.py --name="X-1F_dim800_gamma15" --hidden_dim=800 --test_batch_size=16 --every_test_step=10000 --dataset="ICEWS05_15"
CUDA_VISIBLE_DEVICES=0 python train_TCQE_X_without_entity_logic.py --name="X_without_entity_logic_dim800_gamma15" --hidden_dim=800 --test_batch_size=16 --every_test_step=10000 --dataset="ICEWS05_15"
CUDA_VISIBLE_DEVICES=0 python train_TCQE_X_without_time_logic.py --name="X_without_time_logic_dim800_gamma15" --hidden_dim=800 --test_batch_size=16 --every_test_step=10000 --dataset="ICEWS05_15"
CUDA_VISIBLE_DEVICES=0 python train_TCQE_X_without_logic.py --name="X_without_logic_dim800_gamma15" --hidden_dim=800 --test_batch_size=16 --every_test_step=10000 --dataset="ICEWS05_15"
CUDA_VISIBLE_DEVICES=0 python train_TCQE_Query2box.py --name="Query2box_dim800_gamma15" --hidden_dim=800 --test_batch_size=16 --every_test_step=10000 --dataset="ICEWS05_15"
CUDA_VISIBLE_DEVICES=0 python train_TCQE_BetaE.py --name="BetaE_dim800_gamma15" --hidden_dim=800 --test_batch_size=16 --every_test_step=10000 --dataset="ICEWS05_15"
CUDA_VISIBLE_DEVICES=0 python train_TCQE_ConE.py --name="ConE_dim800_gamma15" --hidden_dim=800 --test_batch_size=16 --every_test_step=10000 --dataset="ICEWS05_15"

# GDELT
CUDA_VISIBLE_DEVICES=0 python train_TCQE_TFLEX.py --name="TFLEX_dim800_gamma15" --hidden_dim=800 --test_batch_size=64 --every_test_step=100000 --dataset="GDELT"
CUDA_VISIBLE_DEVICES=0 python train_TCQE_X+ConE.py --name="X+ConE_dim800_gamma15" --hidden_dim=800 --test_batch_size=64 --every_test_step=100000 --dataset="GDELT"
CUDA_VISIBLE_DEVICES=0 python train_TCQE_X-1F.py --name="X-1F_dim800_gamma15" --hidden_dim=800 --test_batch_size=64 --every_test_step=100000 --dataset="GDELT"
CUDA_VISIBLE_DEVICES=0 python train_TCQE_X_without_entity_logic.py --name="X_without_entity_logic_dim800_gamma15" --hidden_dim=800 --test_batch_size=64 --every_test_step=100000 --dataset="GDELT"
CUDA_VISIBLE_DEVICES=0 python train_TCQE_X_without_time_logic.py --name="X_without_time_logic_dim800_gamma15" --hidden_dim=800 --test_batch_size=64 --every_test_step=100000 --dataset="GDELT"
CUDA_VISIBLE_DEVICES=0 python train_TCQE_X_without_logic.py --name="X_without_logic_dim800_gamma15" --hidden_dim=800 --test_batch_size=64 --every_test_step=100000 --dataset="GDELT"
CUDA_VISIBLE_DEVICES=0 python train_TCQE_Query2box.py --name="Query2box_dim800_gamma15" --hidden_dim=800 --test_batch_size=64 --every_test_step=100000 --dataset="GDELT"
CUDA_VISIBLE_DEVICES=0 python train_TCQE_BetaE.py --name="BetaE_dim800_gamma15" --hidden_dim=800 --test_batch_size=64 --every_test_step=100000 --dataset="GDELT"
CUDA_VISIBLE_DEVICES=0 python train_TCQE_ConE.py --name="ConE_dim800_gamma15" --hidden_dim=800 --test_batch_size=64 --every_test_step=100000 --dataset="GDELT"
```

## Dataset

To generate dataset, please run `python run_sampling_TCQs.py`.

To show the meta of the generated dataset, run `python run_meta.py`.

## Interpreter

To launch an interactive interpreter, please run `python run_reasoning_interpreter.py`

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
