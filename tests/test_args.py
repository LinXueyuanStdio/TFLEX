from dataclasses import dataclass, field
import os
import sys
from toolbox.utils.KGArgs import OutputArguments
from toolbox.utils.KGArgsParser import KGEArgParser


@dataclass
class ModelArguments:
    entity_dim: int = field(default=800, metadata={
        "help": "The dimension of the entity embeddings."
    })
    hidden_dim: int = field(default=800, metadata={"help": "embedding dimension"})
    input_dropout: float = field(default=0.1, metadata={"help": "Input layer dropout."})
    gamma: float = field(default=15.0, metadata={"help": "margin in the loss"})
    center_reg: float = field(default=0.02, metadata={
        "help": 'center_reg for ConE, center_reg balances the in_cone dist and out_cone dist'
    })


@dataclass
class DataArguments:
    data_home: str = field(default="data", metadata={"help": "The folder path to dataset."})
    dataset: str = field(default="ICEWS14", metadata={"help": "Which dataset to use: ICEWS14, ICEWS05_15, GDELT."})


@dataclass
class ExperimentArguments:
    name: str = field(default="TFLEX_base", metadata={"help": "Name of the experiment."})


@dataclass
class TrainingArguments:

    do_train: bool = field(default=False, metadata={
        "help": "Whether to run training."
    })
    do_valid: bool = field(default=False, metadata={
        "help": "Whether to run on the dev set."
    })
    do_test: bool = field(default=False, metadata={
        "help": "Whether to run on the test set."
    })
    seed: int = field(default=42, metadata={"help": "random seed for initialization"})

    # 1. args for training, available only if do_train is True
    resume: bool = field(default=False, metadata={"help": "Resume from output directory."})
    resume_by_score: float = field(default=0.0, metadata={
        "help": "Resume by score from output directory. Resume best if it is 0. Default: 0"
    })
    start_step: int = field(default=0, metadata={"help": "start step."})
    max_steps: int = field(default=200001, metadata={"help": "Number of steps."})
    every_valid_step: int = field(default=10000, metadata={"help": "Number of steps."})
    every_test_step: int = field(default=10000, metadata={"help": "Number of steps."})

    negative_sample_size: 128 = field(type=int, metadata={"help": "negative entities sampled per query"})
    lr: float = field(default=0.0001, metadata={"help": "Learning rate."})

    train_tasks: str = field(default="Pe,Pe2,Pe3,e2i,e3i,"
                             + "Pt,aPt,bPt,Pe_Pt,Pt_sPe_Pt,Pt_oPe_Pt,t2i,t3i,"
                             + "e2i_N,e3i_N,Pe_e2i_Pe_NPe,e2i_PeN,e2i_NPe,"
                             + "t2i_N,t3i_N,Pe_t2i_PtPe_NPt,t2i_PtN,t2i_NPt",
                             metadata={"help": 'the tasks for training'})
    train_all: bool = field(default=False, metadata={
        "help": 'if training all, it will use all tasks in data.train_queries_answers'
    })
    train_batch_size: int = field(default=512, metadata={"help": "for training: batch size"})
    train_shuffle: bool = field(default=True, metadata={"help": "for training: shuffle data"})
    train_drop_last: bool = field(default=True, metadata={"help": "for training: drop last batch"})
    train_num_workers: int = field(default=1, metadata={"help": "for training: number of workers"})
    train_pin_memory: bool = field(default=False, metadata={"help": "for training: pin memory"})
    train_device: str = field(default="cuda:0", metadata={"help": "choice: cuda:0, cuda:1, cpu."})

    # 2. args for evaluation and testing, available only if do_eval or do_test is True
    test_tasks: str = field(default="Pe,Pt,Pe2,Pe3", metadata={"help": 'for testing: the tasks'})
    test_all: bool = field(default=False, metadata={
        "help": 'if testing all, it will use all tasks in data.test_queries_answers'
    })
    test_batch_size: int = field(default=8, metadata={"help": "for testing: batch size"})
    test_shuffle: bool = field(default=False, metadata={"help": "for testing: shuffle data"})
    test_drop_last: bool = field(default=False, metadata={"help": "for testing: drop last batch"})
    test_num_workers: int = field(default=1, metadata={"help": "for testing: number of workers"})
    test_pin_memory: bool = field(default=False, metadata={"help": "for testing: pin memory"})
    test_device: str = field(default="cuda:0", metadata={"help": "choice: cuda:0, cuda:1, cpu."})


# you should defined these: ModelArguments, DataArguments, TrainingArguments
parser = KGEArgParser([
    DataArguments,
    ModelArguments,
    OutputArguments,
    TrainingArguments,
])
if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
else:
    args_data, args_model, args_output, args_training = parser.parse_args_into_dataclasses()
