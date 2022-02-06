import json
import re
import sys
from argparse import ArgumentParser, ArgumentTypeError
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, List, NewType, Optional, Tuple, Union

import dataclasses

DataClass = NewType("DataClass", Any)
DataClassType = NewType("DataClassType", Any)


def string_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError(
            f"Truthy value expected: got {v} but expected one of yes/no, true/false, t/f, y/n, 1/0 (case insensitive)."
        )


class KGEArgParser(ArgumentParser):
    """
    This subclass of `argparse.ArgumentParser` uses type hints on dataclasses to generate arguments.

    The class is designed to play well with the native argparse. In particular, you can add more (non-dataclass backed)
    arguments to the parser after initialization and you'll get the output back after parsing as an additional
    namespace.

    Examples:
        >>> from toolbox.KGArgsParser import KGEArgParser
        >>> # you should defined these: ModelArguments, DataArguments, TrainingArguments
        >>> parser = KGEArgParser((ModelArguments, DataArguments, TrainingArguments))
        >>> if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        >>>     model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
        >>> else:
        >>>     model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    """

    dataclass_types: Iterable[DataClassType]

    def __init__(self, dataclass_types: Union[DataClassType, Iterable[DataClassType]], **kwargs):
        """
        Args:
            dataclass_types:
                Dataclass type, or list of dataclass types for which we will "fill" instances with the parsed args.
            kwargs:
                (Optional) Passed to `argparse.ArgumentParser()` in the regular way.
        """
        super().__init__(**kwargs)
        if dataclasses.is_dataclass(dataclass_types):
            dataclass_types = [dataclass_types]
        self.dataclass_types = dataclass_types
        for dtype in self.dataclass_types:
            self._add_dataclass_arguments(dtype)

    def _add_dataclass_arguments(self, dtype: DataClassType):
        for field in dataclasses.fields(dtype):
            if not field.init:
                continue
            field_name = f"--{field.name}"
            kwargs = field.metadata.copy()
            # field.metadata is not used at all by Data Classes,
            # it is provided as a third-party extension mechanism.
            if isinstance(field.type, str):
                raise ImportError(
                    "This implementation is not compatible with Postponed Evaluation of Annotations (PEP 563),"
                    "which can be opted in from Python 3.7 with `from __future__ import annotations`."
                    "We will add compatibility when Python 3.9 is released."
                )
            typestring = str(field.type)
            for prim_type in (int, float, str):
                for collection in (List,):
                    if (
                            typestring == f"typing.Union[{collection[prim_type]}, NoneType]"
                            or typestring == f"typing.Optional[{collection[prim_type]}]"
                    ):
                        field.type = collection[prim_type]
                if (
                        typestring == f"typing.Union[{prim_type.__name__}, NoneType]"
                        or typestring == f"typing.Optional[{prim_type.__name__}]"
                ):
                    field.type = prim_type

            if isinstance(field.type, type) and issubclass(field.type, Enum):
                kwargs["choices"] = [x.value for x in field.type]
                kwargs["type"] = type(kwargs["choices"][0])
                if field.default is not dataclasses.MISSING:
                    kwargs["default"] = field.default
                else:
                    kwargs["required"] = True
            elif field.type is bool or field.type == Optional[bool]:
                if field.default is True:
                    self.add_argument(f"--no_{field.name}", action="store_false", dest=field.name, **kwargs)

                # Hack because type=bool in argparse does not behave as we want.
                kwargs["type"] = string_to_bool
                if field.type is bool or (field.default is not None and field.default is not dataclasses.MISSING):
                    # Default value is True if we have no default when of type bool.
                    default = True if field.default is dataclasses.MISSING else field.default
                    # This is the value that will get picked if we don't include --field_name in any way
                    kwargs["default"] = default
                    # This tells argparse we accept 0 or 1 value after --field_name
                    kwargs["nargs"] = "?"
                    # This is the value that will get picked if we do --field_name (without value)
                    kwargs["const"] = True
            elif (
                    hasattr(field.type, "__origin__") and re.search(r"^typing\.List\[(.*)\]$",
                                                                    str(field.type)) is not None
            ):
                kwargs["nargs"] = "+"
                kwargs["type"] = field.type.__args__[0]
                assert all(
                    x == kwargs["type"] for x in field.type.__args__
                ), f"{field.name} cannot be a List of mixed types"
                if field.default_factory is not dataclasses.MISSING:
                    kwargs["default"] = field.default_factory()
                elif field.default is dataclasses.MISSING:
                    kwargs["required"] = True
            else:
                kwargs["type"] = field.type
                if field.default is not dataclasses.MISSING:
                    kwargs["default"] = field.default
                elif field.default_factory is not dataclasses.MISSING:
                    kwargs["default"] = field.default_factory()
                else:
                    kwargs["required"] = True
            self.add_argument(field_name, **kwargs)

    def parse_args_into_dataclasses(
            self, args=None, return_remaining_strings=False, look_for_args_file=True, args_filename=None
    ) -> Tuple[DataClass, ...]:
        """
        Parse command-line args into instances of the specified dataclass types.

        This relies on argparse's `ArgumentParser.parse_known_args`. See the doc at:
        docs.python.org/3.7/library/argparse.html#argparse.ArgumentParser.parse_args

        Args:
            args:
                List of strings to parse. The default is taken from sys.argv. (same as argparse.ArgumentParser)
            return_remaining_strings:
                If true, also return a list of remaining argument strings.
            look_for_args_file:
                If true, will look for a ".args" file with the same base name as the entry point script for this
                process, and will append its potential content to the command line args.
            args_filename:
                If not None, will uses this file instead of the ".args" file specified in the previous argument.

        Returns:
            Tuple consisting of:

                - the dataclass instances in the same order as they were passed to the initializer.abspath
                - if applicable, an additional namespace for more (non-dataclass backed) arguments added to the parser
                  after initialization.
                - The potential list of remaining argument strings. (same as argparse.ArgumentParser.parse_known_args)
        """
        if args_filename or (look_for_args_file and len(sys.argv)):
            if args_filename:
                args_file = Path(args_filename)
            else:
                args_file = Path(sys.argv[0]).with_suffix(".args")

            if args_file.exists():
                fargs = args_file.read_text().split()
                args = fargs + args if args is not None else fargs + sys.argv[1:]
                # in case of duplicate arguments the first one has precedence
                # so we append rather than prepend.
        namespace, remaining_args = self.parse_known_args(args=args)
        outputs = []
        for dtype in self.dataclass_types:
            keys = {f.name for f in dataclasses.fields(dtype) if f.init}
            inputs = {k: v for k, v in vars(namespace).items() if k in keys}
            for k in keys:
                delattr(namespace, k)
            obj = dtype(**inputs)
            outputs.append(obj)
        if len(namespace.__dict__) > 0:
            # additional namespace.
            outputs.append(namespace)
        if return_remaining_strings:
            return (*outputs, remaining_args)
        else:
            if remaining_args:
                raise ValueError(f"Some specified arguments are not used by the KGEArgParser: {remaining_args}")

            return (*outputs,)

    def parse_json_file(self, json_file: str) -> Tuple[DataClass, ...]:
        """
        Alternative helper method that does not use `argparse` at all, instead loading a json file and populating the
        dataclass types.
        """
        data = json.loads(Path(json_file).read_text())
        outputs = []
        for dtype in self.dataclass_types:
            keys = {f.name for f in dataclasses.fields(dtype) if f.init}
            inputs = {k: v for k, v in data.items() if k in keys}
            obj = dtype(**inputs)
            outputs.append(obj)
        return (*outputs,)

    def parse_dict(self, args: dict) -> Tuple[DataClass, ...]:
        """
        Alternative helper method that does not use `argparse` at all, instead uses a dict and populating the dataclass
        types.
        """
        outputs = []
        for dtype in self.dataclass_types:
            keys = {f.name for f in dataclasses.fields(dtype) if f.init}
            inputs = {k: v for k, v in args.items() if k in keys}
            obj = dtype(**inputs)
            outputs.append(obj)
        return (*outputs,)

# class KGEArgParser:
#     """The class implements the argument parser for the pykg2vec.
#
#     KGEArgParser defines all the necessary arguments for the global and local
#     configuration of all the modules.
#
#     Attributes:
#         general_group (object): It parses the general arguements used by most of the modules.
#         general_hyper_group (object): It parses the arguments for the hyper-parameter tuning.
#
#     Examples:
#         >>> from toolbox.KGArgs import KGEArgParser
#         >>> args = KGEArgParser().get_args()
#     """
#
#     def __init__(self):
#         self.parser = ArgumentParser(description='Knowledge Graph Embedding tunable configs.')
#
#         ''' argument group for hyperparameters '''
#         self.general_hyper_group = self.parser.add_argument_group('Generic Hyperparameters')
#         self.general_hyper_group.add_argument('-lmda', dest='lmbda', default=0.1, type=float,
#                                               help='The lmbda for regularization.')
#         self.general_hyper_group.add_argument('-b', dest='batch_size', default=128, type=int,
#                                               help='training batch size')
#         self.general_hyper_group.add_argument('-mg', dest='margin', default=0.8, type=float,
#                                               help='Margin to take')
#         self.general_hyper_group.add_argument('-opt', dest='optimizer', default='adam', type=str,
#                                               help='optimizer to be used in training.')
#         self.general_hyper_group.add_argument('-s', dest='sampling', default='uniform', type=str,
#                                               help='strategy to do negative sampling.')
#         self.general_hyper_group.add_argument('-ngr', dest='neg_rate', default=1, type=int,
#                                               help='The number of negative samples generated per positive one.')
#         self.general_hyper_group.add_argument('-l', dest='epochs', default=100, type=int,
#                                               help='The total number of Epochs')
#         self.general_hyper_group.add_argument('-lr', dest='learning_rate', default=0.01, type=float,
#                                               help='learning rate')
#         self.general_hyper_group.add_argument('-k', dest='hidden_size', default=50, type=int,
#                                               help='Hidden embedding size.')
#         self.general_hyper_group.add_argument('-km', dest='ent_hidden_size', default=50, type=int,
#                                               help="Hidden embedding size for entities.")
#         self.general_hyper_group.add_argument('-kr', dest='rel_hidden_size', default=50, type=int,
#                                               help="Hidden embedding size for relations.")
#         self.general_hyper_group.add_argument('-k2', dest='hidden_size_1', default=10, type=int,
#                                               help="Hidden embedding size for relations.")
#         self.general_hyper_group.add_argument('-l1', dest='l1_flag', default=True,
#                                               type=lambda x: (str(x).lower() == 'true'),
#                                               help='The flag of using L1 or L2 norm.')
#         self.general_hyper_group.add_argument('-al', dest='alpha', default=0.1, type=float,
#                                               help='The alpha used in self-adversarial negative sampling.')
#         self.general_hyper_group.add_argument('-fsize', dest='filter_sizes', default=[1, 2, 3], nargs='+', type=int,
#                                               help='Filter sizes to be used in convKB which acts as the widths of the kernals')
#         self.general_hyper_group.add_argument('-fnum', dest='num_filters', default=50, type=int,
#                                               help='Filter numbers to be used in convKB and InteractE.')
#         self.general_hyper_group.add_argument('-fmd', dest='feature_map_dropout', default=0.2, type=float,
#                                               help='feature map dropout value used in ConvE and InteractE.')
#         self.general_hyper_group.add_argument('-idt', dest='input_dropout', default=0.3, type=float,
#                                               help='input dropout value used in ConvE and InteractE.')
#         self.general_hyper_group.add_argument('-hdt', dest='hidden_dropout', default=0.3, type=float,
#                                               help='hidden dropout value used in ConvE.')
#         self.general_hyper_group.add_argument('-hdt1', dest='hidden_dropout1', default=0.4, type=float,
#                                               help='hidden dropout value used in TuckER.')
#         self.general_hyper_group.add_argument('-hdt2', dest='hidden_dropout2', default=0.5, type=float,
#                                               help='hidden dropout value used in TuckER.')
#         self.general_hyper_group.add_argument('-lbs', dest='label_smoothing', default=0.1, type=float,
#                                               help='The parameter used in label smoothing.')
#         self.general_hyper_group.add_argument('-cmax', dest='cmax', default=0.05, type=float,
#                                               help='The parameter for clipping values for KG2E.')
#         self.general_hyper_group.add_argument('-cmin', dest='cmin', default=5.00, type=float,
#                                               help='The parameter for clipping values for KG2E.')
#         self.general_hyper_group.add_argument('-fp', dest='feature_permutation', default=1, type=int,
#                                               help='The number of feature permutations for InteractE.')
#         self.general_hyper_group.add_argument('-rh', dest='reshape_height', default=20, type=int,
#                                               help='The height of the reshaped matrix for InteractE.')
#         self.general_hyper_group.add_argument('-rw', dest='reshape_width', default=10, type=int,
#                                               help='The width of the reshaped matrix for InteractE.')
#         self.general_hyper_group.add_argument('-ks', dest='kernel_size', default=9, type=int,
#                                               help='The kernel size to use for InteractE.')
#         self.general_hyper_group.add_argument('-ic', dest='in_channels', default=9, type=int,
#                                               help='The kernel size to use for InteractE.')
#         self.general_hyper_group.add_argument('-evd', dest='ent_vec_dim', default=200, type=int, help='.')
#         self.general_hyper_group.add_argument('-rvd', dest='rel_vec_dim', default=200, type=int, help='.')
#
#         # basic configs
#         self.general_group = self.parser.add_argument_group('Generic')
#         self.general_group.add_argument('-mn', dest='model_name', default='TransE', type=str, help='Name of model')
#         self.general_group.add_argument('-db', dest='debug', default=False, type=lambda x: (str(x).lower() == 'true'),
#                                         help='To use debug mode or not.')
#         self.general_group.add_argument('-exp', dest='exp', default=False, type=lambda x: (str(x).lower() == 'true'),
#                                         help='Use Experimental setting extracted from original paper. (use Freebase15k by default)')
#         self.general_group.add_argument('-ds', dest='dataset_name', default='Freebase15k', type=str,
#                                         help='The dataset name (choice: fb15k/wn18/wn18_rr/yago/fb15k_237/ks/nations/umls)')
#         self.general_group.add_argument('-dsp', dest='dataset_path', default=None, type=str,
#                                         help='The path to custom dataset.')
#         self.general_group.add_argument('-ld', dest='load_from_data', default=None, type=str,
#                                         help='The path to the pretrained model.')
#         self.general_group.add_argument('-sv', dest='save_model', default=True,
#                                         type=lambda x: (str(x).lower() == 'true'), help='Save the model!')
#         self.general_group.add_argument('-tn', dest='test_num', default=1000, type=int,
#                                         help='The total number of test triples')
#         self.general_group.add_argument('-ts', dest='test_step', default=10, type=int, help='Test every _ epochs')
#         self.general_group.add_argument('-t', dest='tmp', default='../intermediate', type=str,
#                                         help='The folder name to store trained parameters.')
#         self.general_group.add_argument('-r', dest='result', default='../results', type=str,
#                                         help='The folder name to save the results.')
#         self.general_group.add_argument('-fig', dest='figures', default='../figures', type=str,
#                                         help='The folder name to save the figures.')
#         self.general_group.add_argument('-plote', dest='plot_embedding', default=False,
#                                         type=lambda x: (str(x).lower() == 'true'), help='Plot the entity only!')
#         self.general_group.add_argument('-plot', dest='plot_entity_only', default=False,
#                                         type=lambda x: (str(x).lower() == 'true'), help='Plot the entity only!')
#         self.general_group.add_argument('-device', dest='device', default='cpu', type=str, choices=['cpu', 'cuda'],
#                                         help='Device to run pykg2vec (cpu or cuda).')
#         self.general_group.add_argument('-npg', dest='num_process_gen', default=2, type=int,
#                                         help='number of processes used in the Generator.')
#         self.general_group.add_argument('-hpf', dest='hp_abs_file', default=None, type=str,
#                                         help='The path to the hyperparameter configuration YAML file.')
#         self.general_group.add_argument('-ssf', dest='ss_abs_file', default=None, type=str,
#                                         help='The path to the search space configuration YAML file.')
#         self.general_group.add_argument('-mt', dest='max_number_trials', default=100, type=int,
#                                         help='The maximum times of trials for bayesian optimizer.')
#
#     def get_args(self, args):
#         """This function parses the necessary arguments.
#
#         This function is called to parse all the necessary arguments.
#
#         Returns:
#           object: ArgumentParser object.
#         """
#         return self.parser.parse_args(args)
