"""
@author: lxy
@email: linxy59@mail2.sysu.edu.cn
@date: 2022/2/19
@description: null
"""
import argparse
import json
import logging
import os
import re
import time
from configparser import ConfigParser
from copy import deepcopy
from typing import Union

import numpy as np


class MetricLogStoreSchema:
    """实验指标日志
    需要以特定的格式存储和读取
    """

    def __init__(self, log_dir: str):
        self._logger = Logger()
        self.set_log_dir(log_dir)

    def set_log_dir(self, log_dir: str, new_log: bool = False):
        """
        设定log 文件夹的路径(在进行其它操作前必须先指定日志路径)。如果你已经顺利执行了 fitlog.commit()命令，
        log 文件夹会自动设定为.fitconfig 文件中的 default_log_dir 字段的值。在某些情况下，可能需要继续往同
        一个log中写入数据(比如继续训练之前以及保存的模型)，可以通过将log_dir设置为具体的log名。但需要保证
        step的顺序与之前已有的内容是不冲突的，因为相同的step在fitlog中是覆盖的。

        Example::

            # 假设当前的文件结构为
            # logs/
            #    log_20190417_140311
            #    ...
            # main.py
            #以下是main.py中三种设置log位置的方式
            fitlog.commit() # 如果commit成功，则不需要设置logs文件夹了
            fitlog.set_log_dir('logs/') # 设置log文件夹为'logs/', fitlog在每次运行的时候会默认以时间戳的方式在里面生成新的log
            fitlog.set_log_dir('logs/log_20190417_140311') # fitlog将log继续写入到log_20190417_140311里。

        :param log_dir: log 文件夹的路径
        :param new_log: 是否重新创建一个log，仅在同一次python启动但是需要记录多个log时使用(但是只能分阶段地用，即同一时间
            只会有一个logger存在，设置new_log为True时，仅仅是开了一个新的logger，但同时前一个就关闭了。)同一次启动中fit_id以及
            git_id只会在第一次启动时获取，之后的新log只是使用第一次提交的fit_id与git_id
        """
        self._logger.set_log_dir(log_dir, new_log)

    def debug(self, flag=True):
        """
        调用该方法之后，所有的fitlog方法都不会产生任何作用。可用于调试代码时避免输出大量无用的信息。

        Example::

            fitlog.debug()
            fitlog.commit()
            fitlog.add_metric(0.3, f1)

        由于有fitlog.debug(), commit()和add_metric()都不会实际执行的。

        :return:
        """
        self._logger.debug(flag=flag)

    def is_debug(self):
        """
        返回当前是否处于debug状态

        """
        return self._logger.is_debug()

    def finish(self, status: int = 0, send_to_bot: str = None):
        """
            使用此方法告知 fitlog 你的实验已经正确结束。你可以使用此方法来筛选出失败的实验。

            :param int status: 告知当前实验的状态。0: 结束了; 1: 发生了错误
            :param str send_to_bot: 数据上报
        """
        self._logger.finish(status, send_to_bot)

    def add_metric(self, value: Union[int, str, float, dict], step: int, name: str = None, epoch: int = None):
        """
        用于添加 metric 。用此方法添加的值不会显示在表格中，但可以在单次训练的详情曲线图中查看。

        :param value: 类型为 int, float, str, dict中的一种。如果类型为 dict，它的键的类型只能为 str，
                它的键值的类型可以为int, float, str 或符合同样条件的 dict
        :param step: 用于和 loss 对应的 step
        :param name: 如果你传入 name 参数，你传入的 value 参数会被看做形如 {name:value} 的字典
        :param epoch: 前端显示需要记录 epoch
        :return:
        """
        self._logger.add_metric(value, step, name, epoch)

    def add_loss(self, value: Union[int, str, float, dict], step: int, name: str = "loss", epoch: int = None):
        """
        用于添加 loss。用此方法添加的值不会显示在表格中，但可以在单次训练的详情曲线图中查看。

        :param value: 类型为 int, float, str, dict中的一种。如果类型为 dict，它的键的类型只能为 str，
                它的键值的类型可以为int, float, str 或符合同样条件的 dict
        :param step: 用于和 loss 对应的 step
        :param name: 如果你传入 name 参数，你传入的 value 参数会被看做形如 {name:value} 的字典
        :param epoch: 前端显示需要记录 epoch
        :return:
        """
        self._logger.add_loss(value, step, name, epoch)

    def add_best_metric(self, value: Union[int, str, float, dict], name: str = None):
        """
        用于添加最好的 metric 。用此方法添加的值，会被显示在表格中的 metric 列及其子列中。相同key的内容将只保留最后一次传入的值。

        :param value: 类型为 int, float, str, dict中的一种。如果类型为 dict，它的键的类型只能为 str，
                它的键值的类型可以为int, float, str 或符合同样条件的 dict
        :param name: 如果你传入 name 参数，你传入的 value 参数会被看做形如 {name:value} 的字典

        .. warning ::
            如果你在同时记录多个数据集上的performance, 请注意使用不同的名称进行区分

        """
        self._logger.add_best_metric(value, name)

    def add_hyper(self, value: Union[int, str, float, dict, argparse.Namespace, ConfigParser], name=None):
        """
        用于添加超参数。用此方法添加到值，会被放置在表格中的 hyper 列及其子列中

        :param value: 类型为 int, float, str, dict, argparse.Namespace(即ArgumentParser传入的内容), ConfigParser中的一种
                。如果类型为 dict，它的键的类型只能为 str，它的键值的类型可以为int, float, str 或符合同样条件的 dict
        :param name: 如果你传入 name 参数，你传入的 value 参数会被看做形如 {name:value} 的字典
        :return:
        """
        self._logger.add_hyper(value, name)

    def add_hyper_in_file(self, file_path: str):
        """
        从文件读取参数。如下面的文件所示，两行"#####hyper"(至少5个#)之间的参数会被读取出来，并组成一个字典。每个变量最多只能出现在一行中，
        如果多次出现，只会记录第一次出现的值。demo.py::

            from numpy as np
            import fitlog
            # do something

            fitlog.add_hyper_in_file(__file__)  # 会把本python文件的hyper加入进去
            ############hyper
            lr = 0.01 # some comments
            char_embed = word_embed = 300

            hidden_size = 100
            ....
            ############hyper

            # do something
            model = Model(xxx)

        如果你把 demo.py 的文件路径传入此函数，会转换出如下字典，并添加到参数中::

            {
                'lr': '0.01',
                'char_embed': '300'
                'word_embed': '300'
                'hidden_size': '100'
            }

        :param file_path: 文件路径。如果是读取本python文件中的hyper parameter可以直接fitlog.add_hyper_in_file(__file__)
        """
        self._logger.add_hyper_in_file(file_path)

    def add_other(self, value: Union[int, str, float, dict], name: str = None):
        """
        用于添加其它参数。用此方法添加到值，会被放置在表格中的 other 列及其子列中。相同key的内容将只保留最后一次传入的值。

        :param value: 类型为 int, float, str, dict中的一种。如果类型为 dict，它的键的类型只能为 str，
                它的键值的类型可以为int, float, str 或符合同样条件的 dict
        :param name: 如果你传入 name 参数，你传入的 value 参数会被看做形如 {name:value} 的字典
        """
        self._logger.add_other(value, name)

    def add_progress(self, total_steps: int = None):
        """
        传入总的step数量，用于前端计算进度。

        :param total_steps: int, 总共有多少个step
        """
        self._logger.add_progress(total_steps)

    def add_to_line(self, line: Union[str, dict]):
        """
        将str记录到文件中，前端可以从网页跳转打开文件。每次记录是append到之前的记录之后的。

        :param line: 字符串类型或字典类型的数据，将直接写到文件中
        :return:
        """
        self._logger.add_to_file(line)

    def create_log_folder(self):
        """
        默认是生成第一个loss或者metric的时候才会在设置的log文件夹下创建一个新的文件夹，如果需要在代码运行时就创建该文件夹，可以通过
            调用该接口。

        :return:
        """
        self._logger.create_log_folder()

    def set_rng_seed(self, rng_seed: int = None, random: bool = True, numpy: bool = True,
                     pytorch: bool = True, deterministic: bool = True):
        """
        设置模块的随机数种子。由于pytorch还存在cudnn导致的非deterministic的运行，所以一些情况下可能即使seed一样，结果也不一致
        随机种子也算超参数，会影响指标，所以需要记录到日志
        :param int rng_seed: 将这些模块的随机数设置到多少，默认为随机生成一个0-1000,000的随机数。
        :param bool, random: 是否将python自带的random模块的seed设置为rng_seed.
        :param bool, numpy: 是否将numpy的seed设置为rng_seed.
        :param bool, pytorch: 是否将pytorch的seed设置为rng_seed(设置torch.manual_seed和torch.cuda.manual_seed_all).
        :param bool, deterministic: 是否将pytorch的torch.backends.cudnn.deterministic设置为True。如果该值不为True，有时候即使
            全部随机数种子都一样也不能跑出相同的结果; 关掉的话可能会有一点性能损失。
        """
        return self._logger.set_rng_seed(rng_seed, random, numpy, pytorch, deterministic)


def _check_debug(func):
    """
    函数闭包，只有非 debug 模式才会执行原始函数

    :param func: 原始函数，函数的第一个参数必须为 Logger 对象
    :return: 加上闭包后的函数
    """

    def wrapper(*args, **kwargs):
        if args[0].is_debug():
            return
        else:
            return func(*args, **kwargs)

    return wrapper


def _check_log_dir(func):
    """
    函数闭包，检查原始函数执行所需的条件是否满足，只有满足才会执行

    1 如果没有initialize, 说明还没有设置

    2 如果default_log_dir不为None，设置使用default_log_dir调用set_log_dir

    3 否则报错

    :param func: 原始函数，函数的第一个参数必须为 Logger 对象
    :return: 加上闭包后的函数
    """

    def wrapper(*args, **kwargs):
        if not args[0].initialized:
            raise RuntimeError("You have to call `fitlog.set_log_dir()` to set where to save log first.")
        return func(*args, **kwargs)

    return wrapper


class Logger:
    """
    用于处理日志的类，fitlog 的核心
    """

    def __init__(self):
        self.initialized = False
        self.save_on_first_metric_or_loss = True
        self._cache = []
        self._debug = False
        self.fit_id = None
        self._save_log_dir = None  # 存在哪个文件内的，比如log_20191020_193021/。如果
        self._start_time = time.time()

    @_check_log_dir
    def get_log_folder(self, absolute=False):
        """
        返回实际保存log的文件夹，类似log_20200406_055218/这种

        :param bool absolute: 是否返回绝对路径
        :return:
        """
        log_dir = self._save_log_dir
        if absolute:
            if log_dir:
                log_dir = os.path.abspath(log_dir)
        else:
            if log_dir:
                log_dir = os.path.basename(log_dir)
        return log_dir

    def debug(self, flag=True):
        """
        再引入logger之后就调用，本次运行不会记录任何东西。所有函数无任何效用

        :return:
        """
        self._debug = flag

    @_check_debug
    @_check_log_dir
    def create_log_folder(self):
        """
        默认是生成第一个loss或者metric的时候才会在设置的log文件夹下创建一个新的文件夹，如果需要在代码运行时就创建该文件夹，可以通过调用该接口。

        :return:
        """
        self._create_log_files()

    @_check_debug
    def set_log_dir(self, log_dir: str, new_log: bool = False):
        """
        设定log 文件夹的路径，在进行其它操作前必须先指定日志路径

        :param log_dir: log 文件夹的路径
        :param new_log: 是否开始新的一条log记录. 一般用于同一次实验需要记录多个数据集的performance
        :return:
        """
        if new_log:
            self._clear()

        if not os.path.exists(log_dir):
            raise FileNotFoundError(f"`{log_dir}` is not exist.")
        if not os.path.isdir(log_dir):
            raise NotADirectoryError(f"`{log_dir}` is not a directory.")
        if not os.access(log_dir, os.W_OK):
            raise PermissionError(f"write is not allowed in `{log_dir}`. Check your permission.")

        # prepare file directory
        self._save_log_dir = log_dir
        self.initialized = True
        self.create_log_folder()
        self._start_time = time.time()

    def _clear(self):
        """
        内部函数，将logger置为未初始化
        :return:
        """
        # self._save()
        self.initialized = False
        self._cache = []
        for attr_name in ['total_steps']:
            if hasattr(self, attr_name):
                delattr(self, attr_name)
        for attr_name in ['_save_log_dir']:
            setattr(self, attr_name, None)

        for logger_name in ['meta_logger', 'hyper_logger', 'metric_logger', 'other_logger', 'progress_logger',
                            'loss_logger', "best_metric_logger", "file_logger"]:
            if hasattr(self, logger_name):
                _logger = getattr(self, logger_name)
                handlers = _logger.handlers[:]
                for handler in handlers:
                    handler.close()
                    handler.flush()
                    _logger.removeHandler(handler)
                delattr(self, logger_name)

    def _create_log_files(self):
        """
        创建日志文件
        """
        if not hasattr(self, 'meta_logger'):
            # prepare logger
            formatter = logging.Formatter('%(message)s')  # 只保存记录的时间与记录的内容
            for name in ['meta', 'hyper', 'metric', 'other', 'loss', 'progress', 'best_metric', 'file']:
                logger_name = f'metric_log_{name}'
                logger = logging.getLogger(logger_name)
                handler = logging.FileHandler(os.path.join(self._save_log_dir, f'{name}.log'), encoding='utf-8')
                handler.setFormatter(formatter)
                handler.setLevel(logging.INFO)
                logger.setLevel(logging.INFO)
                logger.propagate = False
                logger.addHandler(handler)
                setattr(self, name + '_logger', logger)
            self.__add_meta()

    @_check_debug
    @_check_log_dir
    def __add_meta(self):
        """
        logger自动调用此方法添加meta信息
        """
        _dict = {}
        _dict["state"] = 'running'
        _dict = {'meta': _dict}
        self._write_to_logger(json.dumps(_dict), 'meta_logger')

    @_check_debug
    @_check_log_dir
    def finish(self, status: int = 0, send_to_bot: str = None):
        """
        使用此方法告知 fitlog 你的实验已经正确结束。你可以使用此方法来筛选出失败的实验。

        :param status: 告知当前实验的状态。0: 结束了; 1: 发生了错误
        :param send_to_bot: 飞书机器人的 webhook 地址，设置后可以
        :return:
        """
        if status not in (0, 1):
            raise ValueError("status only supports 0,1 to stand for 'finish','error'.")
        if hasattr(self, 'meta_logger'):
            if status == 0:
                _dict = {'meta': {'state': 'finish'}}
            else:
                _dict = {'meta': {'state': 'error'}}
            self._write_to_logger(json.dumps(_dict), 'meta_logger')
        self.add_other(value=get_hour_min_second(time.time() - self._start_time), name='cost_time')

        if send_to_bot is not None:
            if isinstance(send_to_bot, str):
                if status == 0:
                    title = "[ fitlog 训练完成 ]"
                    text = "fitlog 提醒您：您的训练任务已完成！"
                else:
                    title = "[ fitlog 训练错误 ]"
                    text = "fitlog 提醒您：您的训练任务发生了错误。"
                data = {
                    "msg_type": "post",
                    "content": {
                        "post": {
                            "zh_cn": {
                                "title": title,
                                "content": [
                                    [
                                        {
                                            "tag": "text",
                                            "text": text
                                        },
                                    ]
                                ]
                            }
                        }
                    }
                }
                import requests
                requests.post(url=send_to_bot, headers={'Content-Type': 'application/json'}, data=json.dumps(data))
            else:
                print("[send_to_bot] 应该设置为飞书机器人的 webhook 地址")

    @_check_debug
    @_check_log_dir
    def add_best_metric(self, value: Union[int, str, float, dict], name: str = None):
        """
        用于添加最好的 metric 。用此方法添加的值，会被显示在 metric 这一列中。

        :param value: 类型为 int, float, str, dict中的一种。如果类型为 dict，它的键的类型只能为 str，
                它的键值的类型可以为int, float, str 或符合同样条件的 dict
        :param name: 如果你传入的 value 不是字典，你需要传入 value 对应的名字。

        .. warning ::
            如果你在同时记录多个数据集上的performance, 请注意使用不同的名称进行区分

        """
        _dict = _parse_value(value, name=name, parent_name='metric')
        self._write_to_logger(json.dumps(_dict), 'best_metric_logger')

    @_check_debug
    @_check_log_dir
    def add_to_file(self, value: Union[str, dict]):
        """
        将str记录到文件中，前端可以从网页跳转打开文件。记录是append到之前的记录之后。每个str之后会自动添加一个换行符

        :param value: 字符串类型的数据，将直接写到文件中
        :return:
        """
        assert isinstance(value, (str, dict)), "Only str or dict allowed, not {}.".format(type(value))
        if isinstance(value, dict):
            value = json.dumps(value, indent=2)
        self._write_to_logger(value, 'file_logger')

    @_check_debug
    @_check_log_dir
    def add_metric(self, value: Union[int, str, float, dict], step: int, name: str = None, epoch: int = None):
        """
        用于添加 metric 。用此方法添加的值，会被记录在 metric 这一列中

        :param value: 类型为 int, float, str, dict中的一种。如果类型为 dict，它的键的类型只能为 str，
                它的键值的类型可以为int, float, str 或符合同样条件的 dict
        :param step: 用于和 loss 对应的 step
        :param name: 如果你传入的 value 不是字典，你需要传入 value 对应的名字
        :param epoch: 前端显示需要记录 epoch
        :return:
        """
        assert isinstance(step, int) and step > -1, "Only positive integer is allowed to be `step`."
        _dict = _parse_value(value, name, parent_name='metric')
        _dict['step'] = step
        if epoch is not None:
            assert isinstance(epoch, int) and epoch > -1, "Only positive integer is allowed to be `epoch`."
            _dict['epoch'] = epoch
        _str = json.dumps(_dict)
        _str = 'Step:{}\t'.format(step) + _str
        self._write_to_logger(_str, 'metric_logger')

    @_check_debug
    @_check_log_dir
    def add_loss(self, value: Union[int, str, float, dict], step: int, name: str = "loss", epoch: int = None):
        """
        用于添加 loss。用此方法添加的值，可以通过曲线看出去变化趋势。

        :param value: 类型为 int, float, str, dict中的一种。如果类型为 dict，它的键的类型只能为 str，
                它的键值的类型可以为int, float, str 或符合同样条件的 dict
        :param step: 用于和 loss 对应的 step
        :param name: 如果你传入的 value 不是字典，你需要传入 value 对应的名字
        :param epoch: 前端显示需要记录 epoch
        :return:
        """
        assert isinstance(step, int) and step > -1, "Only positive integer is allowed to be `step`."
        _dict = _parse_value(value, name, parent_name='loss')
        if epoch is not None:
            assert isinstance(epoch, int) and epoch > -1, "Only positive integer is allowed to be `epoch`."
            _dict['epoch'] = epoch
        _dict['step'] = step
        _str = json.dumps(_dict)
        _str = 'Step:{}\t'.format(step) + _str
        self._write_to_logger(_str, 'loss_logger')  # {'loss': {}, 'step':xx, 'epoch':xx}

    @_check_debug
    @_check_log_dir
    def add_hyper(self, value: Union[int, str, float, dict, argparse.Namespace, ConfigParser], name=None):
        """
        用于添加超参数。用此方法添加到值，会被放置在 hyper 这一列中

        :param value: 类型为 int, float, str, dict, argparse.Namespace(即ArgumentParser传入的内容), ConfigParser中的一种
                。如果类型为 dict，它的键的类型只能为 str，它的键值的类型可以为int, float, str 或符合同样条件的 dict
        :param name: 如果你传入的 value 不是字典，你需要传入 value 对应的名字
        :return:
        """
        if isinstance(value, argparse.Namespace):
            value = vars(value)
            value = deepcopy(value)
            _check_dict_value(value)
        elif isinstance(value, ConfigParser):
            value = _convert_configparser_to_dict(value)  # no need to check

        _dict = _parse_value(value, name=name, parent_name='hyper')

        self._write_to_logger(json.dumps(_dict), 'hyper_logger')

    @_check_debug
    @_check_log_dir
    def add_other(self, value: Union[int, str, float, dict], name: str = None):
        """
        用于添加其它参数

        :param value: 类型为 int, float, str, dict中的一种。如果类型为 dict，它的键的类型只能为 str，
                它的键值的类型可以为int, float, str 或符合同样条件的 dict
        :param name: 如果你传入 name 参数，你传入的 value 参数会被看做形如 {name:value} 的字典
        :return:
        """
        if name in ('meta', 'hyper', 'metric', 'loss') and not isinstance(value, dict):
            raise KeyError("Don't use {} as a name. Use fitlog.add_{}() to save it.".format(name, name))

        _dict = _parse_value(value, name=name, parent_name='other')
        self._write_to_logger(json.dumps(_dict), 'other_logger')

    @_check_debug
    @_check_log_dir
    def add_hyper_in_file(self, file_path: str):
        """
        从文件读取参数。如demo.py所示，两行"#######hyper"(至少5个#)之间的参数会被读取出来，并组成一个字典。每个变量最多只能出现在一行中，
        如果多次出现，只会记录第一次出现的值。另外等号最右侧的不能是一个变量，fitlog无法知道变量取什么值。demo.py::

            from numpy as np
            # do something

            ############hyper
            lr = 0.01 # some comments
            char_embed = word_embed = 300
            # char_embed = args.char_embed  # 非法的，不支持变量赋值

            hidden_size = 100
            # num_layers = 3 # 这个值不会被记录，通过#注释掉的行将被忽略
            ....
            ############hyper

            # do something
            model = Model(xxx)

        如果你把 demo.py 的文件路径传入此函数，会转换出如下字典，并添加到参数中::

            {
                'lr': '0.01',
                'char_embed': '300'
                'word_embed': '300'
                'hidden_size': '100'
            }


        :param file_path: 文件路径
        :return:
        """
        file_path = os.path.abspath(file_path)
        if not os.path.isfile(file_path):
            raise RuntimeError("{} is not a regular file.".format(file_path))
        if not file_path.endswith('.py'):
            raise RuntimeError("{} is not a python file.".format(file_path))
        _dict = {}
        between = False
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if len(re.findall('^#####+hyper$', line)) != 0:
                    between = not between
                elif between:
                    if len(line) != 0 and not line.startswith('#'):
                        line = re.sub(r'#[^#]*$', '', line).strip()  # 删除结尾的注释
                        # replace space before an after =
                        line = re.sub(r'\s*=\s*', '=', line)
                        values = line.split('=')
                        # 删除str开头结尾的'"
                        last_value = values[-1].rstrip("'").rstrip('"').lstrip("'").lstrip('"')
                        if last_value == 'False':
                            last_value = False
                        elif last_value == 'True':
                            last_value = True
                        for value in values[:-1]:
                            _dict[value] = last_value
        if len(_dict) != 0:
            self.add_hyper(_dict)

    @_check_debug
    @_check_log_dir
    def add_progress(self, total_steps: int = None):
        """
        用于前端显示当前进度条。传入总的step数量

        :param total_steps: int, 总共有多少个step
        :return:
        """
        assert isinstance(total_steps, int) and total_steps > 0
        if hasattr(self, 'total_steps'):
            raise RuntimeError("Cannot set total_steps twice.")
        self.total_steps = total_steps
        self._write_to_logger(json.dumps({"total_steps": total_steps}), 'progress_logger')

    def set_rng_seed(self, rng_seed: int = None, random: bool = True, numpy: bool = True,
                     pytorch: bool = True, deterministic: bool = True):
        """
        设置模块的随机数种子。
        由于pytorch还存在cudnn导致的非deterministic的运行，所以一些情况下可能即使seed一样，结果也不一致
        需要在fitlog.commit()或fitlog.set_log_dir()之后运行才会记录该rng_seed到log中
        :param int rng_seed: 将这些模块的随机数设置到多少，默认为随机生成一个。
        :param bool, random: 是否将python自带的random模块的seed设置为rng_seed.
        :param bool, numpy: 是否将numpy的seed设置为rng_seed.
        :param bool, pytorch: 是否将pytorch的seed设置为rng_seed(设置torch.manual_seed和torch.cuda.manual_seed_all).
        :param bool, deterministic: 是否将pytorch的torch.backends.cudnn.deterministic设置为True
        """
        if rng_seed is None:
            import time
            import math
            rng_seed = int(math.modf(time.time())[0] * 1000000)
        if random:
            import random
            random.seed(rng_seed)
        if numpy:
            try:
                import numpy
                numpy.random.seed(rng_seed)
            except:
                pass
        if pytorch:
            try:
                import torch
                torch.manual_seed(rng_seed)
                torch.cuda.manual_seed(rng_seed)
                torch.cuda.manual_seed_all(rng_seed)
                if deterministic:
                    torch.backends.cudnn.deterministic = True
            except:
                pass
        if self.initialized:
            self.add_other(rng_seed, 'rng_seed')
        os.environ['PYTHONHASHSEED'] = str(rng_seed)  # 为了禁止hash随机化，使得实验可复现。
        return rng_seed

    @_check_debug
    @_check_log_dir
    def _save(self):
        if len(self._cache) != 0:
            self._create_log_files()
            for value, logger_name in self._cache:
                _logger = getattr(self, logger_name)
                _logger.info(value)
            self._cache = []

    def _write_to_logger(self, _str: str, logger_name: str):
        """
        把记录的内容写到logger里面`

        :param _str: 要记录的内容
        :param logger_name: 所用logger的名称
        :return:
        """
        assert isinstance(logger_name, str) and isinstance(_str, str)
        if self._save_log_dir is None:
            if logger_name in ('metric_logger', 'best_metric_logger', 'loss_logger'):
                self._create_log_files()
                self._save()  # 将之前的内容存下来
        if logger_name not in ('file_logger',):
            _str = re.sub('-(?!\d)', '_', _str.replace('\n', ' '))
        if hasattr(self, logger_name):
            _logger = getattr(self, logger_name)
            _logger.info(_str)
        else:  # 如果还没有初始化就先cache下来
            self._cache.append([_str, logger_name])

    def is_debug(self):
        """
        返回当前是否是debug状态
        """
        return self._debug


def _convert_configparser_to_dict(config: ConfigParser) -> dict:
    """
    将ConfigParser类型的对象转成字典

    :param config: 代转换的对象
    :return: 转换成的字典
    """
    _dict = {}
    for section in config.sections():
        __dict = {}
        options = config.options(section)
        for option in options:
            __dict[option] = config.get(section, option)
        _dict[section] = __dict

    return _dict


def _parse_value(value: Union[int, str, float, dict, np.ndarray], name: str, parent_name: str = None) -> dict:
    """
    检查传入的value是否是符合要求的。并返回dict

    1 如果value是基本类型，则name不为None
    2 如果value是dict类型，则保证所有value是可以转为(int, str, float)的

    :param value: int, float, str或者dict类型
    :param name:
    :param parent_name:
    :return:
    """
    if name is not None:
        assert isinstance(name, str), f"name can only be `str` type, not {name}."
    _dict = {}

    if isinstance(value, (int, float, str, bool)) or value is None:
        if name is None:
            raise RuntimeError(f"When value is {type(value)}, you must pass `name`.")
    elif isinstance(value, dict):
        _check_dict_value(value)
    elif 'torch.Tensor' in str(type(value)):
        assert name is not None, f"When value is `{type(value)}`, you must pass a name."
        try:
            value = value.item()
        except:
            value = str(value.tolist())
    elif 'numpy.ndarray' in str(type(value)):
        assert name is not None, f"When value is `{type(value)}`, you must pass a name."
        total_ele = 1
        for dim in value.shape:
            total_ele *= dim
        if total_ele == 1:
            value = value.reshape(1)[0]
        else:
            value = str(value.tolist())
    elif isinstance(value, np.bool_):
        value = bool(value)
    elif isinstance(value, np.integer):
        value = int(value)
    elif isinstance(value, np.floating):
        value = float(value)
    else:
        value = str(value)  # 直接专为str类型
        assert name is not None, f"When value is `{type(value)}`, you must pass a name."
    if parent_name is not None and name is not None:
        _dict = {parent_name.replace(' ', '_'): {name.replace(' ', '_'): value}}
    elif parent_name is not None:
        _dict = {parent_name.replace(' ', '_'): value}
    elif name is not None:
        _dict = {name.replace(' ', '_'): value}
    else:
        _dict = value
    return _dict


def _check_dict_value(_dict: dict, prefix: str = ''):
    """
    递归检查字典中任意字段的值是否符合要求

    :param _dict: 被检查的字典
    :param prefix: 递归时键值的前缀
    :return:
    """
    keys = list(_dict.keys())
    for key in keys:
        value = _dict[key]
        if isinstance(value, (np.str, str)) or value is None:
            continue
        elif isinstance(value, dict):
            _check_dict_value(value, prefix=prefix + ':' + key)
        elif 'torch.Tensor' in str(type(value)):
            try:
                value = value.item()
                _dict[key] = value
            except:
                value = str(value.tolist())
                _dict[key] = value
        elif 'numpy.ndarray' in str(type(value)):
            total_ele = 1
            for dim in value.shape:
                total_ele *= dim
            if total_ele == 1:
                _dict[key] = value.reshape(1)[0]
            else:
                _dict[key] = str(value.tolist())
        elif isinstance(value, (np.bool_, bool)):
            _dict[key] = bool(value)
        elif isinstance(value, (np.integer, int)):
            _dict[key] = int(value)
        elif isinstance(value, (np.floating, float)):
            _dict[key] = float(value)
        else:
            _dict[key] = str(value)


def get_hour_min_second(seconds):
    # seconds: int
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    f = ''
    f += '{:d}h'.format(int(h))
    f += '{:d}m'.format(int(m))
    f += '{:d}s'.format(s)
    return f
