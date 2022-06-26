"""
@date: 2022/2/20
@description: 监听远程日志服务器
"""
import json
import os
import threading
import time
import traceback
from collections import defaultdict
from typing import List

import requests


class LogAgent:
    def __init__(self):
        self.remote_log_servers = set()

    def set_remote_log_servers(self, remote_log_servers):
        self.remote_log_servers = self.remote_log_servers.union(set(remote_log_servers))

    def read_logs(self, ignore_log_names: dict = None) -> List[dict]:
        data = {
            "ignore_log_names": ignore_log_names
        }
        res = []
        for server in self.remote_log_servers:
            url = f"{server}/logs"
            resp = requests.get(url=url, headers={'Content-Type': 'application/json'}, data=json.dumps(data))
            resp = resp.json()
            res.extend(resp)
        return res

    def read_certain_logs(self, log_dir_names):
        data = {
            "log_dir_names": log_dir_names
        }
        res = []
        for server in self.remote_log_servers:
            url = f"{server}/certain_logs"
            resp = requests.get(url=url, headers={'Content-Type': 'application/json'}, data=json.dumps(data))
            resp = resp.json()
            res.extend(resp)
        return res


def is_log_dir_has_step(_save_log_dir: str, check_files=('metric.log', 'loss.log')) -> bool:
    """
    给定log_dir, 判断是否有step数据

    :param _save_log_dir 日志存放的目录
    :param check_files: 检查file是否含有step
    :return: 是否有step数据
    """
    if not is_dirname_log_record(_save_log_dir):
        return False
    try:
        filenames = check_files
        for filename in filenames:
            filepath = os.path.join(_save_log_dir, filename)
            if not os.path.exists(filepath):
                continue
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith('S'):
                        return True
    except Exception as e:
        traceback.print_exc()
        print(f"Exception raised when read {os.path.abspath(filepath)}")
    return False


def is_dirname_log_record(dir_path: str) -> bool:
    """
    检查dir_path是否是一个合法的log目录。合法的log目录里必须包含meta.log。

    :param dir_path: 被检测的路径
    :return: 是否合法
    """
    if not os.path.isdir(dir_path):
        return False
    return os.path.exists(os.path.join(dir_path, 'meta.log'))  # 至少要有meta.log表明这个是合法的log


def is_log_record_finish(save_log_dir: str) -> bool:
    """
    检测日志的记录是否已经结束

    :param save_log_dir: 日志存放的目录
    :return:
    """
    if is_dirname_log_record(save_log_dir):
        with open(os.path.join(save_log_dir, 'meta.log'), 'r', encoding='utf-8') as f:
            line = ''
            for line in f:
                pass
            if len(line.strip()) != 0:
                try:
                    _d = json.loads(line)
                except:
                    return False
                if 'state' in _d['meta'] and _d['meta']['state'] in ('finish', 'error'):
                    return True
    return False


def flatten_dict(prefix, _dict, connector='-'):
    """
    给定一个dict, 将其展平，比如{"a":{"v": 1}} -> {"a-v":1}

    :param prefix:
    :param _dict:
    :param connector:
    :return:
    """
    new_dict = {}
    for key, value in _dict.items():
        if prefix != '':
            new_prefix = prefix + connector + str(key)
        else:
            new_prefix = str(key)
        if isinstance(value, dict):
            new_dict.update(flatten_dict(new_prefix, value, connector))
        else:
            new_dict[new_prefix] = value
    return new_dict


class StandbyStepLogReader(threading.Thread):
    """
    用于多线程读取日志的类. 配合画图使用的。

    :param save_log_dir: 日志存放的目录
    :param uuid: 用于唯一识别 Reader 的 uuid
    :param wait_seconds:  在文件关闭后再等待{wait_seconds}秒结束进程
    :param max_no_updates: 在{max_no_updates}次都没有更新时结束进程
    """

    def __init__(self, save_log_dir: str, uuid: str, wait_seconds: int = 60, max_no_updates: int = 30):
        super().__init__()

        self.save_log_dir = save_log_dir
        self._file_handlers = {}

        self.uuid = uuid
        self._last_access_time = time.time()
        # 如果这么长时间没有读取到新的数据，就认为是不需要再读取的了
        # 如果这么长时间没有再次调用，就关掉文件
        self._wait_seconds = wait_seconds

        self.unfinish_lines = {}  # 防止读写冲突, key: line
        self._stop_flag = False
        self._quit = False
        self._no_update_count = 0
        self.max_no_update = max_no_updates

        self._last_meta_md_time = None
        self._meta_path = os.path.join(self.save_log_dir, 'meta.log')
        self._total_steps = None

    def _create_file_handler(self, filenames=('metric.log', 'loss.log')):
        """
        检查是否有未加入的handler，有则加入进来

        :return:
        """
        for filename in filenames:
            handler_name = filename.split('.')[0]
            if handler_name in self._file_handlers:
                continue
            filepath = os.path.join(self.save_log_dir, filename)
            handler = open(filepath, 'r', encoding='utf-8')
            self._file_handlers[handler_name] = handler

    def _is_finish_in_meta(self) -> bool:
        """
        检查是否已经在meta中写明了finish的状态了

        :return: bool
        """

        last_meta_md_time = os.path.getmtime(self._meta_path)
        if self._last_meta_md_time is None or self._last_meta_md_time != last_meta_md_time:
            with open(self._meta_path, 'r', encoding='utf-8') as f:
                line = ''
                for line in f:
                    pass
                line = line.strip()
                if len(line) != 0:
                    try:
                        _dict = json.loads(line)['meta']
                        if 'state' in _dict and _dict['state'] in ('finish', 'error'):
                            return True
                    except:
                        pass
        self._last_meta_md_time = last_meta_md_time
        return False

    @staticmethod
    def read_update_single_log(filepaths: List[str], ranges: dict) -> dict:
        """
        调用这个函数，获取filepaths中满足range_min, range_max的log

        :param filepaths: 完整的path路径
        :param ranges: {'metric':[min, max] }
        :return: 返回值的结构如下。loss这个list是进行了step排序的
            {
                loss: [dict('step':x, epoch:value, 'loss':{'loss1':xx})],
                metric:[dict('step':x, epoch:value, 'metric':{'SpanFMetric':{'f':xx}})]
            }

        """
        updates = defaultdict(list)
        for filepath in filepaths:
            filename = os.path.basename(filepath).split('.')[0]
            range_min = int(ranges[filename][0])
            range_max = int(ranges[filename][1])

            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.endswith('\n'):  # 结尾不是回车，说明没有读完
                        pass
                    else:
                        if line.startswith('S'):
                            step = int(line[line.index(':') + 1:line.index('\t')])
                            if range_min <= step <= range_max:
                                line = line[line.index('\t') + 1:].strip()
                            try:
                                _dict = json.loads(line)
                                updates[filename].append(_dict)
                            except:
                                pass
                if filename in updates and len(updates[filename]) != 0:  # 对step排序，保证不要出现混乱
                    updates[filename].sort(key=lambda x: x['step'])
        return updates

    def read_update(self, only_once: bool = False, handler_names=('metric', 'loss')) -> dict:
        """
        调用这个函数，获取新的更新。如果第一次调用则是读取到当前所有的记录。

        :param only_once: 是否只读取内容一次。是的话就不会保持读取到的行数，之后直接退出了
        :param handler_names: 只check包含在handler_name的内容
        :return: 返回值的结构如下
            {
                loss: [dict('step':x, epoch:value, 'loss':{}), ...],  # 或[dict('step':x, epoch:value, 'loss':value), ...]
                metric:[dict('step':x, epoch:value, 'metric':{'SpanFMetric':xxx})],
                finish:bool(not every time),
                total_steps:int(only the first access)
            }

        """
        updates = {}
        if not self._quit:
            flag = False
            self._create_file_handler([fn + '.log' for fn in handler_names])
            updates = defaultdict(list)
            if self._last_access_time is None:
                filepath = os.path.join(self.save_log_dir, 'progress.log')
                if os.path.exists(filepath):
                    with open(filepath, 'r', encoding='utf-8') as f:
                        line = f.readline()
                        try:
                            _dict = json.loads(line.strip())
                            if 'total_steps' in _dict:
                                self._total_steps = _dict['total_steps']
                                updates['total_steps'] = _dict['total_steps']
                        except:
                            pass
                flag = True
            self._last_access_time = time.time()
            for filename, handler in self._file_handlers.items():
                if filename not in handler_names:
                    continue
                for line in handler.readlines():
                    if filename in self.unfinish_lines:
                        line = self.unfinish_lines.pop(filename) + line
                    if not line.endswith('\n'):  # 结尾不是回车，说明没有读完
                        self.unfinish_lines[filename] = line
                    else:
                        if line.startswith('S'):
                            line = line[line.index('\t') + 1:].strip()
                            try:
                                _dict = json.loads(line)
                                updates[filename].append(_dict)
                            except:
                                pass
                if filename in updates and len(updates[filename]) != 0:  # 对step排序，保证不要出现混乱
                    updates[filename].sort(key=lambda x: x['step'])
            if not only_once:
                if len(updates) == 0:
                    self._no_update_count += 1
                else:
                    self._no_update_count = 0
                if flag:
                    self.start()
            else:  # 如果确定只读一次，则直接关闭。应该是finish了
                self._close_file_handler()
                updates['finish'] = True
        if self._quit or self._no_update_count > self.max_no_update:
            updates = {'finish': True}
        if self._is_finish_in_meta():
            updates['finish'] = True
        if 'finish' in updates:
            self._quit = True
            self.stop()

        return updates

    def _close_file_handler(self):
        for key in list(self._file_handlers.keys()):
            handler = self._file_handlers[key]
            handler.close()
        self._file_handlers.clear()

    def stop(self):
        """
        如果手动停止某个任务

        :return:
        """
        self._stop_flag = True
        self._close_file_handler()
        count = 0
        while not self._quit:
            time.sleep(1)
            if count > 3:
                raise RuntimeError("Multi-thread bug here. It should not run twice.")
            count += 1

    def run(self):
        """
        重载了多线程的运行函数

        :return:
        """
        while time.time() - self._last_access_time < self._wait_seconds and not self._stop_flag and \
                self._no_update_count < self.max_no_update:
            time.sleep(0.5)
        print(f"Reader:{self.uuid} for log {os.path.basename(self.save_log_dir)} will quit now.")
        self._quit = True
        self._close_file_handler()


class MultiStandbyStepLogReader(threading.Thread):
    """
    用于multi_chart读取多个log的数据时使用

    """

    def __init__(self, root_log_dir, logs, uuid, wait_seconds: int = 60, max_no_updates: int = 30):
        """

        :param str root_log_dir: root的loader
        :param list[str] logs: 具体的log的名称
        :param str uuid: 一个独特的uuid
        :param int wait_seconds:
        :param int max_no_updates:
        """
        super().__init__()
        self.log_readers = {}
        for log_id in logs:
            self.log_readers[log_id] = StandbyStepLogReader(save_log_dir=os.path.join(root_log_dir, log_id),
                                                            uuid=uuid, wait_seconds=wait_seconds,
                                                            max_no_updates=max_no_updates)
        self._stop_flag = False

    def read_update(self, handler_names=('metric', 'loss')) -> dict:
        """
        调用这个函数，获取新的更新。如果第一次调用则是读取到当前所有的记录。

        :param handler_names: 只check包含在handler_name的内容
        :return: 返回值的结构如下
                {
                    metric-1: {
                        log_1: [
                            [value, step, epoch],
                            []
                        ]
                    }
                    ...
                }
        """
        results = defaultdict(dict)
        """
            {
                loss: [dict('step':x, epoch:value, 'loss':{})],
                metric:[dict('step':x, epoch:value, 'metric':{'SpanFMetric':{}}})],
                finish:bool(not every time),
                total_steps:int(only the first access)
            }
        """

        results['finish_logs'] = []
        for log_id in list(self.log_readers.keys()):
            reader = self.log_readers.get(log_id, None)
            if reader is None:
                results['finish_logs'].append(log_id)
                continue
            log_res = reader.read_update(only_once=False, handler_names=handler_names)
            if 'finish' in log_res:
                results['finish_logs'].append(log_id)
            for handler_name in handler_names:
                if handler_name in log_res:
                    res_lst = log_res[handler_name]
                    for _dict in res_lst:
                        sub_dict = _dict[handler_name]
                        step = _dict.get('step')
                        epoch = _dict.get('epoch', -1)  # 可能会没有这个值
                        flat_dict = flatten_dict(handler_name, sub_dict, connector='-')
                        for key, value in flat_dict.items():
                            if key not in ('metric-epoch', 'metric-step'):
                                if log_id not in results[key]:
                                    results[key][log_id] = []
                                results[key][log_id].append([value, step, epoch])
        return results

    def run(self) -> None:
        for key in list(self.log_readers.keys()):
            self.log_readers[key].start()  # 将这个线程开始起来
        while len(self.log_readers) > 0 and not self._stop_flag:
            for key in list(self.log_readers.keys()):
                reader = self.log_readers[key]
                if reader._quit:
                    self.log_readers.pop(key)
            time.sleep(0.5)

    @property
    def _quit(self):
        if len(self.log_readers) == 0:
            return True
        return False

    def stop(self):
        for key in list(self.log_readers.keys()):
            reader = self.log_readers[key]
            reader.stop()
        self._stop_flag = True


log_agent = LogAgent()
