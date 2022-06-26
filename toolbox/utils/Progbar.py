"""
@date: 2022/2/19
@description: 进度条
"""
import sys
import time
from typing import Dict, Any, Union, Tuple, List
import datetime
import numpy as np


class Progbar(object):
    """Progress bar class inspired by keras 进度条

        Examples:
           >>> from toolbox.utils.Progbar import Progbar
           >>> progbar = Progbar(max_step=100)
           >>> for i in range(100):
           >>>     progbar.update(i, [("step", i), ("next", i+1)])
    """

    def __init__(self, max_step: int, width: int = 15, mode: str = "instant"):
        self.max_step: int = max_step
        self.width: int = width
        self.mode: str = mode
        self.last_width: int = 0

        self.sum_values: Dict[str, Any] = {}

        self.start: float = time.time()
        self.last_step: int = 0

        self.info: str = ""
        self.bar: str = ""

    def _update_values(self, curr_step: int, values: List[Tuple[str, Union[float, str, int]]]):
        for k, v in values:
            if k not in self.sum_values:
                if isinstance(v, float) or isinstance(v, int):
                    if self.mode == "instant":
                        self.sum_values[k] = v
                    else:
                        self.sum_values[k] = [v * (curr_step - self.last_step), curr_step - self.last_step]
                elif isinstance(v, str):
                    self.sum_values[k] = (v + "              ")[:20]
                else:
                    self.sum_values[k] = (str(v) + "              ")[:20]
            else:
                if isinstance(v, float) or isinstance(v, int):
                    if self.mode == "instant":
                        self.sum_values[k] = v
                    else:
                        self.sum_values[k][0] += v * (curr_step - self.last_step)
                        self.sum_values[k][1] += (curr_step - self.last_step)
                elif isinstance(v, str):
                    self.sum_values[k] = (v + "              ")[:20]
                else:
                    self.sum_values[k] = (str(v) + "              ")[:20]

    def _write_bar(self, curr_step: int):
        last_width = self.last_width
        sys.stdout.write("\b" * last_width)
        sys.stdout.write("\r")

        numdigits = int(np.floor(np.log10(self.max_step))) + 1
        barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
        bar = barstr % (curr_step, self.max_step)
        prog = float(curr_step) / self.max_step
        prog_width = int(self.width * prog)
        if prog_width > 0:
            bar += ('=' * (prog_width - 1))
            if curr_step < self.max_step:
                bar += '>'
            else:
                bar += '='
        bar += ('.' * (self.width - prog_width))
        bar += ']'
        sys.stdout.write(bar)

        return bar

    def _get_eta(self, curr_step: int):
        now = time.time()
        if curr_step:
            time_per_step = (now - self.start) / curr_step
        else:
            time_per_step = 0
        eta = time_per_step * (self.max_step - curr_step)

        if curr_step < self.max_step:
            info = ' - ETA: %s' % str(datetime.timedelta(seconds=eta))
        else:
            info = ' - %s' % str(datetime.timedelta(seconds=now - self.start))

        return info

    def _get_values_sum(self):
        info = ""
        for name, value in self.sum_values.items():
            if isinstance(value, str):
                info += ' - %s: %s' % (name, value)
            else:
                if self.mode == "instant":
                    if isinstance(value, int):
                        info += ' - %s: %d' % (name, value)
                    else:
                        info += ' - %s: %.6f' % (name, value)
                else:
                    info += ' - %s: %.6f' % (name, value[0] / max(1, value[1]))
        return info

    def _write_info(self, curr_step: int):
        info = ""
        info += self._get_eta(curr_step)
        info += self._get_values_sum()

        sys.stdout.write(info)

        return info

    def _update_width(self, curr_step: int):
        curr_width = len(self.bar) + len(self.info)
        if curr_width < self.last_width:
            sys.stdout.write(" " * (self.last_width - curr_width))

        if curr_step >= self.max_step:
            sys.stdout.write("\n")

        sys.stdout.flush()

        self.last_width = curr_width

    def update(self, curr_step: int, values: Union[Dict[str, Any], List[Tuple[str, Any]]]):
        """Updates the progress bar.
        The progress bar will display averages for these values.

        Args:
            values: Dict or List of tuples (name, value_for_last_step).
        """
        if isinstance(values, dict):
            values = [(k, v) for k, v in values.items()]
        self._update_values(curr_step, values)
        self.bar = self._write_bar(curr_step)
        self.info = self._write_info(curr_step)
        self._update_width(curr_step)
        self.last_step = curr_step
        return values
