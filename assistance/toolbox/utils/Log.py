"""
@date: 2022/2/19
@description: 日志
"""
import logging
from typing import Optional

# logging.basicConfig(format='%(message)s', level=logging.INFO)
DEBUG_SUCCESS_NUM = 1001
DEBUG_FAILED_NUM = 1002
logging.addLevelName(DEBUG_SUCCESS_NUM, "SUCCESS")
logging.addLevelName(DEBUG_FAILED_NUM, "FAILED")


def debug_success(self, message, *args, **kws):
    if self.isEnabledFor(DEBUG_SUCCESS_NUM):
        self._log(DEBUG_SUCCESS_NUM, message, args, **kws)


def debug_failed(self, message, *args, **kws):
    if self.isEnabledFor(DEBUG_FAILED_NUM):
        self._log(DEBUG_FAILED_NUM, message, args, **kws)


logging.Logger.success = debug_success
logging.Logger.failed = debug_failed


blue = "\x1b[34m"
cyan = "\x1b[36;21m"
green = "\x1b[32;21m"
orange = "\x1b[33;21m"
grey = "\x1b[38;21m"
yellow = "\x1b[33;21m"
red = "\x1b[31;21m"
bold_red = "\x1b[31;1m"
reset = "\x1b[0m"


class ColorFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""
    time_prefix = "[%(asctime)s]"
    filename_prefix = " (%(filename)s:%(lineno)d)  "
    msg = "%(levelname)s - %(message)s"

    prefix = orange + time_prefix + reset + grey + filename_prefix + reset

    default_formatter = logging.Formatter(prefix + cyan + msg + reset)

    FORMATS = {
        logging.DEBUG: logging.Formatter(prefix + blue + msg + reset),
        logging.INFO: logging.Formatter(prefix + cyan + msg + reset),
        logging.WARNING: logging.Formatter(prefix + yellow + msg + reset),
        logging.ERROR: logging.Formatter(prefix + red + msg + reset),
        logging.CRITICAL: logging.Formatter(prefix + bold_red + msg + reset),
        DEBUG_SUCCESS_NUM: logging.Formatter(prefix + green + msg + reset),
        DEBUG_FAILED_NUM: logging.Formatter(prefix + bold_red + msg + reset),
    }

    def format(self, record):
        formatter = self.FORMATS.get(record.levelno, self.default_formatter)
        return formatter.format(record)


def Log(filename: Optional[str]=None, name_scope="0", write_to_console=True):
    """Return instance of logger 统一的日志样式

        Examples:
           >>> from toolbox.utils.Log import Log
           >>> log = Log("./train.log")
           >>> log.debug("debug message")
           >>> log.info("info message")
           >>> log.warning("warning message")
           >>> log.error("error message")
           >>> log.critical("critical message")
    """
    logger = logging.getLogger('log-%s' % name_scope)
    logger.setLevel(logging.DEBUG)

    if filename is not None:
        file_handler = logging.FileHandler(filename)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('[%(asctime)s] p%(process)s (%(filename)s:%(lineno)d) - %(message)s', '%m-%d %H:%M:%S'))
        logger.addHandler(file_handler)

    if write_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(ColorFormatter())
        logger.addHandler(console_handler)

    return logger


def log_result(logger, result):
    """
    :param logger: from toolbox.utils.Log()
    :param result: from toolbox.Evaluate.evaluate()
    """
    from toolbox.evaluate.Evaluate import pretty_print
    pretty_print(result, logger.info)

if __name__ == '__main__':
    log = Log()
    log.info("test")
    log.warn("test")
    log.error("test")
    log.debug("test")
    log.success("test")
    log.failed("test")
    log.critical("test")
    log.debug_success("test")
    log.debug_failed("test")
