import numpy as np

from toolbox.exp.OutputSchema import OutputSchema
from toolbox.utils.LaTeXSotre import EvaluateLaTeXStoreSchema
from toolbox.utils.MetricLogStore import MetricLogStoreSchema
from toolbox.utils.ModelParamStore import ModelParamStoreSchema
from toolbox.utils.Visualize import VisualizeSchema


class Experiment:

    def __init__(self, output: OutputSchema, local_rank: int = -1):
        self.output = output
        self.local_rank = local_rank
        self.debug = self.log_in_main_node(output.logger.debug)
        self.log = self.log_in_main_node(output.logger.info)
        self.warn = self.log_in_main_node(output.logger.warn)
        self.error = self.log_in_main_node(output.logger.error)
        self.critical = self.log_in_main_node(output.logger.critical)
        self.success = self.log_in_main_node(output.logger.success)
        self.fail = self.log_in_main_node(output.logger.failed)
        self.vis = VisualizeSchema(str(output.pathSchema.dir_path_visualize))
        self.model_param_store = ModelParamStoreSchema(output.pathSchema)
        self.metric_log_store = MetricLogStoreSchema(str(output.pathSchema.dir_path_log))
        self.latex_store = EvaluateLaTeXStoreSchema(output.pathSchema)

    def re_init(self, output: OutputSchema, local_rank: int = -1):
        self.output = output
        self.local_rank = local_rank
        self.debug = self.log_in_main_node(output.logger.debug)
        self.log = self.log_in_main_node(output.logger.info)
        self.warn = self.log_in_main_node(output.logger.warn)
        self.error = self.log_in_main_node(output.logger.error)
        self.critical = self.log_in_main_node(output.logger.critical)
        self.success = self.log_in_main_node(output.logger.success)
        self.fail = self.log_in_main_node(output.logger.failed)
        self.vis = VisualizeSchema(str(output.pathSchema.dir_path_visualize))
        self.model_param_store = ModelParamStoreSchema(output.pathSchema)
        self.metric_log_store = MetricLogStoreSchema(str(output.pathSchema.dir_path_log))
        self.latex_store = EvaluateLaTeXStoreSchema(output.pathSchema)

    def log_in_main_node(self, log_func):
        if self.local_rank == 0:
            return log_func
        return lambda x: [x]

    def dump_model(self, model):
        self.debug(model)
        self.debug("")
        self.debug("Trainable parameters:")
        num_params = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                ps = np.prod(param.size())
                num_params += ps
                self.debug(f"{name}: {sizeof_fmt(ps)}")
        self.log('Total Parameters: %s' % sizeof_fmt(num_params))
        self.debug("")


def sizeof_fmt(num, suffix='B'):
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)
