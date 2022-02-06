from toolbox.exp.OutputSchema import OutputSchema
from toolbox.utils.Store import StoreSchema
from toolbox.utils.Visualize import VisualizeSchema


class Experiment:

    def __init__(self, output: OutputSchema):
        self.output = output
        self.debug = output.logger.debug
        self.log = output.logger.info
        self.warn = output.logger.warn
        self.error = output.logger.error
        self.critical = output.logger.critical
        self.success = output.logger.success
        self.fail = output.logger.failed
        self.log_loss = output.child_log("loss.log").info
        self.log_config = output.child_log("config.log").info
        self.vis = VisualizeSchema(str(output.pathSchema.dir_path_visualize))
        self.store = StoreSchema(output.pathSchema)

    def re_init(self, output: OutputSchema):
        self.output = output
        self.debug = output.logger.debug
        self.log = output.logger.info
        self.warn = output.logger.warn
        self.error = output.logger.error
        self.critical = output.logger.critical
        self.success = output.logger.success
        self.fail = output.logger.failed
        self.log_loss = output.child_log("loss.log").info
        self.log_config = output.child_log("config.log").info
        self.vis = VisualizeSchema(str(output.pathSchema.dir_path_visualize))
        self.store = StoreSchema(output.pathSchema)

