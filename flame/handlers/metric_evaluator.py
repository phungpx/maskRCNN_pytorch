from ..core.engine.torchvision_engine import Evaluator
# from ..core.engine.yolov3_engine import Evaluator
# from ..core.engine.efficientdet_engine import Evaluator
from ignite.engine import Events


class MetricEvaluator(Evaluator):
    def init(self):
        super(MetricEvaluator, self).init()
        self.frame['engine'].engine.add_event_handler(Events.EPOCH_COMPLETED, self._run)

    def _run(self, engine):
        self.run()
