# metric 使用手册
 ```python
from eval.openicl.icl_evaluator import hf_metrics
metric = hf_metrics.JiebaRougeEvaluator
print(metric.compute("如果你需要使用官方实现评测","如果你需要使用官方实现评测"))
 ```
