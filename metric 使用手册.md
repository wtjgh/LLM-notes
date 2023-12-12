# metric 使用手册
完成前面的安装后，可以使用eval中提供的metric计算函数来完成相应的指标计算。下面是一个使用示例：
 ```python
from eval.openicl.icl_evaluator import hf_metrics
metric = hf_metrics.JiebaRougeEvaluator
print(metric.compute("如果你需要使用官方实现评测","如果你需要使用官方实现评测"))
 ```
接下来，会介绍eval支持外部使用的几种metric计算函数使用场景及使用样例。

## Accuracy
Accuracy是在处理的总样本中正确预测的样本所占的比例。它可以用以下公式计算：Accuracy = (TP + TN) / (TP + TN + FP + FN) 其中：TP：真正例（True Positive） TN：真负例（True Negative） FP：假正例（False Positive） FN：假负例（False Negative）。使用示例如下：
 ```python
from eval.openicl.icl_evaluator import hf_metrics
metric = hf_metrics.Accuracy
print(metric.compute(references=[0, 1], predictions=[0, 1]))
{'accuracy': 1.0}
"-----------------------------------------------------------"
print(metric.compute(references=[0, 1, 2, 0, 1, 2], predictions=[0, 1, 1, 2, 1, 0]))
{'accuracy': 0.5}
 ```
