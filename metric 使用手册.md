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
## Rouge
ROUGE或者Recall-Oriented Understudy是一组用于评估自然语言处理中的摘要生成任务和机器翻译任务的度量指标。这一度量指标主要衡量的是：生成的摘要或翻译与参考的摘要或翻译之间的差异。
**请注意，ROUGE是不区分大小写的，即大写字母与小写字母被视为相同。并且，Rouge不能用来衡量中文数据集的差异。** 其中，"rouge1"：基于一个单词（1-gram）的评分。"rouge2"：基于两个单词（2-gram）的评分。使用示例如下：
"rougeL"：基于最长公共子序列的评分。"rougeLsum"：使用 "\n" 拆分文本。
 ```python
from eval.openicl.icl_evaluator import hf_metrics
metric = hf_metrics.Rouge
print(metric.compute(references=["hello there", "general kenobi"], predictions=["hello there", "general kenobi"]))
{'rouge1': 1.0, 'rouge2': 1.0, 'rougeL': 1.0, 'rougeLsum': 1.0}
"-----------------------------------------------------------"
print(metric.compute(references=[["hello", "there"], ["general kenobi", "general yoda"]], predictions=["hello there", "general kenobi"]))
{'rouge1': 0.8333, 'rouge2': 0.5, 'rougeL': 0.8333, 'rougeLsum': 0.8333}
 ```

## Rouge Chinese
由于Rouge不能用来衡量中文数据集的差异，所以采用Rouge Chinese来处理相应中文数据集。使用示例如下：
 ```python
from eval.openicl.icl_evaluator import hf_metrics
metric = hf_metrics.JiebaRougeEvaluator
print(metric.compute("如果你需要使用官方实现评测","如果你需要使用官方实现评测"))
{'rouge1': 99.99999949999997, 'rouge2': 0.0, 'rougeL': 99.99999949999997}
"-----------------------------------------------------------"
metric = hf_metrics.JiebaRougeEvaluator
print(metric.compute("如果你需要使用官方实现评测","如果你想要使用官方实现评测"))
{'rouge1': 92.30769184615382, 'rouge2': 0.0, 'rougeL': 92.30769184615382}
 ```
