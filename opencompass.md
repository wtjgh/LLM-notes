# opencompass-notes
![图片](https://github.com/wtjgh/LLM-notes/assets/34306488/80588b4e-4e0d-4c4e-93f0-554ff02fc181)  
 
其中run.py文件是整个评估过程的启动文件，同样也是包含以下几个阶段：**配置** -> **推理** -> **评估** -> **可视化**。  

**配置**：这是整个工作流的起点。需要配置整个评估过程，选择要评估的模型和数据集。此外，还可以选择评估策略、计算后端等，并定义显示结果的方式。  

```python
from mmengine.config import read_base

with read_base():
    from .datasets.siqa.siqa_ppl import siqa_datasets
    from .datasets.winogrande.winogrande_ppl import winogrande_datasets
    from .models.opt.hf_opt_125m import opt125m
    # from .models.opt.hf_opt_350m import opt350m

datasets = [*siqa_datasets,*winogrande_datasets]
models = [opt125m]


from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import siqaDataset

siqa_reader_cfg = dict(
    input_columns=['context', 'question', 'answerA', 'answerB', 'answerC'],
    output_column='label',
    test_split='validation')

siqa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            1:
            dict(round=[
                dict(role='HUMAN', prompt="{context}\nQuestion: {question}\nAnswer:"),
                dict(role='BOT', prompt="{answerA}")
            ]),
            2:
            dict(round=[
                dict(role='HUMAN', prompt="{context}\nQuestion: {question}\nAnswer:"),
                dict(role='BOT', prompt="{answerB}")
            ]),
            3:
            dict(round=[
                dict(role='HUMAN', prompt="{context}\nQuestion: {question}\nAnswer:"),
                dict(role='BOT', prompt="{answerC}")
            ]),
        }),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer))

siqa_eval_cfg = dict(evaluator=dict(type=AccEvaluator))

siqa_datasets = [
    dict(
        abbr="siqa",
        type=siqaDataset,
        path='./data/siqa',
        reader_cfg=siqa_reader_cfg,
        infer_cfg=siqa_infer_cfg,
        eval_cfg=siqa_eval_cfg)
]

```
其中，reader_cfg是数据集加载配置，infer_cfg是推理配置。eval_cfg为评估配置，AccEvaluator为评估指标。  

**推理与评估**：在这个阶段，OpenCompass 将会开始对模型和数据集进行并行推理和评估。**推理**阶段主要是让模型从数据集产生输出，而**评估**阶段则是衡量这些输出与标准答案的匹配程度。这两个过程会被拆分为多个同时运行的“任务”以提高效率，但请注意，如果计算资源有限，这种策略可能会使评测变得更慢。而需要拆分为多个任务，这需要我们配置推理和评估阶段的执行策略。  
OpenCompass 支持自定义评测任务的任务划分器（`Partitioner`），实现评测任务的灵活切分；同时配合 `Runner` 控制任务执行的平台，如本机及集群。通过二者的组合，OpenCompass 可以将大评测任务分割到大量计算节点上运行，高效利用计算资源，从而大大加速评测流程。默认情况下，OpenCompass 向用户隐藏了这些细节，并自动选择推荐的执行策略。但是，用户仍然可以根据自己需求定制其策略，只需向配置文件中添加 `infer` 和/或 `eval` 字段即可：

```python
from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import SlurmRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=5000),
    runner=dict(
        type=SlurmRunner,
        max_num_workers=64,
        task=dict(type=OpenICLInferTask),
        retry=5),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=32,
        task=dict(type=OpenICLEvalTask)),
)
```
