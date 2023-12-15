# Dataset使用手册
Eval中已经预先继承了多个数据集的处理方式，用户可以直接调用这一部分代码，省去大量处理数据集的重复工作。Dataset可以分为外部使用与内部使用两种方法。内部使用方法已经在Eval中处理完毕，只需要按照要求配置路径即可。
这里主要介绍外部使用方法。
## 外部使用
下面使用siqa数据集做一个使用示例。
 ```python
from eval.datasets.siqa import siqaDataset
data = siqaDataset
dataset = data.load("./data/siqa")
print(dataset)
------------------------------------------------------------------------------------------
DatasetDict({
    train: Dataset({
        features: ['context', 'question', 'answerA', 'answerB', 'answerC', 'label'],
        num_rows: 33410
    })
    validation: Dataset({
        features: ['context', 'question', 'answerA', 'answerB', 'answerC', 'label'],
        num_rows: 1954
    })
})
 ```
如上所示，使用 ```load() ```函数传递数据集的路径，即可成功加载数据集。DatasetDict中为加载出来的数据集内容。可以看到，train，validation是siqa数据集的一种划分
方式。以train子集为例，features表示siqa数据集的内容。其中'context', 'question', 'answerA', 'answerB', 'answerC'为模型的输入，'label'为对应输入的标签。
num_rows表示数据集的样本个数。为了进一步明确说明数据集的模型输入与对应标签的关系，请看下面huatuo数据集的介绍：
 ```python
from eval.datasets.ht_gen import HTGenDataset
data = HTGenDataset
dataset = data.load("/ssd/tjwu/huatuo26m-lite/")
print(dataset)
---------------------------------------------------------
Dataset({
    features: ['content', 'abst'],
    num_rows: 1421
})
---------------------------------------------------------
print(dataset['content'][0])
孩子今年有大三阳因为他爸爸有一干儿大三阳，所以当时怀孕的时候抱着侥幸的心理认为孩子不会被遗传，我想知道小孩有大三阳需注意什么？
print(dataset['abst'][0])
孩子得到大三阳的可能性较高，需要定期检查肝功和乙肝病毒DNA，及时应用保肝的药物和抗病毒的药物。此外，营养摄入要均衡，保证足够的睡眠。
 ```
对于huatuo数据集，'content'表示送入大模型的问题，'abst'为对应输入的标签。
## 内部使用
内部使用时，只需要到 ```Fanlin-LLM-One/Eval/configs/datasets ```路径下，找到对应dataset的配置文件修改数据集路径即可，以siqa数据集为例。其数据集配置文件位于
```Fanlin-LLM-One/Eval/configs/datasets/siqa/siqa_ppl_e8d8c5.py ```。对应修改代码如下：
 ```python
siqa_datasets = [
    dict(
        abbr="siqa",
        type=siqaDataset,
        path='./data/siqa',  #将该路径替换为自己目录下的路径
        reader_cfg=siqa_reader_cfg,
        infer_cfg=siqa_infer_cfg,
        eval_cfg=siqa_eval_cfg)
]
 ```
## 新增数据集
尽管 Eval 已经包含了大多数常用数据集，用户在支持新数据集的时候需要完成以下几个步骤：

1. 在 `eval/datasets` 文件夹新增数据集脚本 `mydataset.py`, 该脚本需要包含：

   - 数据集及其加载方式，需要定义一个 `MyDataset` 类，实现数据集加载方法 `load`，该方法为静态方法，需要返回 `datasets.Dataset` 类型的数据。这里我们使用 huggingface dataset 作为数据集的统一接口，避免引入额外的逻辑。以huatuo数据集为例，具体示例如下：
  ```python
 import os.path as osp
import json
from datasets import Dataset
from eval.registry import LOAD_DATASET, TEXT_POSTPROCESSORS
from .base import BaseDataset

@LOAD_DATASET.register_module()
class HTGenDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        src_path = osp.join(path, 'dev.json')
        with open(src_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        result_dict = {'content': [], 'abst': []}

        for item in data:
            conversations = item.get('conversations', [])
            # result_dict = {'content': [], 'target': []}
            
            for conversation in conversations:
                if conversation['from'] == 'user':
                    result_dict['content'].append(conversation['value'])
                elif conversation['from'] == 'assistant':
                    result_dict['abst'].append(conversation['value'])

        dataset = Dataset.from_dict({
            'content': result_dict['content'],
            'abst': result_dict['abst']
        })
        return dataset
   ```
- （可选）如果 Eval已有的后处理方法不能满足需要，需要用户定义 `mydataset_postprocess` 方法，根据输入的字符串得到相应后处理的结果。具体示例如下：
```python
@TEXT_POSTPROCESSORS.register_module('htgen')
def htgen_postprocess(text: str) -> str:
    text = text.strip().split('\n')[0]
    text = text.replace('1. ', '') if text.startswith('1. ') else text
    text = text.replace('- ', '') if text.startswith('- ') else text
    text = text.strip('“，！”')
    return text
```
2. 在定义好数据集加载、评测以及数据后处理等方法之后，需要在配置文件中(Eval/configs/datasets/HT_generate/ht_gen.py)新增以下配置：

```python
from eval.openicl.icl_prompt_template import PromptTemplate
from eval.openicl.icl_retriever import ZeroRetriever
from eval.openicl.icl_inferencer import GenInferencer
from eval.openicl.icl_evaluator import JiebaRougeEvaluator
from eval.datasets import HTGenDataset,htgen_postprocess

htgen_reader_cfg = dict(input_columns=['content'], output_column='abst')

htgen_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(role='HUMAN', prompt='{content}\n：'),
        ])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

htgen_eval_cfg = dict(
    evaluator=dict(type=JiebaRougeEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=htgen_postprocess),
   
)

htgen_datasets = [
    dict(
        type=HTGenDataset,
        abbr='htgen',
        path='/ssd/tjwu/huatuo26m-lite/',
        reader_cfg=htgen_reader_cfg,
        infer_cfg=htgen_infer_cfg,
        eval_cfg=htgen_eval_cfg)
]

 ```
