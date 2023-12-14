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
---------------------------------------------------------
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
