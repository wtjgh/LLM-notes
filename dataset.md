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
如上所示，使用 ```python load() ```
