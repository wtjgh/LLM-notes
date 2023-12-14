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
