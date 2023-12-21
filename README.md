# models 定义与加载
目前，为了与huggingface的模型格式对其。我们采用的models 定义与加载共有两种模式。模式一：定义模型的同时加载模型。模式二：先定义模型，再加载模型。
## 模式一
```python
from modeling_qwen import QWenLMHeadModel
from transformers.generation import GenerationConfig
from transformers import AutoConfig,AutoTokenizer
path = "/ssd/ssli/Qwen/Qwen-1_8B-Chat"
#通过AutoConfig加载自定义的模型结构参数文件，只需要文件所在目录路径，但config的文件名必须命名为config.json        
config = AutoConfig.from_pretrained(path,trust_remote_code=True)
#通过QWenLMHeadModel类创建模型结构，使用from_pretrained函数加载模型的参数
model = QWenLMHeadModel.from_pretrained(pretrained_model_name_or_path=path,config=config,trust_remote_code=True,device_map="auto",)

#通过下面的例子验证模型加载到正确
tokenizer = AutoTokenizer.from_pretrained("/ssd/ssli/Qwen/Qwen-1_8B-Chat", trust_remote_code=True)
model.eval()
model.generation_config = GenerationConfig.from_pretrained("/ssd/tjwu/output_qwen/Qwen-1_8B-Chat/", trust_remote_code=True)
response, history = model.chat(tokenizer, "浙江的省会在哪里？", history=None)
print(response)
------------------------------------------------------------------------------------------------------------------------------------------------
before: 浙江的省会在哪里？
after: context_tokens: [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 101211, 9370, 65770, 105542, 101314, 11319, 151645, 198, 151644, 77091, 198] raw_text: <|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
浙江的省会在哪里？<|im_end|>
<|im_start|>assistant

input: tensor([[151644,   8948,    198,   2610,    525,    264,  10950,  17847,     13,
         151645,    198, 151644,    872,    198, 101211,   9370,  65770, 105542,
         101314,  11319, 151645,    198, 151644,  77091,    198]],
       device='cuda:1')
output: tensor([151644,   8948,    198,   2610,    525,    264,  10950,  17847,     13,
        151645,    198, 151644,    872,    198, 101211,   9370,  65770, 105542,
        101314,  11319, 151645,    198, 151644,  77091,    198, 105678,  36993,
         20412, 104130,   1773, 151645, 151643], device='cuda:1')
浙江省会是杭州。
```
使用模式一的方法如上面代码所示。需要注意的是：**使用这一方法需要import对应模型结构文件，例如代码中就import QWenLMHeadModel**。通过这一方法，蒸馏训练好的自定义模型，即使没有通过
auto注册，也可以成功加载使用蒸馏训练好的自定义模型。

## 模式二
```python
from modeling_qwen import QWenLMHeadModel
from transformers.generation import GenerationConfig
from transformers import AutoConfig,AutoTokenizer
import torch
path = "/ssd/ssli/Qwen/Qwen-1_8B-Chat"
#通过AutoConfig加载自定义的模型结构参数文件，只需要文件所在目录路径，但config的文件名必须命名为config.json          
config = AutoConfig.from_pretrained(path,trust_remote_code=True)
#通过QWenLMHeadModel类，根据config创建模型结构
model = QWenLMHeadModel(config)
#通过pytorch中的load_state_dict方法加载模型参数
model.load_state_dict(torch.load("/ssd/tjwu/output_qwen/Qwen-1_8B-Chat/test/pytorch_model.bin"))
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model.eval()
model.generation_config = GenerationConfig.from_pretrained(path, trust_remote_code=True)
response, history = model.chat(tokenizer, "浙江的省会在哪里？", history=None)
print(response)
```
使用模式二的方法如上面代码所示。这一方法更加灵活，适合训练使用。

## 模型保存
由于都采用huggingface 定义的模型格式。那么都会继承PretrainedModel这一个类。这个类自带一个save_pretrained函数，我们可以通过他来保存模型。
```python
def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        push_to_hub: bool = False,
        max_shard_size: Union[int, str] = "5GB",
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        token: Optional[Union[str, bool]] = None,
        save_peft_format: bool = True,
        **kwargs,
    ):
```
这里我们只需要关注三个参数。
