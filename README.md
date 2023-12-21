# models 定义与加载
目前，为了与huggingface的模型格式对其。我们采用的models 定义与加载共有两种模式。模式一：定义模型的同时加载模型。模式二：先定义模型，再加载模型。
## 模式一
```python
from modeling_qwen import QWenLMHeadModel
from transformers.generation import GenerationConfig
from transformers import AutoConfig,AutoTokenizer
path = "/ssd/ssli/Qwen/Qwen-1_8B-Chat"        
config = AutoConfig.from_pretrained(path,trust_remote_code=True)
model = QWenLMHeadModel.from_pretrained(pretrained_model_name_or_path=path,config=config,trust_remote_code=True,device_map="auto",)
tokenizer = AutoTokenizer.from_pretrained("/ssd/ssli/Qwen/Qwen-1_8B-Chat", trust_remote_code=True)
model.eval()
model.generation_config = GenerationConfig.from_pretrained("/ssd/tjwu/output_qwen/Qwen-1_8B-Chat/", trust_remote_code=True)
response, history = model.chat(tokenizer, "浙江的省会在哪里？", history=None)
print(response)
```
