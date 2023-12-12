# 安装
1. 准备 Eval 运行环境：

   ```bash
   conda create --name eval python=3.10 pytorch torchvision pytorch-cuda -c nvidia -c pytorch -y
   conda activate eval
   ```

   如果你希望自定义 PyTorch 版本或相关的 CUDA 版本，请参考 [官方文档](https://pytorch.org/get-started/locally/) 准备 PyTorch 环境。需要注意的是，Eval 要求 `pytorch>=1.13`。

2. 安装 Eval：

   ```bash
   git clone https://github.com/fanlinfuture/Fanlin-LLM-One.git
   cd Fanlin-LLM-One/Eval
   pip install requirements.txt
   apt-get update && apt-get install libgl1
   pip install -e .
   ```

3. 安装 humaneval（可选）：

   如果你需要**在 humaneval 数据集上评估模型代码能力**，请执行此步骤，否则忽略这一步。

   <details>
   <summary><b>点击查看详细</b></summary>

   ```bash
   git clone https://github.com/openai/human-eval.git
   cd human-eval
   pip install -r requirements.txt
   pip install -e .
   cd ..
   ```

   请仔细阅读 `human_eval/execution.py` **第48-57行**的注释，了解执行模型生成的代码可能存在的风险，如果接受这些风险，请取消**第58行**的注释，启用代码执行评测。

   </details>

4. 安装 Llama（可选）：

   如果你需要**使用官方实现评测 Llama / Llama-2 / Llama-2-chat 模型**，请执行此步骤，否则忽略这一步。

   <details>
   <summary><b>点击查看详细</b></summary>

   ```bash
   git clone https://github.com/facebookresearch/llama.git
   cd llama
   pip install -r requirements.txt
   pip install -e .
   cd ..
   ```

   你可以在 `configs/models` 下找到所有 Llama / Llama-2 / Llama-2-chat 模型的配置文件示例。
   
# 数据集准备
在 Eval 项目根目录下运行下面命令，将数据集准备至 `${Eval}/data` 目录下：

```bash
cp /ssd/tjwu/test_opencompass/test/data .
```
数据集我暂时放在204节点服务器上，对应路径如上。后续会上传git.
接下来，你可以阅读 `${Eval}/docs`评估使用手册了解 Eval 的基本用法。
