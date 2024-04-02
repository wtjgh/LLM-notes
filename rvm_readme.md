# 安装
1. 安装 RVM 推理所需环境
 
 由于我们只需要用RVM进行推理，所以我们只需要安装推理所需的依赖包：
   ```bash
   git clone http://10.128.218.101:1234/wutianjian/robustvideomatting.git
   cd robustvideomatting
   pip install -r requirements_inference.txt
   ```

# 使用
程序的主入口在run.py,使用如下的命令即可运行程序对视频进行抠图。
 ```bash
   
   python run.py --variant mobilenetv3 --checkpoint /ssd/tjwu/Person_Skeleton_Estimation/rvm/rvm_mobilenetv3.pth \
   --device cpu --input-source  /ssd/tjwu/Person_Skeleton_Estimation/full_data/test/joe_wong_host_v1.mp4 \
   --output-composition rvm.mp4 --output-video-mbps 2 
   ```
这些参数的意义为：
- `variant`：模型结构有mobilenetv3和resnet50两种。
- `checkpoint`：模型权重路径。
- `device`：模型加载设备可选（cpu 或 cuda）。
- `input-source`：待处理视频路径。
- `output-composition`：若导出视频，提供文件路径。。
- `output-video-mbps`：若导出视频，提供视频码率。


或者使用提供的视频转换 API：
   ```python
import torch
from model import MattingNetwork
from run import convert_video
model = MattingNetwork(variant='mobilenetv3').eval().cuda() # 或 variant="resnet50"
model.load_state_dict(torch.load('rvm_mobilenetv3.pth'))

convert_video(
    model,                           # 模型，可以加载到任何设备（cpu 或 cuda）
    input_source='input.mp4',        # 视频文件，或图片序列文件夹
    input_resize=(1920, 1080),       # [可选项] 缩放视频大小
    downsample_ratio=0.25,           # [可选项] 下采样比，若 None，自动下采样至 512px
    output_type='video',             # 可选 "video"（视频）或 "png_sequence"（PNG 序列）
    output_composition='com.mp4',    # 若导出视频，提供文件路径。若导出 PNG 序列，提供文件夹路径
    output_video_mbps=4,             # 若导出视频，提供视频码率
    seq_chunk=1,                     # 设置多帧并行计算
    num_workers=1,                   # 只适用于图片序列输入，读取线程
    progress=True                    # 显示进度条
)
   ```
