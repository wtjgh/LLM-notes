# 安装
1. 安装 RVM 推理所需环境
 
 由于我们只需要用RVM进行推理，所以我们只需要安装推理所需的依赖包：
   ```bash
   git clone
   cd RobustVideoMatting
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
