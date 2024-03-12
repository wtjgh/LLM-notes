# 安装
1. 安装 mmcv
mmpose对mmcv有依赖，所以在安装mmpose之前最好先安装mmcv。mmpose中对mmcv的版本限制有问题，最好使用项目中自带的mmpose进行安装。
我们先安装mmcv：
   ```bash
   pip install mmcv==2.1.0
   ```
2. 安装 mmpose：

   ```bash
   git clone git@github.com:fanlinfuture/video_human.git
   cd video_human/mmpose
   pip install -r requirements.txt
   pip install -v -e .
   # "-v" means verbose, or more output
   # "-e" means installing a project in editable mode,
   # thus any local modifications made to the code will take effect without reinstallation.
   ```
# 使用
程序的主入口在run.py,使用如下的命令即可运行程序拼接视频。
 ```bash
   
   python run.py --det_checkpoint checkpoint/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
   --pose_checkpoint checkpoint/td-hm_hrnet-w32_8xb64-210e_ubody-coco-256x192-7c227391_20230807.pth \
   --input  /ssd/tjwu/Person_Skeleton_Estimation/full_data/ \
   --output-root /ssd/tjwu/Person_Skeleton_Estimation/full_data/whole/output/ \
   --rife checkpoint --target_time 133
   ```
这些参数的意义为：
- `det_checkpoint`：检测模型权重路径。
- `pose_checkpoint`：姿态估计模型权重路径。
- `input`：待处理视频文件夹路径。
- `output-root`：检测到的姿态数据与处理后的视频的输出路径。
- `rife`：插帧模型权重路径。
- `target_time`：想生成视频的长度，以秒为单位。

当然，还有一个参数min_loop_time，表示剪切视频loop最少持续时间。默认为10秒。如非必要，不建议修改。
   
