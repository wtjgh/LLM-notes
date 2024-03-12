# 安装
1. 安装 mmcv
mmpose对mmcv有依赖，所以在安装mmpose之前最好先安装mmcv。mmpose中对mmcv的版本限制有问题，最好使用项目中自带的mmpose进行安装。
我们先安装mmcv：
   ```bash
   pip install mmcv==2.1.0
   ```
2. 安装 mmpose：

   ```bash
   cd mmpose
   pip install -r requirements.txt
   pip install -v -e .
   # "-v" means verbose, or more output
   # "-e" means installing a project in editable mode,
   # thus any local modifications made to the code will take effect without reinstallation.
   ```
