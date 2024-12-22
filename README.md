1. 前提条件：

    CA-SINet 仅在 Ubuntu 操作系统上进行了测试。它可能在其他操作系统（例如 Windows）上也能运行，但我们不保证这一点。

    + 在终端中创建虚拟环境：`conda create -n CA-SINet python=3.7`。

    + 安装必要的包：[PyTorch >= 1.13](https://pytorch.org/)、[opencv-python](https://pypi.org/project/opencv-python/)。

2. 准备数据：

    + 下载测试数据集并将其移动到 `./Dataset/TestDataset/`，可以在 [Google Drive](https://drive.google.com/file/d/1V0iSEdYJrT0Y_DHZfVGMg6TySFRNTy4o/view?usp=sharing) 找到。

    + 下载训练/验证数据集并将其移动到 `./Dataset/TrainValDataset/`，可以在 [Google Drive](https://drive.google.com/file/d/1M8-Ivd33KslvyehLK9_IUBGJ_Kf52bWG/view?usp=sharing) 找到。
    
    + 下载 PVTv2 权重，可以在 [Google Drive](https://drive.google.com/file/d/1JgCwftYFZIiL-r2I8vHwdOPRY9beEw6z/view) 找到。

3. 训练配置：

    + 在 `train.py` 中指定自定义路径，例如 `--train_save` 和 `--train_path`。

    + 只需在终端中运行 `python train.py` 即可开始训练。

4. 测试配置：

    + 下载测试数据集后，只需运行 `testing.py` 即可生成最终的预测图：替换你的训练模型目录（`--pth_path`）。
    
5. 验证模型：

   + 运行 `eval.py` 即可得到指标