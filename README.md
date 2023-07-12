# PyTorch
pytorch notebook, 运行环境为 python3

**目录：**
 
----------------------------------------
- [01.简介](./pytorch/01.简介.md)
 
 
 
----------------------------------------
- [02.安装](./pytorch/02.安装.md)
  - 1. torch、torchvision、python 版本
  - 2. 安装
    - a. pip
    - b. conda
  - 3. 查看 GPU
 
 
 
----------------------------------------
- [03.语句](./pytorch/03.语句.md)
  - 1. 新建tensor
  - 2. tensor 转 numpy   (x.numpy(), torch.from_numpy(x))
  - 3. 尺寸变换（x.view() ）
  - 4. 交换维度位置 （x.permute()  ）
  - 5. 增加、减少维度
  - 6. 合并、分割
  - 7. 矩阵乘、点乘
    - a. 矩阵乘(.mm, .matmul)
    - b. 点乘(.mul)
  - 8. 随机种子 torch.manual_seed(args.seed)
 
 
 
----------------------------------------
- [04.数据集](./pytorch/04.数据集.md)
  - 1. 定于Dataset
  - 2. dataloader
 
 
 
----------------------------------------
- [05.模型加载保存](./pytorch/05.模型加载保存.md)
  - 1. 加载模型（torch.load() + model.load_state_dict()）
  - 2. 保存
 
 
 
----------------------------------------
- [06.网络](./pytorch/06.网络.md)
  - 1. 网络op
    - a. 全连接（nn.Linear()）
    - b. 卷积（nn.Conv2，nn.Conv3d）
    - c. op组合（nn.Sequential()）
  - 2. 定义网络
    - a. 定义 ResNet
    - b. 定义 capsNet
    - c. 定义 ResNet3D_VAE
 
 
 
----------------------------------------
- [07.训练](./pytorch/07.训练.md)
 
 
 
----------------------------------------
- [08.demos](./pytorch/08.demos.md)
  - 1. 3d_capsNet
 
 
 
----------------------------------------
- [09.tensorRT](./pytorch/09.tensorRT.md)
  - 1. 安装 tensorRT
    - a. 安装
      - i. 下载 
      - ii. 解压
      - iii. 添加环境变量
    - b. 安装TensorRT组件
      - i. python调用tensorrt
      - ii. 安装uff组件
      - iii. 安装graphsurgeon组件
    - c. 测试
      - i. 编译samples
      - ii. 验证
  - 2. 安装ONNX
    - a. 环境准备
    - b. 编译安装
    - c. 验证
  - 3. 使用
    - a. PyTorch模型转ONNX
    - b. ONNX转TensorRT
  - 4. 参考
 
 
