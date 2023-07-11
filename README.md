# pytorch
pytorch notebook（笔记）
运行环境为 python3



- [pytorch](#pytorch)
  - [安装](#安装)
  - [常规](#常规)
      - [新建tensor](#新建tensor)
      - [tensor 转 numpy   (x.numpy(), torch.from\_numpy(x))](#tensor-转-numpy---xnumpy-torchfrom_numpyx)
      - [尺寸变换（x.view() ）](#尺寸变换xview-)
      - [交换维度位置 （x.permute()  ）](#交换维度位置-xpermute--)
      - [增加、减少维度](#增加减少维度)
      - [合并、分割](#合并分割)
      - [矩阵乘、点乘](#矩阵乘点乘)
        - [（1）矩阵乘(.mm, .matmul)](#1矩阵乘mm-matmul)
        - [（2）点乘(.mul)](#2点乘mul)
  - [网络op](#网络op)
      - [全连接（nn.Linear()）](#全连接nnlinear)
      - [卷积（nn.Conv2，nn.Conv3d）](#卷积nnconv2nnconv3d)
      - [op组合（nn.Sequential()）](#op组合nnsequential)
      - [加载模型（torch.load() + model.load\_state\_dict()）](#加载模型torchload--modelload_state_dict)
      - [随机种子 torch.manual\_seed(args.seed)](#随机种子-torchmanual_seedargsseed)
  - [16.tensorRT](#16tensorrt)



## 安装


torch  VS  torchvision  VS  python
https://github.com/pytorch/vision#installation




查看 torch 是否能调用GPU：
torch.cuda.is_available()  # 返回 True 表示可以



## 常规

#### 新建tensor
```python
a = torch.ones(5)
```

#### tensor 转 numpy   (x.numpy(), torch.from_numpy(x))
```python
a = torch.ones(5)
b = a.numpy()
c = torch.from_numpy(b)
```

#### 尺寸变换（x.view() ）

把原先tensor中的数据按照行优先的顺序排成一个一维的数据（这里应该是因为要求地址是连续存储的），然后按照参数组合成其他维度的tensor。
```python
a=torch.Tensor([[[1,2,3],[4,5,6]]])
print(a.view(3,2))
# 从1，2，3，4，5，6顺序的拿数组来填充需要的形状。
```

#### 交换维度位置 （x.permute()  ）
```python
a=np.array([[[1,2,3],[4,5,6]]])
unpermuted = torch.tensor(a)
print(unpermuted.size())    #  ——>  torch.Size([1, 2, 3])

permuted = unpermuted.permute(2,0,1)
print(permuted.size())      #  ——>  torch.Size([3, 1, 2])
```


#### 增加、减少维度
```python
b = a.unsqueeze(-1)  #增加一个维度 
b = a.squeeze(dim=1)  #减少一个维度 
```


#### 合并、分割
https://www.jianshu.com/p/4e57dbe1d281

方法 作用 区别
cat 合并 保持原有维度的数量
stack 合并 原有维度数量加1
split 分割 按照长度去分割
chunk 分割 等分

#### 矩阵乘、点乘
##### （1）矩阵乘(.mm, .matmul)
```python
A = torch.tensor([[1, 2, 3], [2, 3, 4]])   
B = torch.tensor([[1, 0, 1], [2, 1, -1]])
C = torch.tensor([[1, 0], [0, 1], [-1, 0]])


torch.mm(A, B)      # RuntimeError: size
torch.mm(A, C)        # tensor([[-2, 2], [-2, 3]])
torch.matmul(A, C)   # tensor([[-2, 2], [-2, 3]])
```

##### （2）点乘(.mul)
```python
torch.mul(A, B)     # tensor([[ 1, 0, 3], [ 4, 3, -4]])
torch.mul(A, C)      # RuntimeError: size 
```


## 网络op



#### 全连接（nn.Linear()）
```python
# in_features由输入张量的形状决定，out_features则决定了输出张量的形状 
fc = nn.Linear(in_features = 64*64*3, out_features = 1)

# 假定输入的图像形状为[64,64,3]
input = t.randn(1,64,64,3)

# 将四维张量转换为二维张量之后，才能作为全连接层的输入
input = input.view(1,64*64*3)
output = fc(input) # 调用全连接层
```

#### 卷积（nn.Conv2，nn.Conv3d）
```python
# ==== 二维
nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)

# ==== 三维
# Sample intput | 随机输入
net_input = torch.randn(32, 3, 10, 224, 224)

# 所有维度同一个参数配置
conv = nn.Conv3d(3, 64, kernel_size=3, stride=2, padding=1)
net_output = conv(net_input) # shape=[32, 64, 5, 112, 112] | 相当于每一个维度上的卷积核大小都是3，步长都是2，pad都是1

# 每一维度不同参数配置
conv = nn.Conv3d(3, 64, (2, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
net_output = conv(net_input) # shape=[32, 64, 9, 112, 112]

```




#### op组合（nn.Sequential()）
```python
self.layer1=nn.Sequential(nn.Linear(in_dim,n_hidden_1), nn.ReLU(True))
```

#### 加载模型（torch.load() + model.load_state_dict()）
```python
    # Prepare model
    config = CONFIGS[args.model_type]

    num_classes = 10 if args.dataset == "cifar10" else 1000

    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes)

    state_dict = torch.load(args.pretrained_dir)
    print(' #### state_dict: ', state_dict.keys)
    print(' #### type(state_dict): ', type(state_dict))
    state_dict_del = {}
    for name, weights in state_dict.items():
        if not ('head.weight' in name or 'head.bias' in name):
            state_dict_del[name] = weights
    print(' #### state_dict_del: ', state_dict_del.keys)

    model.load_state_dict(state_dict_del, False)
```




#### 随机种子 torch.manual_seed(args.seed)


















## 16.tensorRT

（1）安装

https://zhuanlan.zhihu.com/p/379287312

（2）使用
https://zhuanlan.zhihu.com/p/395590559







