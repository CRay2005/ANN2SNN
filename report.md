
## cifar10
          T=0       1         2         4         8         16        32        64   
QCFS      95.78     88.93     90.67     94.21     95.4      95.73     95.74     95.83     
NegSpike            88.93     90.55     94.54     95.56     95.78     95.77     95.82    
SRP(Tp=4)                               95.43     95.55     95.58     95.57     95.6
SRP+Neg                                 95.48     95.56     95.6      95.62     95.61

## cifar100
          T=0       1         2         4         8         16        32        64   
QCFS      77.19     58.65     64.95     71.24     75.14     77.01     77.18     77.42
NegSpike            58.65     65.31     72.0      76.04     77.23     77.37     77.17
SRP(Tp=4)                               76.27     76.45     76.59     76.61     76.55   
SRP+Neg                                 76.06     76.56     76.68     76.66     76.63

## cifar100 数据集
### T=4

(base) root@autodl-container-5145498c7c-decf52af:~/autodl-tmp/backup_cxqian/ANN_SNN_QCFS-main# python main_test.py -id=vgg16_L[4] -data=cifar100 -T=4 -dev=0
Files already downloaded and verified
Files already downloaded and verified
71.24
(base) root@autodl-container-5145498c7c-decf52af:~/autodl-tmp/backup_cxqian/ANN_SNN_QCFS-main# python main_test.py -id=vgg16_L[4] -data=cifar100 -T=4 -dev=0
Files already downloaded and verified
Files already downloaded and verified
72.0(negspike)

(base) root@autodl-container-5145498c7c-decf52af:~/autodl-tmp/backup_cxqian/ANN_SNN_QCFS-main# python main_test.py -id=vgg16_L[4] -data=cifar100 -T=4 -dev=0
Files already downloaded and verified
Files already downloaded and verified
76.27(SRP,预训练t=4，小于0的神经元置为dead)

### T=8
(base) root@autodl-container-5145498c7c-decf52af:~/autodl-tmp/backup_cxqian/ANN_SNN_QCFS-main# python main_test.py -id=vgg16_L[4] -data=cifar100 -T=8 -dev=0
Files already downloaded and verified
Files already downloaded and verified
75.14

(base) root@autodl-container-5145498c7c-decf52af:~/autodl-tmp/backup_cxqian/ANN_SNN_QCFS-main# python main_test.py -id=vgg16_L[4] -data=cifar100 -T=8 -dev=0
Files already downloaded and verified
Files already downloaded and verified
76.04(negspike)

(base) root@autodl-container-5145498c7c-decf52af:~/autodl-tmp/backup_cxqian/ANN_SNN_QCFS-main# python main_test.py -id=vgg16_L[4] -data=cifar100 -T=8 -dev=0
Files already downloaded and verified
Files already downloaded and verified
76.45(SRP,预训练t=4，小于0的神经元置为dead)

### T=64
(base) root@autodl-container-5145498c7c-decf52af:~/autodl-tmp/backup_cxqian/ANN_SNN_QCFS-main# python main_test.py -id=vgg16_L[4] -data=cifar100 -T=64 -dev=0
Files already downloaded and verified
Files already downloaded and verified
77.42
初始设mem=0, 76.16
(base) root@autodl-container-5145498c7c-decf52af:~/autodl-tmp/backup_cxqian/ANN_SNN_QCFS-main# python main_test.py -id=vgg16_L[4] -data=cifar100 -T=64 -dev=0
Files already downloaded and verified
Files already downloaded and verified
77.17(negspike)
额外测试：初始设mem=0, 77.07

(base) root@autodl-container-5145498c7c-decf52af:~/autodl-tmp/backup_cxqian/ANN_SNN_QCFS-main# python main_test.py -id=vgg16_L[4] -data=cifar100 -T=64 -dev=0
Files already downloaded and verified
Files already downloaded and verified
76.55(SRP,预训练t=4，小于0的神经元置为dead)


## cifar10 数据集

### T=4
(base) root@autodl-container-5145498c7c-decf52af:~/autodl-tmp/backup_cxqian/ANN_SNN_QCFS-main# python main_test.py -id=vgg16_wd[0.0005] -data=cifar10 -T=4 -dev=0
Files already downloaded and verified
Files already downloaded and verified
94.21

(base) root@autodl-container-5145498c7c-decf52af:~/autodl-tmp/backup_cxqian/ANN_SNN_QCFS-main# python main_test.py -id=vgg16_wd[0.0005] -data=cifar10 -T=4 -dev=0
Files already downloaded and verified
Files already downloaded and verified
94.54(negspike)

(base) root@autodl-container-5145498c7c-decf52af:~/autodl-tmp/backup_cxqian/ANN_SNN_QCFS-main# python main_test.py -id=vgg16_wd[0.0005] -data=cifar10 -T=4 -dev=0
Files already downloaded and verified
Files already downloaded and verified
95.43(SRP,预训练t=4，小于0的神经元置为dead)

(base) root@autodl-container-5145498c7c-decf52af:~/autodl-tmp/ANN_SNN_QCFS-main# python main_test.py -id=vgg16_wd[0.0005] -data=cifar10 -T=4 -Tp=4 -dev=0
Files already downloaded and verified
Files already downloaded and verified
acc:95.52
95.52(叠加negspike+SRP)

### T=8
(base) root@autodl-container-5145498c7c-decf52af:~/autodl-tmp/backup_cxqian/ANN_SNN_QCFS-main# python main_test.py -id=vgg16_wd[0.0005] -data=cifar10 -T=8 -dev=0
95.4

(base) root@autodl-container-5145498c7c-decf52af:~/autodl-tmp/backup_cxqian/ANN_SNN_QCFS-main# python main_test.py -id=vgg16_wd[0.0005] -data=cifar10 -T=8 -dev=0
Files already downloaded and verified
Files already downloaded and verified
95.56(negspike)

(base) root@autodl-container-5145498c7c-decf52af:~/autodl-tmp/backup_cxqian/ANN_SNN_QCFS-main# python main_test.py -id=vgg16_wd[0.0005] -data=cifar10 -T=4 -dev=0
Files already downloaded and verified
Files already downloaded and verified
95.55(SRP,预训练t=4，小于0的神经元置为dead)

(base) root@autodl-container-5145498c7c-decf52af:~/autodl-tmp/ANN_SNN_QCFS-main# python main_test.py -id=vgg16_wd[0.0005] -data=cifar10 -T=8 -Tp=4 -dev=0
Files already downloaded and verified
Files already downloaded and verified
acc:95.49(叠加negspike+SRP)

### T=64

(base) root@autodl-container-5145498c7c-decf52af:~/autodl-tmp/backup_cxqian/ANN_SNN_QCFS-main# python main_test.py -id=vgg16_wd[0.0005] -data=cifar10 -T=64 -dev=0
Files already downloaded and verified
Files already downloaded and verified
95.83

(base) root@autodl-container-5145498c7c-decf52af:~/autodl-tmp/backup_cxqian/ANN_SNN_QCFS-main# python main_test.py -id=vgg16_wd[0.0005] -data=cifar10 -T=64 -dev=0
Files already downloaded and verified
Files already downloaded and verified
95.82(negspike)

(base) root@autodl-container-5145498c7c-decf52af:~/autodl-tmp/backup_cxqian/ANN_SNN_QCFS-main# python main_test.py -id=vgg16_wd[0.0005] -data=cifar10 -T=64 -dev=0
Files already downloaded and verified
Files already downloaded and verified
95.6(SRP,预训练t=4，小于0的神经元置为dead)

(base) root@autodl-container-5145498c7c-decf52af:~/autodl-tmp/ANN_SNN_QCFS-main# python main_test.py -id=vgg16_wd[0.0005] -data=cifar10 -T=64 -Tp=4 -dev=0
Files already downloaded and verified
Files already downloaded and verified
acc:95.62 (叠加negspike+SRP)


## 分析结论

一 、对于网络初始，第一次输入到IF中，情形如下

对于IF，输入的向量是相同的T个数量的副本，也就是说输入的值相同，是正数或者副数，不会出现正数和负数交叉的情况。

相关代码在/root/autodl-tmp/backup_cxqian/ANN_SNN_QCFS-main 目录下。

### 设置hook，观察IF的输入和输出
(negspike)
为了便于观察，将batchsize设置为 1 ，即只读取一张图片。
对应的执行命令为：
```bash
python main_test.py -b=1 -id=vgg16_wd[0.0005] -data=cifar10 -T=8 -dev=0
```

将IF的输入和输出保存到hook_outputs文件夹中，从输出文件的结果来看：
输入：
每一个图像对应 64*32*32，即每一行 64个数，有1024行；
在之前add_dimension处理时，将每一张输入的照片复制了8份，因此这个文件每1024行代表一个图像的卷积输出，共重复8次，即8192行。

输出：

对应的输出结构也是一样，结构为 64*32*32*8 = 64* 8192行
输出的值为 0 或 0.214320 (对应的脉冲值)

举例分析如下：
/root/autodl-tmp/backup_cxqian/ANN_SNN_QCFS-main/hook_outputs/IF_20250219_102440(batchsize=1).txt

```python
    def forward(self, x):
        if self.T > 0:
            thre = self.thresh.data
            x = self.expand(x)
            mem = 0.5 * thre
            mem1 = 0.5 * thre
            mem2 = 0.5 * thre
            spike_pot = []
            for t in range(self.T):
                mem = mem + x[t, ...]
                spike = self.act(mem - thre, self.gama) * thre
                mem = mem - spike
                spike_pot.append(spike)
                #print(x[t,...])
            x = torch.stack(spike_pot, dim=0)
            x = self.merge(x)
```

初始mem=0.5 * thre =0.214320* 0.5

对于input的第3行（行号L4）（即其中一个像素点的64个卷积后的特征值）：

-0.016510 0.010329 0.007739 0.027301 0.033605 0.036969 0.050920 0.047421 0.037445 0.046250 0.056641 0.213208 0.119347 0.016816 -0.006565 -0.038390 -0.082402 -0.149130 -0.133653 -0.044573 -0.025064 -0.022914 -0.042698 -0.073775 -0.096169 -0.093124 -0.114993 -0.096996 -0.046595 -0.006788 -0.014577 -0.027380 -0.029363 -0.003262 -0.076516 -0.027291 0.029673 0.039507 0.045357 0.045018 0.039936 0.046312 0.052956 0.087609 0.000594 -0.028008 -0.079270 -0.156596 -0.145696 -0.150665 -0.103227 -0.010505 0.013564 0.002503 -0.021694 -0.052234 -0.040948 -0.066747 -0.131244 -0.156125 -0.096473 -0.011703 -0.008808 -0.022661


对于IF的for t in range(self.T)第一次循环中，
可以看到，这一行中只有 0.213208 0.119347 这两个值 加上mem后大于thre，释放脉冲

对应上述文件的行号L8197，输出两个脉冲
0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.214320 0.214320 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000

L8197 +1024 ......  ,依次递推，能看到T循环中，输出的所有脉冲情况。
以上输出的所有spike追加到spike_pot序列中，作为IF的输出。


二、随后的网络结构层中，在T循环内，IF的每次输入不再相同。
第一个IF在T循环内每次的输入向量相同，但脉冲值输出不相同，因此，通过卷积后，可能为正也可能为负，在输入到下一层IF时，即可能产生不均匀误差（Unevenness error）。

## 关于 thresh值
在IF的初始化函数中，设定了thresh=8.0，但在hook函数中打印这个值可以看到已经发生变化，这个值为  ** 0.214320 **

```python
    def save_hook(m, x, y):
        print(f"Input tensor:\n{x[0]}")     
        print(f"m.thresh:{m.thresh}")
        print(f"Outnput tensor:\n{y}")   
```

deepseek的解释如下：

在IF类的__init__函数中，thresh=8.0是默认参数值。self.thresh是通过nn.Parameter创建的，其初始值为torch.tensor([thresh])，因此self.thresh的值将是一个包含[8.0]的张量。
具体来说，self.thresh的值将是：
tensor([8.], requires_grad=True)
这个张量是可训练的（requires_grad=True），意味着在训练过程中它的值可能会被更新。



```python
class IF(nn.Module):
    def __init__(self, T=0, L=8, thresh=8.0, tau=1., gama=1.0):
        super(IF, self).__init__()
        self.act = ZIF.apply
        self.thresh = nn.Parameter(torch.tensor([thresh]), requires_grad=True)
        self.tau = tau
        self.gama = gama
        self.expand = ExpandTemporalDim(T)
        self.merge = MergeTemporalDim(T)
        self.L = L
        self.T = T
        self.loss = 0

```


```python
    #对于给定的输入size，查看模型的输入输出结构
    summary(model, input_size=[[3, 32, 32]])
```

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
  MergeTemporalDim-1            [-1, 3, 32, 32]               0
            Conv2d-2           [-1, 64, 32, 32]           1,792
       BatchNorm2d-3           [-1, 64, 32, 32]             128
 ExpandTemporalDim-4        [-1, 2, 64, 32, 32]               0
  MergeTemporalDim-5           [-1, 64, 32, 32]               0
                IF-6           [-1, 64, 32, 32]               0
           Dropout-7           [-1, 64, 32, 32]               0
            Conv2d-8           [-1, 64, 32, 32]          36,928
       BatchNorm2d-9           [-1, 64, 32, 32]             128
ExpandTemporalDim-10        [-1, 2, 64, 32, 32]               0
 MergeTemporalDim-11           [-1, 64, 32, 32]               0
               IF-12           [-1, 64, 32, 32]               0
          Dropout-13           [-1, 64, 32, 32]               0
        AvgPool2d-14           [-1, 64, 16, 16]               0
           Conv2d-15          [-1, 128, 16, 16]          73,856
      BatchNorm2d-16          [-1, 128, 16, 16]             256
ExpandTemporalDim-17       [-1, 2, 128, 16, 16]               0
 MergeTemporalDim-18          [-1, 128, 16, 16]               0
               IF-19          [-1, 128, 16, 16]               0
          Dropout-20          [-1, 128, 16, 16]               0
           Conv2d-21          [-1, 128, 16, 16]         147,584
      BatchNorm2d-22          [-1, 128, 16, 16]             256
ExpandTemporalDim-23       [-1, 2, 128, 16, 16]               0
 MergeTemporalDim-24          [-1, 128, 16, 16]               0
               IF-25          [-1, 128, 16, 16]               0
          Dropout-26          [-1, 128, 16, 16]               0
        AvgPool2d-27            [-1, 128, 8, 8]               0
           Conv2d-28            [-1, 256, 8, 8]         295,168
      BatchNorm2d-29            [-1, 256, 8, 8]             512
ExpandTemporalDim-30         [-1, 2, 256, 8, 8]               0
 MergeTemporalDim-31            [-1, 256, 8, 8]               0
               IF-32            [-1, 256, 8, 8]               0
          Dropout-33            [-1, 256, 8, 8]               0
           Conv2d-34            [-1, 256, 8, 8]         590,080
      BatchNorm2d-35            [-1, 256, 8, 8]             512
ExpandTemporalDim-36         [-1, 2, 256, 8, 8]               0
 MergeTemporalDim-37            [-1, 256, 8, 8]               0
               IF-38            [-1, 256, 8, 8]               0
          Dropout-39            [-1, 256, 8, 8]               0
           Conv2d-40            [-1, 256, 8, 8]         590,080
      BatchNorm2d-41            [-1, 256, 8, 8]             512
ExpandTemporalDim-42         [-1, 2, 256, 8, 8]               0
 MergeTemporalDim-43            [-1, 256, 8, 8]               0
               IF-44            [-1, 256, 8, 8]               0
          Dropout-45            [-1, 256, 8, 8]               0
        AvgPool2d-46            [-1, 256, 4, 4]               0
           Conv2d-47            [-1, 512, 4, 4]       1,180,160
      BatchNorm2d-48            [-1, 512, 4, 4]           1,024
ExpandTemporalDim-49         [-1, 2, 512, 4, 4]               0
 MergeTemporalDim-50            [-1, 512, 4, 4]               0
               IF-51            [-1, 512, 4, 4]               0
          Dropout-52            [-1, 512, 4, 4]               0
           Conv2d-53            [-1, 512, 4, 4]       2,359,808
      BatchNorm2d-54            [-1, 512, 4, 4]           1,024
ExpandTemporalDim-55         [-1, 2, 512, 4, 4]               0
 MergeTemporalDim-56            [-1, 512, 4, 4]               0
               IF-57            [-1, 512, 4, 4]               0
          Dropout-58            [-1, 512, 4, 4]               0
           Conv2d-59            [-1, 512, 4, 4]       2,359,808
      BatchNorm2d-60            [-1, 512, 4, 4]           1,024
ExpandTemporalDim-61         [-1, 2, 512, 4, 4]               0
 MergeTemporalDim-62            [-1, 512, 4, 4]               0
               IF-63            [-1, 512, 4, 4]               0
          Dropout-64            [-1, 512, 4, 4]               0
        AvgPool2d-65            [-1, 512, 2, 2]               0
           Conv2d-66            [-1, 512, 2, 2]       2,359,808
      BatchNorm2d-67            [-1, 512, 2, 2]           1,024
ExpandTemporalDim-68         [-1, 2, 512, 2, 2]               0
 MergeTemporalDim-69            [-1, 512, 2, 2]               0
               IF-70            [-1, 512, 2, 2]               0
          Dropout-71            [-1, 512, 2, 2]               0
           Conv2d-72            [-1, 512, 2, 2]       2,359,808
      BatchNorm2d-73            [-1, 512, 2, 2]           1,024
ExpandTemporalDim-74         [-1, 2, 512, 2, 2]               0
 MergeTemporalDim-75            [-1, 512, 2, 2]               0
               IF-76            [-1, 512, 2, 2]               0
          Dropout-77            [-1, 512, 2, 2]               0
           Conv2d-78            [-1, 512, 2, 2]       2,359,808
      BatchNorm2d-79            [-1, 512, 2, 2]           1,024
ExpandTemporalDim-80         [-1, 2, 512, 2, 2]               0
 MergeTemporalDim-81            [-1, 512, 2, 2]               0
               IF-82            [-1, 512, 2, 2]               0
          Dropout-83            [-1, 512, 2, 2]               0
        AvgPool2d-84            [-1, 512, 1, 1]               0
          Flatten-85                  [-1, 512]               0
           Linear-86                 [-1, 4096]       2,101,248
ExpandTemporalDim-87              [-1, 2, 4096]               0
 MergeTemporalDim-88                 [-1, 4096]               0
               IF-89                 [-1, 4096]               0
          Dropout-90                 [-1, 4096]               0
           Linear-91                 [-1, 4096]      16,781,312
ExpandTemporalDim-92              [-1, 2, 4096]               0
 MergeTemporalDim-93                 [-1, 4096]               0
               IF-94                 [-1, 4096]               0
          Dropout-95                 [-1, 4096]               0
           Linear-96                   [-1, 10]          40,970
ExpandTemporalDim-97                [-1, 2, 10]               0
================================================================



## 其它
1.在IF类的实现差异上，一个是定义了自己的IF类，一个是使用了spikingjelly.clock_driven.neuron.IFNode类。

2.在处理T时，第一个代码将同一个数据复制了T份，在定义网络时（如VGG）通过add-dimenison函数来实现，然后，在IF类中，通过expand函数扩展出T的维度，在类中循环T次，实现T个IF。而第二个代码，则是把T的实现放在val中，通过循环运行T次，实现T个IF。
两种方式实现效果是一样的，只需要把模电压、thre的初始化和复位处理好即可。


3.在floor函数处理上，略有差异。
第一个代码，

            x = x / self.thresh
            x = torch.clamp(x, 0, 1)
            x = myfloor(x*self.L+0.5)/self.L
            x = x * self.thresh

第二个代码，s
        x = x / self.up
        x = qcfs(x*self.t+0.5)/self.t
        x = torch.clamp(x, 0, 1)
        x = x * self.up






## ZIF的backward什么时候会被调用


当运行main_train进行训练，参数设置 T>0 时，进入SNN分支。

### PyTorch的自动求导机制。
ZIF类继承自torch.autograd.Function，所以当使用这个自定义Function时，PyTorch会自动在反向传播时调用其backward方法。
在 PyTorch 的自动求导系统中，`ZIF.backward()` 方法会在以下**两个条件同时满足时自动调用**：
注：实际运行测试时可看到，`ZIF.backward()` 有问题，运行时会报错

#### 触发条件
1. **参与计算图构建**  
   当使用 `ZIF.apply()` 的操作（即 `self.act` 的调用）位于需要梯度追踪的计算图中：
   ```python
   # 在 IF 类的 forward 中
   spike = self.act(mem - thre, self.gama) * thre  # 👈 关键调用点
   ```

2. **执行反向传播**  
   当调用 `loss.backward()` 时，PyTorch 会沿着计算图**自动回溯**，触发所有参与运算的 `backward()` 方法。

---

### 具体调用过程
```mermaid
sequenceDiagram
    participant Loss
    participant Backward Engine
    participant ZIF
    
    Loss->>Backward Engine: loss.backward()
    Backward Engine->>ZIF: 检测到ZIF操作在计算图中
    ZIF->>ZIF.backward(): 自动调用
    ZIF-->>Backward Engine: 返回梯度计算结果
```

# 关键特性说明
1. **隐式调用机制**  
   不需要手动调用，完全由 PyTorch 的 autograd 系统自动管理

2. **梯度依赖链**  
   只有当 `ZIF.forward()` 的输出被用于最终 loss 计算时，才会触发对应的 `backward()`

3. **上下文保存**  
   `ctx.save_for_backward()` 保存的数据会在 `backward()` 中被自动取出：
   ```python
   # forward 中保存
   ctx.save_for_backward(input, out, torch.tensor([gama]))
   
   # backward 中取出
   (input, out, gama) = ctx.saved_tensors
   ```


## self.thresh 的值是如何改变的
### 当T>0时，并没有实现相关的训练代码，因此，T>0(即对于SNN网络)无需讨论self.thresh 值的梯度变化。

### 当T=0时，self.thresh 的值是通过反向传播得到的，即self.thresh 的梯度。
首先，查看IF类的初始化部分，self.thresh被定义为nn.Parameter，并且requires_grad=True，这意味着它在训练过程中会计算梯度。
```python
class IF(nn.Module):
    def __init__(self, ...):
        self.thresh = nn.Parameter(torch.tensor([thresh]), requires_grad=True)  # 👈 关键定义
```

而当T=0时，在else分支中，输入x被除以self.thresh，然后进行量化操作，最后再乘以self.thresh。根据之前的讨论，self.thresh是一个nn.Parameter，其值的改变依赖于梯度计算和优化器的更新。只要self.thresh参与了计算图的构建，并且requires_grad为True，就会在反向传播时计算梯度，并由优化器更新。

在反向传播时，loss相对于self.thresh的梯度会被计算，优化器在step()时会更新self.thresh的值。

### 参数更新机制
1. **梯度产生点**：
   - `x / self.thresh` 产生对 `self.thresh` 的梯度 $\frac{\partial loss}{\partial thresh}$
   - `x * self.thresh` 也会产生梯度

2. **梯度计算示例**：
   假设原始输入为 `x`，经过以下变换：
   ```python
   y = (x / thresh) * thresh  # 数学上等价于 y = x
   ```
   但 PyTorch 会认为这是两个独立操作，计算梯度时会累积两个操作的梯度

3. **实际梯度公式**：
   $$
   \frac{\partial \mathcal{L}}{\partial thresh} = \sum \left( -\frac{x}{thresh^2} \cdot \text{下游梯度} \right) + \sum \left( x_{quantized} \cdot \text{下游梯度} \right)
   $$
   其中 `x_quantized` 是经过 clamp 和 floor 操作后的值

训练时会观察到：
```
[ANN 模式] thresh 前向值: 8.0000  # 初始值
           当前梯度: None

# 执行一次 backward 后
[ANN 模式] thresh 前向值: 7.9321  # 更新后的值 
           当前梯度: tensor([-0.0123])  # 梯度值
```

---

### 关键结论
只要：
1. `self.thresh` 参与了计算图（在 `x / self.thresh` 和 `x * self.thresh` 中）
2. 有反向传播过程（调用 `loss.backward()`）
3. 优化器包含该参数（如 `optimizer = Adam(model.parameters())`）

`self.thresh` 就会通过梯度下降自动更新，其更新方向由量化误差的梯度决定。

---

## 汇总梳理一下，当T>0和T=0时，self.thresh 的值分别是如何改变的


### 更新机制对比表
| 特征                | T>0 (SNN 模式)                              | T=0 (ANN 模式)                              |
|---------------------|--------------------------------------------|--------------------------------------------|
| **前向使用位置**     | `spike = act(mem - thre) * thre`           | `x = x / thresh` 和 `x = x * thresh`       |
| **梯度来源**         | 脉冲发放时的膜电位超阈值量                 | 量化缩放操作的数值变换                     |
| **梯度计算公式**     | $\frac{\partial \mathcal{L}}{\partial t} = \sum (-\text{脉冲梯度} \cdot gama^{-2} \cdot (gama - |\Delta V|))$ | $\frac{\partial \mathcal{L}}{\partial t} = \sum (-\frac{x}{t^2} \cdot \delta_{down}) + \sum (x_{quant} \cdot \delta_{down})$ |
| **参数影响**         | 控制神经元发放脉冲的阈值                   | 控制量化范围和输出幅值归一化               |
| **更新触发条件**     | 1. 参与计算图 <br> 2. 调用 loss.backward() | 同左                                       |
| **典型学习目标**     | 优化脉冲发放时机                           | 最小化量化误差                             |

---

### 共同更新原理
1. **自动梯度计算**  
   无论哪种模式，只要满足：
   - `self.thresh` 参与前向计算
   - 执行 `loss.backward()`
   - 优化器包含该参数（如 `Adam(model.parameters())`）

2. **优化器驱动更新**  
   参数实际更新发生在优化器 step 时：
   ```python
   optimizer.step()  # 执行: thresh = thresh - lr * thresh.grad
   ```

---

### 示例代码路径说明
```python
class IF(nn.Module):
    def forward(self, x):
        if self.T > 0:
            # SNN 模式梯度路径
            mem = mem + x[t] → spike = act(mem - thresh) → loss
        else:
            # ANN 模式梯度路径 
            x = (x / thresh) → quantize → (x * thresh) → loss
```



## 下一步方向
T>64，对于4090显卡报错超出内存

SNN的准确率在相关领域、相关数据集上已经很容易接近和超过ANN。

最近大模型的火爆、deepseek在计算资源方面的大幅度降低，让人们将注意力在关注进一步优化效果的同时，
也将更多的注意力转移到更节省资源的模型算法上。
"Inference-Scale Complexity in ANN-SNN Conversion for High-Performance and  Low-Power Applications"
"Nevertheless, the pursuit of larger models raises concerns about high energy consumption for model 
inference and training. The deployment of large models on resource-constrained devices has also become a challenge."





在cifar-10数据集上Un. ErrorII 统计的数据如下：
以Layer0， T=4为例，其它层统计数据分布趋势基本相似
αι=0,                                                                  0.3344
αι>0, 0.6656
Case 1: αι=0，ϕι (T ) >αι                                                      0.1593
Case 2: αι>0，ϕι (T ) >αι  (包含Case 4)            0.0862
Case 3: αι>0，ϕι (T ) <αι                                                      0.2118
hook_outputs 中大于0的浮点数: 4383151 (比例: 0.3344)
hook_outputs 中等于0的浮点数: 8724049 (比例: 0.6656)
hook_outputs 中等于0，且 hook_outputs-SNN 等于 hook_outputs 的浮点数: 6636575 (比例: 0.5063)
hook_outputs 中等于0，且 hook_outputs-SNN 大于 hook_outputs 的浮点数: 2087474 (比例: 0.1593)
hook_outputs 中大于0，且 hook_outputs-SNN 等于 hook_outputs 的浮点数: 477602 (比例: 0.0364)
hook_outputs 中大于0，且 hook_outputs-SNN 大于 hook_outputs 的浮点数: 1129790 (比例: 0.0862)
hook_outputs 中大于0，且 hook_outputs-SNN 小于 hook_outputs 的浮点数: 2775759 (比例: 0.2118)

Layer0:    T=8

浮点数总数: 13107200
hook_outputs 中大于0的浮点数: 4383151 (比例: 0.3344)
hook_outputs 中等于0的浮点数: 8724049 (比例: 0.6656)
hook_outputs 中等于0，且 hook_outputs-SNN 等于 hook_outputs 的浮点数: 5945379 (比例: 0.4536)
hook_outputs 中等于0，且 hook_outputs-SNN 大于 hook_outputs 的浮点数: 2778670 (比例: 0.2120)
hook_outputs 中大于0，且 hook_outputs-SNN 等于 hook_outputs 的浮点数: 521597 (比例: 0.0398)
hook_outputs 中大于0，且 hook_outputs-SNN 大于 hook_outputs 的浮点数: 1102790 (比例: 0.0841)
hook_outputs 中大于0，且 hook_outputs-SNN 小于 hook_outputs 的浮点数: 2758764 (比例: 0.2105)
Layer1:    T=4

浮点数总数: 13107200
hook_outputs 中大于0的浮点数: 3932910 (比例: 0.3001)
hook_outputs 中等于0的浮点数: 9174290 (比例: 0.6999)
hook_outputs 中等于0，且 hook_outputs-SNN 等于 hook_outputs 的浮点数: 5800633 (比例: 0.4426)
hook_outputs 中等于0，且 hook_outputs-SNN 大于 hook_outputs 的浮点数: 3373657 (比例: 0.2574)
hook_outputs 中大于0，且 hook_outputs-SNN 等于 hook_outputs 的浮点数: 5021 (比例: 0.0004)
hook_outputs 中大于0，且 hook_outputs-SNN 大于 hook_outputs 的浮点数: 905176 (比例: 0.0691)
hook_outputs 中大于0，且 hook_outputs-SNN 小于 hook_outputs 的浮点数: 3022713 (比例: 0.2306)
Layer2:    T=4

浮点数总数: 6553600
hook_outputs 中大于0的浮点数: 2072654 (比例: 0.3163)
hook_outputs 中等于0的浮点数: 4480946 (比例: 0.6837)
hook_outputs 中等于0，且 hook_outputs-SNN 等于 hook_outputs 的浮点数: 3024473 (比例: 0.4615)
hook_outputs 中等于0，且 hook_outputs-SNN 大于 hook_outputs 的浮点数: 1456473 (比例: 0.2222)
hook_outputs 中大于0，且 hook_outputs-SNN 等于 hook_outputs 的浮点数: 652 (比例: 0.0001)
hook_outputs 中大于0，且 hook_outputs-SNN 大于 hook_outputs 的浮点数: 393637 (比例: 0.0601)
hook_outputs 中大于0，且 hook_outputs-SNN 小于 hook_outputs 的浮点数: 1678365 (比例: 0.2561)

Layer3:    T=4

浮点数总数: 6553600
hook_outputs 中大于0的浮点数: 1390589 (比例: 0.2122)
hook_outputs 中等于0的浮点数: 5163011 (比例: 0.7878)
hook_outputs 中等于0，且 hook_outputs-SNN 等于 hook_outputs 的浮点数: 3886878 (比例: 0.5931)
hook_outputs 中等于0，且 hook_outputs-SNN 大于 hook_outputs 的浮点数: 1276133 (比例: 0.1947)
hook_outputs 中大于0，且 hook_outputs-SNN 等于 hook_outputs 的浮点数: 91513 (比例: 0.0140)
hook_outputs 中大于0，且 hook_outputs-SNN 大于 hook_outputs 的浮点数: 165136 (比例: 0.0252)
hook_outputs 中大于0，且 hook_outputs-SNN 小于 hook_outputs 的浮点数: 1133940 (比例: 0.1730)


