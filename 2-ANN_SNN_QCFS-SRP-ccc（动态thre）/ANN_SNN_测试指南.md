# ANN和SNN准确率测试指南

## 概述

修改后的 `main_test.py` 现在支持分别测试ANN和SNN两种模式的准确率，帮助您对比两种训练方法的性能差异。

## 核心功能

### 🧠 ANN模式测试 (T=0)
- 使用量化激活函数
- 直接前向传播，无时间维度操作
- 适合传统神经网络推理

### ⚡ SNN模式测试 (T>0)  
- 使用脉冲发放机制
- 包含时间维度积分和平均
- 适合脉冲神经网络推理

## 使用方法

### 1. 直接使用main_test.py

```bash
# 基本用法
python main_test.py --dataset cifar10 --model vgg16 --identifier your_model_name --time 4

# 完整参数示例
python main_test.py \
    --dataset cifar100 \
    --model resnet20 \
    --identifier resnet20_L[4]_cray-grad \
    --time 8 \
    --device 0 \
    --batch_size 200
```

### 2. 使用交互式测试脚本

```bash
python test_ann_snn_example.py
```

然后按照提示选择预设配置或自定义配置。

## 参数说明

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `--dataset` | 数据集名称 | - | cifar10, cifar100, imagenet |
| `--model` | 模型架构 | - | vgg16, resnet18, resnet20, resnet34 |
| `--identifier` | 模型文件名(不含.pth) | - | vgg16_L[4]_cray-grad |
| `--time` | SNN时间步长 | 4 | 4, 8, 16 |
| `--device` | GPU设备号 | 0 | 0, 1, 2 |
| `--batch_size` | 测试批次大小 | 200 | 100, 200, 500 |

## 输出示例

```
🚀 开始测试模型: vgg16 on cifar10
📁 模型文件: vgg16_L[4]_cray-grad.pth
🔧 设备: cuda

============================================================
🧪 测试 ANN 模式 (T=0)
============================================================
📊 ANN 模式准确率: 85.32%

============================================================
🧪 测试 SNN 模式 (T=4)
============================================================
📊 SNN 模式准确率: 84.67%

============================================================
📈 准确率对比结果
============================================================
🧠 ANN模式 (T=0):     85.32%
⚡ SNN模式 (T=4):     84.67%
📊 准确率差异:        0.65%
⚠️  ANN模式表现更好，SNN损失 0.65%
```

## 测试流程

1. **模型加载**: 加载训练好的模型权重
2. **阈值设置**: 自动读取并设置IF层的动态阈值
3. **ANN测试**: 设置T=0，测试ANN模式准确率
4. **SNN测试**: 设置T>0，测试SNN模式准确率  
5. **结果对比**: 显示两种模式的性能差异

## 注意事项

- 确保模型文件存在于 `{dataset}-checkpoints/` 目录下
- 阈值文件应位于 `log/IF_{layer_id}_thresholds_stats.csv`
- SNN模式的时间步长会影响推理速度和准确率
- 建议使用相同的随机种子确保结果可重复

## 故障排除

### 模型文件不存在
```
FileNotFoundError: [Errno 2] No such file or directory: 'cifar10-checkpoints/model.pth'
```
**解决方案**: 检查模型文件路径和名称是否正确

### 阈值文件缺失
```
未找到阈值文件，使用默认阈值
```
**解决方案**: 确保已运行过阈值优化，或使用默认阈值继续测试

### GPU内存不足
```
CUDA out of memory
```
**解决方案**: 减小batch_size参数，如 `--batch_size 100`

## 扩展功能

如需添加更多测试功能，可以修改 `test_model_mode` 函数：

```python
def test_model_mode(model, test_loader, device, T, mode_name):
    # 添加更多测试指标
    # 如：推理时间、内存使用、能量消耗等
    pass
```
