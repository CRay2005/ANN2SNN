#!/usr/bin/env python3
"""
ANN和SNN准确率测试示例脚本

使用方法:
1. 测试CIFAR-10数据集上的VGG16模型:
   python main_test.py --dataset cifar10 --model vgg16 --identifier vgg16_L[4]_cray-grad --time 4

2. 测试CIFAR-100数据集上的ResNet20模型:
   python main_test.py --dataset cifar100 --model resnet20 --identifier resnet20_L[4]_cray-grad --time 8

3. 测试ImageNet数据集上的ResNet18模型:
   python main_test.py --dataset imagenet --model resnet18 --identifier resnet18_L[4]_cray-grad --time 4

参数说明:
- --dataset: 数据集名称 (cifar10, cifar100, imagenet)
- --model: 模型架构 (vgg16, resnet18, resnet20, resnet34)
- --identifier: 训练好的模型文件名（不包含.pth扩展名）
- --time: SNN模式的时间步长 (T参数)
- --device: GPU设备号 (默认: 0)
- --batch_size: 测试批次大小 (默认: 200)
"""

import subprocess
import sys
import os
import torch
import argparse
from Models import modelpool
from Preprocess import datapool
from utils import val, seed_all
from Models.layer import *
# from Models.layer import load_model_compatible  # 该函数在当前版本中不存在
import pandas as pd

def test_model_mode(model, test_loader, device, T, mode_name):
    """测试模型在指定模式下的准确率"""
    print(f"\n{'='*60}")
    print(f"🧪 测试 {mode_name} 模式 (T={T})")
    print(f"{'='*60}")
    
    # 设置模型模式
    model.set_T(T)
    
    # 测试准确率
    acc = val(model, test_loader, device, T, optimize_thre_flag=False)
    
    print(f"📊 {mode_name} 模式准确率: {acc:.2f}%")
    return acc

def run_test(dataset, model, identifier, time_steps=4, device='0', batch_size=200):
    """运行ANN和SNN准确率测试"""
    
    print(f"🚀 开始测试模型: {model} on {dataset}")
    print(f"📁 模型文件: {identifier}.pth")
    print(f"🔧 设备: GPU {device}")
    print(f"📊 测试配置:")
    print(f"   - 数据集: {dataset}")
    print(f"   - 模型: {model}")
    print(f"   - 模型文件: {identifier}.pth")
    print(f"   - SNN时间步: {time_steps}")
    print(f"   - 设备: GPU {device}")
    print(f"   - 批次大小: {batch_size}")
    print()
    
    try:
        # 设置环境变量
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 设置随机种子
        seed_all(42)
        
        # 准备数据
        train_loader, test_loader = datapool(dataset, batch_size)
        
        # 准备模型
        model = modelpool(model, dataset)
        
        # 加载模型权重
        model_dir = f'{dataset}-checkpoints'
        state_dict = torch.load(os.path.join(model_dir, identifier + '.pth'), map_location=torch.device('cpu'))
        
        # 处理旧版本state_dict
        keys = list(state_dict.keys())
        for k in keys:
            if "relu.up" in k:
                state_dict[k[:-7]+'act.thresh'] = state_dict.pop(k)
            elif "up" in k:
                state_dict[k[:-2]+'thresh'] = state_dict.pop(k)
        
        # 加载模型权重
        try:
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"模型加载失败: {e}")
            return False
        
        model.to(device_obj)
        model.set_L(8)
        
        # 读取阈值文件并设置给IF层
        if_count = 0
        for name, module in model.named_modules():
            if isinstance(module, IF):
                # 构建阈值文件路径
                thre_file = f'/root/autodl-tmp/0-ANN2SNN-Allinone/2-ANN_SNN_QCFS-SRP-ccc（动态thre）/log/IF_{if_count}_thresholds_stats.csv'
                
                if os.path.exists(thre_file):
                    # 读取CSV文件
                    thre_df = pd.read_csv(thre_file)
                    
                    # 设置阈值 - 根据通道数自动判断
                    threshold_values = thre_df['均值'].values if len(thre_df) > 1 else [thre_df['50分位'].mean()]
                    
                    # 统一创建tensor
                    module.neuron_thre = torch.tensor(
                        threshold_values,
                        dtype=module.thresh.dtype,
                        device=module.thresh.device
                    )
                else:
                    # 如果文件不存在，使用原有thresh
                    module.neuron_thre = module.thresh.clone()
                
                if_count += 1
        
        # 分别测试ANN和SNN模式
        ann_acc = test_model_mode(model, test_loader, device_obj, T=0, mode_name="ANN")
        snn_acc = test_model_mode(model, test_loader, device_obj, T=time_steps, mode_name="SNN")
        
        # 对比结果
        print(f"\n{'='*60}")
        print(f"📈 准确率对比结果")
        print(f"{'='*60}")
        print(f"🧠 ANN模式 (T=0):     {ann_acc:.2f}%")
        print(f"⚡ SNN模式 (T={time_steps}):   {snn_acc:.2f}%")
        print(f"📊 准确率差异:        {abs(ann_acc - snn_acc):.2f}%")
        
        if snn_acc > ann_acc:
            print(f"✅ SNN模式表现更好，提升 {snn_acc - ann_acc:.2f}%")
        elif ann_acc > snn_acc:
            print(f"⚠️  ANN模式表现更好，SNN损失 {ann_acc - snn_acc:.2f}%")
        else:
            print(f"🔄 两种模式表现相同")
        
        # 打印所有IF层的thresh参数值
        print(f"\n{'='*60}")
        print(f"🔍 IF层参数信息")
        print(f"{'='*60}")
        if_count = 0
        for name, module in model.named_modules():
            if isinstance(module, IF):
                print(f"第{if_count}层IF ({name}):")
                print(f"  - thresh参数值: {module.thresh.item():.6f}")
                print(f"  - neuron_thre形状: {module.neuron_thre.shape}")
                print(f"  - neuron_thre值: {module.neuron_thre.flatten()[:5].tolist()}...")  # 只显示前5个值
                if_count += 1
        
        if if_count == 0:
            print("未找到任何IF层")
        else:
            print(f"\n总共找到 {if_count} 个IF层")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数 - 自动测试所有可用配置"""
    
    print("🧪 ANN和SNN准确率测试工具")
    print("="*50)
    
    # 预设的测试配置 - 专门测试thre训练模型
    test_configs = {
        '1': {
            'name': 'VGG16 thre[20,5,0.010] cray-grad',
            'dataset': 'cifar10',
            'model': 'vgg16',
            'identifier': 'vgg16_L[4]_thre[20,5,0.010]_cray-grad',
            'time': 4
        },
        '2': {
            'name': 'VGG16 thre[100,5,0.010] cray-grad',
            'dataset': 'cifar10',
            'model': 'vgg16',
            'identifier': 'vgg16_L[4]_thre[100,5,0.010]_cray-grad',
            'time': 4
        },
        '3': {
            'name': 'VGG16 thre[20,5,0.010] thre-run',
            'dataset': 'cifar10',
            'model': 'vgg16',
            'identifier': 'vgg16_L[4]_thre[20,5,0.010]_thre-run',
            'time': 4
        },
        '4': {
            'name': 'VGG16 thre[50,5,0.010] thre-run',
            'dataset': 'cifar10',
            'model': 'vgg16',
            'identifier': 'vgg16_L[4]_thre[50,5,0.010]_thre-run',
            'time': 4
        },
        '5': {
            'name': 'VGG16 thre[100,5,0.010] thre-run',
            'dataset': 'cifar10',
            'model': 'vgg16',
            'identifier': 'vgg16_L[4]_thre[100,5,0.010]_thre-run',
            'time': 4
        }
    }
    
    print("📋 开始自动测试所有可用配置...")
    
    # 自动测试所有配置
    for key, config in test_configs.items():
        print(f"\n{'='*80}")
        print(f"🎯 测试配置 {key}: {config['name']}")
        print(f"{'='*80}")
        
        # 检查模型文件是否存在
        model_dir = f"{config['dataset']}-checkpoints"
        model_file = os.path.join(model_dir, f"{config['identifier']}.pth")
        
        if not os.path.exists(model_file):
            print(f"⚠️  跳过: 模型文件不存在: {model_file}")
            continue
        
        # 运行测试
        success = run_test(
            dataset=config['dataset'],
            model=config['model'],
            identifier=config['identifier'],
            time_steps=config['time']
        )
        
        if success:
            print(f"\n✅ 配置 {key} 测试成功完成!")
        else:
            print(f"\n❌ 配置 {key} 测试失败!")
    
    print(f"\n{'='*80}")
    print("🏁 所有测试完成!")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
