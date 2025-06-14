#!/usr/bin/env python3
"""
测试不同剪枝比例对模型性能的影响
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Models.VGG import VGG
from Models.layer import IF
from Preprocess import datapool

def test_pruning_effect():
    """测试不同剪枝比例的效果"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 测试配置
    test_configs = [
        {"name": "无剪枝", "conv_ratio": 0.0, "fc_ratio": 0.0},
        {"name": "轻度剪枝", "conv_ratio": 0.1, "fc_ratio": 0.05},
        {"name": "中度剪枝", "conv_ratio": 0.2, "fc_ratio": 0.1},
        {"name": "重度剪枝", "conv_ratio": 0.3, "fc_ratio": 0.2},
    ]
    
    # 加载数据
    print("加载CIFAR10数据集...")
    train_loader, test_loader = datapool('CIFAR10', 200)
    
    # 创建模型并设置为SNN模式
    def create_model_with_pruning(conv_ratio, fc_ratio):
        """创建带有指定剪枝比例的模型"""
        # 修改VGG类来支持不同的剪枝比例
        class PrunedVGG(VGG):
            def _make_layers(self, cfg, dropout):
                layers = []
                for x in cfg:
                    if x == 'M':
                        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                    else:
                        layers.append(nn.Conv2d(self.init_channels, x, kernel_size=3, padding=1))
                        layers.append(nn.BatchNorm2d(x))
                        # 为卷积层设置剪枝比例
                        layers.append(IF(conv_pruning_ratio=conv_ratio, fc_pruning_ratio=fc_ratio))
                        layers.append(nn.Dropout(dropout))
                        self.init_channels = x
                return nn.Sequential(*layers)

            def __init__(self, vgg_name, num_classes, dropout, conv_ratio, fc_ratio):
                super().__init__(vgg_name, num_classes, dropout)
                # 重新构建分类器，为全连接层设置剪枝比例
                if vgg_name == 'VGG5':
                    self.classifier = nn.Sequential(
                        nn.Flatten(),
                        nn.Linear(512, 4096),
                        IF(conv_pruning_ratio=conv_ratio, fc_pruning_ratio=fc_ratio),
                        nn.Dropout(dropout),
                        nn.Linear(4096, 4096),
                        IF(conv_pruning_ratio=conv_ratio, fc_pruning_ratio=fc_ratio),
                        nn.Dropout(dropout),
                        nn.Linear(4096, num_classes)
                    )
                else:  # VGG16
                    self.classifier = nn.Sequential(
                        nn.Flatten(),
                        nn.Linear(512*7*7, 4096),
                        IF(conv_pruning_ratio=conv_ratio, fc_pruning_ratio=fc_ratio),
                        nn.Dropout(dropout),
                        nn.Linear(4096, 4096),
                        IF(conv_pruning_ratio=conv_ratio, fc_pruning_ratio=fc_ratio),
                        nn.Dropout(dropout),
                        nn.Linear(4096, num_classes)
                    )
        
        return PrunedVGG('VGG16', 10, 0.1, conv_ratio, fc_ratio)
    
    results = []
    
    for config in test_configs:
        print(f"\n{'='*50}")
        print(f"测试配置: {config['name']}")
        print(f"卷积层剪枝比例: {config['conv_ratio']}")
        print(f"全连接层剪枝比例: {config['fc_ratio']}")
        print(f"{'='*50}")
        
        # 创建模型
        model = create_model_with_pruning(config['conv_ratio'], config['fc_ratio'])
        model.to(device)
        
        # 加载预训练权重
        checkpoint_path = 'CIFAR10-checkpoints/VGG16_CIFAR10_0.pth'
        if os.path.exists(checkpoint_path):
            print("加载预训练权重...")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint, strict=False)
        else:
            print("警告: 未找到预训练权重，使用随机初始化")
        
        # 设置为SNN模式
        model.set_T(4)
        model.eval()
        
        # 测试性能
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i, (data, target) in enumerate(test_loader):
                if i >= 5:  # 只测试前5个batch以节省时间
                    break
                    
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                if i == 0:  # 第一个batch会触发剪枝初始化
                    print(f"已处理batch {i+1}/5")
        
        accuracy = 100 * correct / total
        results.append({
            'config': config['name'],
            'conv_ratio': config['conv_ratio'],
            'fc_ratio': config['fc_ratio'],
            'accuracy': accuracy
        })
        
        print(f"准确率: {accuracy:.2f}%")
    
    print(f"\n{'='*60}")
    print("测试结果汇总:")
    print(f"{'='*60}")
    for result in results:
        print(f"{result['config']:10} | Conv:{result['conv_ratio']:4.1f} | FC:{result['fc_ratio']:4.1f} | Acc:{result['accuracy']:6.2f}%")
    print(f"{'='*60}")

if __name__ == "__main__":
    test_pruning_effect() 