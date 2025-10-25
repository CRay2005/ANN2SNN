#!/usr/bin/env python3
"""
测试阈值训练后的ANN和SNN准确率
"""
import argparse
import os
import torch
import warnings
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from Models import modelpool
from Preprocess import datapool
from utils import train, val, seed_all, get_logger
from Models.layer_copy import *
import pandas as pd

# 设置环境变量抑制cuDNN警告
os.environ['CUDNN_V8_API_DISABLED'] = '1'
warnings.filterwarnings("ignore", category=UserWarning)
# 抑制PyTorch相关警告
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

def test_model_accuracy(model, test_loader, device, T_values=[0, 4, 8]):
    """
    测试模型在不同T值下的准确率
    
    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        device: 设备
        T_values: 要测试的T值列表，0表示ANN模式
    
    Returns:
        dict: 包含不同T值下准确率的字典
    """
    results = {}
    
    for T in T_values:
        print(f"\n=== 测试 T={T} (ANN模式)" if T == 0 else f"=== 测试 T={T} (SNN模式) ===")
        
        # 设置模型的T值
        model.set_T(T)
        
        # 测试准确率
        accuracy = val(model, test_loader, device, T, optimize_thre_flag=False)
        results[T] = accuracy
        
        print(f"T={T} 准确率: {accuracy:.4f}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='测试阈值训练后的ANN和SNN准确率')
    parser.add_argument('-data', '--dataset', default='cifar10', type=str, help='数据集')
    parser.add_argument('-arch', '--model', default='vgg16', type=str, help='模型')
    parser.add_argument('-id', '--identifier', type=str, default='vgg16_L[4]_thre[50,5,0.010]_thre-run', help='模型文件标识符')
    parser.add_argument('-dev', '--device', default='0', type=str, help='设备')
    parser.add_argument('-b', '--batch_size', default=200, type=int, help='批次大小')
    parser.add_argument('--seed', default=42, type=int, help='随机种子')
    
    args = parser.parse_args()
    
    # 设置设备
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 设置随机种子
    seed_all(args.seed)
    
    print(f"使用设备: {device}")
    print(f"数据集: {args.dataset}")
    print(f"模型: {args.model}")
    print(f"模型文件: {args.identifier}")
    
    # 准备数据
    train_loader, test_loader = datapool(args.dataset, args.batch_size)
    
    # 准备模型
    model = modelpool(args.model, args.dataset)
    
    # 加载模型权重
    model_dir = f'{args.dataset}-checkpoints'
    model_path = os.path.join(model_dir, args.identifier + '.pth')
    
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        return
    
    print(f"加载模型: {model_path}")
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    
    # 处理旧版本state_dict
    keys = list(state_dict.keys())
    for k in keys:
        if "relu.up" in k:
            state_dict[k[:-7]+'act.thresh'] = state_dict.pop(k)
        elif "up" in k:
            state_dict[k[:-2]+'thresh'] = state_dict.pop(k)
    
    # 加载模型权重
    try:
        load_model_compatible(model, state_dict)
        print("使用兼容性加载成功")
    except Exception as e:
        print(f"兼容性加载失败，尝试常规加载: {e}")
        model.load_state_dict(state_dict, strict=False)
        print("使用非严格模式加载成功")
    
    model.to(device)
    model.set_L(4)  # 设置量化级别
    
    # 读取并设置阈值文件（如果存在）
    print("\n=== 设置IF层阈值 ===")
    if_count = 0
    for name, module in model.named_modules():
        if isinstance(module, IF):
            # 构建阈值文件路径
            thre_file = f'/root/autodl-tmp/0-ANN2SNN-Allinone/2-ANN_SNN_QCFS-SRP-ccc（动态thre）/log/IF_{if_count}_thresholds_stats.csv'
            
            if os.path.exists(thre_file):
                # 读取CSV文件
                thre_df = pd.read_csv(thre_file)
                
                # 设置阈值 - 使用50分位值
                threshold_values = thre_df['50分位'].values if len(thre_df) > 1 else [thre_df['50分位'].mean()]
                
                # 创建tensor
                module.neuron_thre = torch.tensor(
                    threshold_values,
                    dtype=module.thresh.dtype,
                    device=module.thresh.device
                )
                
                print(f"第{if_count}层IF ({name}): 设置{len(threshold_values)}个通道的阈值")
            else:
                # 如果文件不存在，使用原有thresh
                module.neuron_thre = module.thresh.clone()
                print(f"第{if_count}层IF ({name}): 使用原有阈值")
            
            if_count += 1
    
    print(f"\n总共处理了 {if_count} 个IF层")
    
    # 测试不同T值下的准确率
    T_values = [0, 4, 8]  # 0=ANN, 4和8=SNN
    results = test_model_accuracy(model, test_loader, device, T_values)
    
    # 输出结果总结
    print("\n" + "="*50)
    print("准确率测试结果总结:")
    print("="*50)
    for T, acc in results.items():
        mode = "ANN" if T == 0 else f"SNN(T={T})"
        print(f"{mode:12}: {acc:.4f}")
    
    # 计算SNN相对于ANN的准确率损失
    if 0 in results:
        ann_acc = results[0]
        print(f"\nANN基准准确率: {ann_acc:.4f}")
        for T in T_values[1:]:  # 跳过T=0
            snn_acc = results[T]
            loss = ann_acc - snn_acc
            loss_percent = (loss / ann_acc) * 100
            print(f"SNN(T={T})准确率损失: {loss:.4f} ({loss_percent:.2f}%)")

if __name__ == "__main__":
    main()
