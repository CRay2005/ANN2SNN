#!/usr/bin/env python3
"""
分析SNN在静态输入下的概率化ReLU行为
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from Models.layer import IF, ZIF

def analyze_snn_as_probabilistic_relu():
    """分析SNN IF层作为概率化ReLU的行为"""
    print("🎯 SNN作为概率化ReLU机制分析")
    print("="*80)
    
    # 创建IF层
    T = 8  # 增加时间步以获得更好的统计特性
    if_layer = IF(T=T, thresh=1.0)
    if_layer.eval()
    
    # 测试不同强度的输入
    input_values = torch.linspace(0, 3.0, 15)  # 0到3倍阈值
    batch_size = 100  # 增加batch size获得统计平均
    
    ann_outputs = []
    snn_firing_rates = []
    snn_std = []
    
    print(f"📊 测试参数: T={T}, thresh={if_layer.thresh.item():.2f}, batch_size={batch_size}")
    print("-" * 60)
    
    with torch.no_grad():
        for input_val in input_values:
            # 1. ANN模式 (T=0) - 标准ReLU
            if_layer.T = 0
            ann_input = torch.full((batch_size, 10), input_val.item())
            ann_output = if_layer(ann_input)
            ann_mean = ann_output.mean().item()
            ann_outputs.append(ann_mean)
            
            # 2. SNN模式 (T>0) - 脉冲发放
            if_layer.T = T
            # 创建静态输入（模拟add_dimention的效果）
            snn_input = ann_input.repeat(T, 1, 1)  # [T, batch, features]
            snn_input_flat = snn_input.view(T * batch_size, 10)  # [T*batch, features]
            
            snn_output = if_layer(snn_input_flat)  # [T*batch, features]
            snn_output_reshaped = snn_output.view(T, batch_size, 10)  # [T, batch, features]
            
            # 计算每个样本的发放率
            firing_rates = snn_output_reshaped.mean(dim=0)  # 对时间维度平均
            mean_firing_rate = firing_rates.mean().item()
            std_firing_rate = firing_rates.std().item()
            
            snn_firing_rates.append(mean_firing_rate)
            snn_std.append(std_firing_rate)
            
            print(f"输入={input_val:.2f}: ANN={ann_mean:.3f}, SNN发放率={mean_firing_rate:.3f}±{std_firing_rate:.3f}")
    
    # 分析结果
    print(f"\n🔍 关键发现:")
    print("-" * 40)
    
    # 计算相关性
    correlation = np.corrcoef(ann_outputs, snn_firing_rates)[0, 1]
    print(f"✅ ANN-SNN相关性: {correlation:.4f}")
    
    # 计算近似误差
    mse = np.mean((np.array(ann_outputs) - np.array(snn_firing_rates))**2)
    print(f"📏 均方误差: {mse:.6f}")
    
    # 分析线性关系
    slope = np.polyfit(ann_outputs, snn_firing_rates, 1)[0]
    print(f"📈 线性斜率: {slope:.4f} (理想情况应该≈1.0)")
    
    return input_values, ann_outputs, snn_firing_rates, snn_std

def analyze_temporal_dynamics():
    """分析时序动态：相同输入如何产生不同时间步输出"""
    print(f"\n🔄 时序动态分析")
    print("="*80)
    
    if_layer = IF(T=4, thresh=1.5)
    if_layer.eval()
    
    # 创建一个固定输入
    input_val = 2.0  # 高于阈值
    test_input = torch.full((1, 5), input_val)  # [1, 5]
    
    print(f"📥 测试输入: {input_val} (阈值={if_layer.thresh.item()})")
    
    with torch.no_grad():
        # 手动模拟IF层的时序处理
        if_layer.T = 4
        snn_input = test_input.repeat(4, 1, 1)  # [4, 1, 5]
        snn_input_flat = snn_input.view(4, 5)   # [4, 5]
        
        # 手动执行时序处理来观察每个时间步
        thresh = if_layer.thresh.data
        x = if_layer.expand(snn_input_flat)  # [4, 1, 5]
        
        mem = 0.5 * thresh  # 初始膜电位
        spike_pot = []
        
        print(f"\n⚡ 时序处理过程:")
        print("-" * 40)
        
        for t in range(4):
            print(f"时间步 {t}:")
            print(f"  输入: {x[t, 0, :3].numpy()} (都相同)")
            
            mem = mem + x[t, ...]
            print(f"  膜电位: {mem[0, :3].numpy()}")
            
            spike = if_layer.act(mem - thresh, if_layer.gama) * thresh
            print(f"  脉冲输出: {spike[0, :3].numpy()}")
            
            mem = mem - spike
            print(f"  重置后膜电位: {mem[0, :3].numpy()}")
            spike_pot.append(spike)
            print()
        
        # 分析时间步之间的差异
        all_spikes = torch.stack(spike_pot, dim=0)  # [4, 1, 5]
        
        print(f"📊 时间步差异分析:")
        print("-" * 30)
        for t in range(4):
            spike_sum = all_spikes[t, 0, :].sum().item()
            print(f"时间步{t}总脉冲: {spike_sum:.3f}")
        
        # 计算时间步之间的相似性
        similarities = []
        for i in range(3):
            sim = torch.cosine_similarity(
                all_spikes[i, 0, :], 
                all_spikes[i+1, 0, :], 
                dim=0
            ).item()
            similarities.append(sim)
            print(f"时间步{i}与{i+1}相似性: {sim:.4f}")

def demonstrate_rate_coding_principle():
    """演示脉冲发放率编码原理"""
    print(f"\n📡 脉冲发放率编码原理")
    print("="*80)
    
    print("🎯 核心原理:")
    print("  • 输入强度 → 膜电位累积速度")
    print("  • 膜电位累积 → 脉冲发放频率") 
    print("  • 发放频率 → 模拟连续值")
    print("  • 多时间步平均 → 近似ReLU输出")
    
    print(f"\n🧮 数学关系:")
    print("  ANN: y = max(0, x)")
    print("  SNN: y ≈ (Σ spikes) / T")
    print("  当T足够大时: SNN ≈ ANN")
    
    print(f"\n💡 关键洞察:")
    print("  ✅ SNN确实是概率化的ReLU近似")
    print("  ✅ 静态输入下缺乏真正的时序信息")
    print("  ✅ 时序分化来自神经元动态，非输入变化")
    print("  ✅ 这是ANN-SNN转换的巧妙策略")

def main():
    """主函数"""
    input_vals, ann_outs, snn_rates, snn_stds = analyze_snn_as_probabilistic_relu()
    analyze_temporal_dynamics()
    demonstrate_rate_coding_principle()
    
    print(f"\n" + "="*80)
    print("🎯 结论")
    print("="*80)
    print("❓ 问题: SNN是否更像概率处理机制？")
    print("✅ 答案: 是的！在静态输入场景下:")
    print("")
    print("🔸 SNN通过脉冲发放率编码模拟ReLU函数")
    print("🔸 多个时间步的平均效果近似连续激活值")
    print("🔸 膜电位动态创造内在的'随机性'")
    print("🔸 这是ANN→SNN转换的核心机制")
    print("")
    print("💭 深层思考:")
    print("  • 真正的SNN应该处理时序变化的输入")
    print("  • 当前方法是工程上的权衡：保持权重兼容性")
    print("  • 这种设计实现了'用时间换精度'的策略")

if __name__ == "__main__":
    main() 