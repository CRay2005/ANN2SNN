#!/usr/bin/env python3
"""
详细解释ExpandTemporalDim和MergeTemporalDim的作用和原理
"""

import torch
import torch.nn as nn
from Models.layer import ExpandTemporalDim, MergeTemporalDim, IF

def explain_temporal_operations():
    """解释时间维度操作的核心原理"""
    print("🔄 时间维度处理步骤详解")
    print("="*80)
    
    # 1. 理解问题背景
    print("📚 背景知识:")
    print("-" * 50)
    print("🎯 ANN vs SNN 的核心区别:")
    print("   • ANN: 每层一次前向传播，输出即时值")
    print("   • SNN: 每层需要多个时间步，模拟神经元的时序动态")
    print("   • 问题: 如何在同一个网络中支持两种模式？")
    print("   • 解决: 通过时间维度的动态展开和合并")
    
    # 2. 演示具体操作
    print("\n🧪 时间维度操作演示:")
    print("-" * 50)
    
    # 创建示例数据 (batch_size=2, features=4)
    batch_size = 2
    features = 4
    T = 3  # 时间步数
    
    print(f"设置参数: batch_size={batch_size}, features={features}, T={T}")
    
    # ANN模式的输入数据
    ann_input = torch.randn(batch_size, features)
    print(f"\n📥 ANN模式输入: {ann_input.shape}")
    print(f"数据内容:\n{ann_input}")
    
    # 3. 演示ExpandTemporalDim
    print(f"\n🔀 ExpandTemporalDim操作:")
    print("-" * 30)
    
    expand_op = ExpandTemporalDim(T)
    
    # 首先需要为SNN准备输入（复制T次）
    snn_input = ann_input.repeat(T, 1, 1)  # [T, batch, features]
    snn_input_flat = snn_input.view(T * batch_size, features)  # [T*batch, features]
    
    print(f"SNN预处理输入: {snn_input_flat.shape}")
    print(f"数据内容:\n{snn_input_flat}")
    
    # 应用ExpandTemporalDim
    expanded = expand_op(snn_input_flat)
    print(f"\nExpandTemporalDim输出: {expanded.shape}")
    print(f"数据内容:\n{expanded}")
    
    print(f"\n🔍 ExpandTemporalDim详解:")
    print(f"   • 输入: [T*batch, features] = [{T * batch_size}, {features}]")
    print(f"   • 输出: [T, batch, features] = [{T}, {batch_size}, {features}]")
    print(f"   • 作用: 将扁平化的时序数据重新组织成时间步结构")
    print(f"   • 核心: 为SNN的时序处理做准备")
    
    # 4. 演示MergeTemporalDim
    print(f"\n🔀 MergeTemporalDim操作:")
    print("-" * 30)
    
    merge_op = MergeTemporalDim(T)
    
    # 模拟IF层处理后的输出（仍然是时间步格式）
    if_output = torch.randn(T, batch_size, features)
    print(f"IF层时序输出: {if_output.shape}")
    print(f"数据内容:\n{if_output}")
    
    # 应用MergeTemporalDim
    merged = merge_op(if_output)
    print(f"\nMergeTemporalDim输出: {merged.shape}")
    print(f"数据内容:\n{merged}")
    
    print(f"\n🔍 MergeTemporalDim详解:")
    print(f"   • 输入: [T, batch, features] = [{T}, {batch_size}, {features}]")
    print(f"   • 输出: [T*batch, features] = [{T * batch_size}, {features}]")
    print(f"   • 作用: 将时序数据扁平化，便于后续层处理")
    print(f"   • 核心: 将SNN的时序输出转换回ANN兼容格式")


def demonstrate_if_layer_workflow():
    """演示IF层中时间维度操作的完整工作流程"""
    print("\n" + "="*80)
    print("🔄 IF层中时间维度操作的完整工作流程")
    print("="*80)
    
    # 创建IF层
    T = 4
    if_layer = IF(T=T, L=8, thresh=8.0)
    
    # 准备输入数据
    batch_size = 2
    features = 3
    input_data = torch.randn(batch_size, features) * 5  # 放大便于观察
    
    print(f"📥 原始输入: {input_data.shape}")
    print(f"数据:\n{input_data}")
    
    # 为SNN模式准备输入
    snn_input = input_data.repeat(T, 1, 1)  # [T, batch, features]
    snn_input_flat = snn_input.view(T * batch_size, features)  # [T*batch, features]
    
    print(f"\n🔧 SNN预处理输入: {snn_input_flat.shape}")
    
    # 跟踪IF层内部的时间维度操作
    print(f"\n📋 IF层内部处理流程 (T={T}):")
    print("-" * 50)
    
    if_layer.eval()
    with torch.no_grad():
        # 模拟IF层的forward过程
        x = snn_input_flat
        print(f"1️⃣ IF层输入: {x.shape}")
        
        # ExpandTemporalDim
        x_expanded = if_layer.expand(x)
        print(f"2️⃣ Expand后: {x_expanded.shape}")
        print(f"   形状变化: {x.shape} → {x_expanded.shape}")
        
        # 模拟时序处理（简化版）
        thre = if_layer.thresh.data
        spike_pot = []
        mem = torch.zeros_like(x_expanded[0]) + 0.5 * thre
        
        print(f"3️⃣ 时序处理 (阈值={thre.item():.1f}):")
        for t in range(T):
            mem = mem + x_expanded[t]
            spike = (mem >= thre).float() * thre
            mem = mem - spike
            spike_pot.append(spike)
            print(f"   时间步{t}: 膜电位峰值={mem.max().item():.2f}, 脉冲数={spike.sum().item():.0f}")
        
        # 组合时序输出
        x_temporal = torch.stack(spike_pot, dim=0)
        print(f"4️⃣ 时序输出: {x_temporal.shape}")
        
        # MergeTemporalDim
        x_merged = if_layer.merge(x_temporal)
        print(f"5️⃣ Merge后: {x_merged.shape}")
        print(f"   形状变化: {x_temporal.shape} → {x_merged.shape}")
    
    print(f"\n🎯 关键理解:")
    print(f"   • Expand: 为时序处理重塑数据结构")
    print(f"   • 时序循环: 模拟神经元的动态过程")
    print(f"   • Merge: 恢复原始batch结构，便于下一层处理")


def compare_ann_snn_modes():
    """对比ANN和SNN模式下的处理差异"""
    print("\n" + "="*80)
    print("🆚 ANN vs SNN 模式对比")
    print("="*80)
    
    # 创建相同的输入数据
    batch_size = 2
    features = 3
    input_data = torch.tensor([[1.0, 4.0, 9.0],
                              [2.0, 6.0, 12.0]])
    
    print(f"📥 测试输入: {input_data.shape}")
    print(f"数据:\n{input_data}")
    
    # ANN模式 (T=0)
    print(f"\n1️⃣ ANN模式 (T=0):")
    print("-" * 30)
    
    if_ann = IF(T=0, L=8, thresh=8.0)
    if_ann.eval()
    
    with torch.no_grad():
        output_ann = if_ann(input_data)
    
    print(f"   输入: {input_data.shape}")
    print(f"   输出: {output_ann.shape}")
    print(f"   数据:\n{output_ann}")
    print(f"   特点: 直接量化，无时间维度操作")
    
    # SNN模式 (T=4)
    print(f"\n2️⃣ SNN模式 (T=4):")
    print("-" * 30)
    
    T = 4
    if_snn = IF(T=T, L=8, thresh=8.0)
    if_snn.eval()
    
    # 准备SNN输入
    snn_input = input_data.repeat(T, 1, 1)
    snn_input_flat = snn_input.view(T * batch_size, features)
    
    with torch.no_grad():
        output_snn = if_snn(snn_input_flat)
    
    print(f"   输入: {snn_input_flat.shape}")
    print(f"   输出: {output_snn.shape}")
    print(f"   数据:\n{output_snn}")
    print(f"   特点: 时序处理，使用Expand/Merge操作")
    
    # 分析输出差异
    print(f"\n📊 输出分析:")
    print("-" * 30)
    print(f"   • ANN输出范围: [{output_ann.min().item():.2f}, {output_ann.max().item():.2f}]")
    print(f"   • SNN输出范围: [{output_snn.min().item():.2f}, {output_snn.max().item():.2f}]")
    print(f"   • ANN模式: 连续量化值")
    print(f"   • SNN模式: 离散脉冲值（通常是0或阈值）")


def explain_design_rationale():
    """解释设计原理"""
    print("\n" + "="*80)
    print("🎯 设计原理解析")
    print("="*80)
    
    print("🔑 为什么需要这两个操作？")
    print("-" * 40)
    
    print("1️⃣ 兼容性问题:")
    print("   • ANN: [batch, features] 格式")
    print("   • SNN: [time, batch, features] 格式")
    print("   • 需要动态转换以支持双模式")
    
    print("\n2️⃣ 时序建模需求:")
    print("   • SNN需要模拟神经元的时间动态")
    print("   • 膜电位积累、阈值判断、脉冲发放")
    print("   • 必须按时间步逐步处理")
    
    print("\n3️⃣ 网络流水线:")
    print("   • 前一层输出: [T*batch, features]")
    print("   • IF层处理: [T, batch, features]")
    print("   • 后一层输入: [T*batch, features]")
    print("   • 保持网络层间的数据格式一致性")
    
    print("\n✅ 核心价值:")
    print("-" * 30)
    print("🎯 实现了同一个网络的双模式运行:")
    print("   • T=0: ANN模式，量化感知训练")
    print("   • T>0: SNN模式，脉冲神经网络")
    print("   • 无需重新设计网络结构")
    print("   • 训练一次，两种推理模式")


def main():
    """主函数"""
    explain_temporal_operations()
    demonstrate_if_layer_workflow()
    compare_ann_snn_modes()
    explain_design_rationale()
    
    print("\n" + "="*80)
    print("📝 总结")
    print("="*80)
    print("🔄 时间维度处理的两个关键步骤:")
    print("")
    print("🔸 ExpandTemporalDim:")
    print("   • 作用: [T*batch, features] → [T, batch, features]")
    print("   • 时机: SNN模式下，IF层处理前")
    print("   • 目的: 为时序神经元建模重组数据")
    print("")
    print("🔸 MergeTemporalDim:")
    print("   • 作用: [T, batch, features] → [T*batch, features]")
    print("   • 时机: SNN模式下，IF层处理后")
    print("   • 目的: 恢复网络兼容的数据格式")
    print("")
    print("🎯 这种设计实现了ANN-SNN的无缝转换，是QCFS方法的核心创新之一！")


if __name__ == "__main__":
    main() 