#!/usr/bin/env python3
"""
详细分析VGG classifier结构中的IF层
"""

import torch
import torch.nn as nn
from Models import modelpool
from Models.layer import IF

def analyze_data_flow():
    """分析数据流经classifier的过程"""
    print("🔍 VGG16 Classifier数据流分析")
    print("="*80)
    
    # 创建模型
    model = modelpool('vgg16', 'cifar10')
    model.eval()
    
    # 创建测试输入 (CIFAR-10: 32x32x3)
    batch_size = 4
    test_input = torch.randn(batch_size, 3, 32, 32)
    
    print("📥 输入数据:")
    print(f"  原始输入形状: {test_input.shape} (batch, channels, height, width)")
    
    # 通过卷积层获得feature maps
    with torch.no_grad():
        out = model.layer1(test_input)
        print(f"  Layer1输出: {out.shape}")
        out = model.layer2(out)
        print(f"  Layer2输出: {out.shape}")
        out = model.layer3(out)
        print(f"  Layer3输出: {out.shape}")
        out = model.layer4(out)
        print(f"  Layer4输出: {out.shape}")
        out = model.layer5(out)
        print(f"  Layer5输出: {out.shape}")
        
        feature_maps = out
        print(f"\n🎯 进入Classifier前的特征: {feature_maps.shape}")
        
        # 分步分析classifier的每一层
        print("\n📋 Classifier逐层分析:")
        print("-" * 60)
        
        # Layer 0: Flatten
        current = model.classifier[0](feature_maps)
        print(f"[0] Flatten: {feature_maps.shape} → {current.shape}")
        print(f"    扁平化特征向量，每个样本有 {current.shape[1]} 个特征")
        
        # Layer 1: Linear(512, 4096)
        current = model.classifier[1](current)
        print(f"[1] Linear(512→4096): {current.shape}")
        print(f"    第1个全连接层，输出 {current.shape[1]} 个神经元")
        
        # Layer 2: IF() - 第1个IF层
        if_layer_1 = model.classifier[2]
        current = if_layer_1(current)
        print(f"[2] IF(): {current.shape}")
        print(f"    🔸 第1个IF层作用于 {current.shape[1]} 个神经元")
        print(f"    🔸 阈值参数: {if_layer_1.thresh.data.item()}")
        print(f"    🔸 每个神经元的输出都受到同一个阈值控制")
        
        # Layer 3: Dropout
        current = model.classifier[3](current)
        print(f"[3] Dropout: {current.shape}")
        print(f"    随机失活，训练时防止过拟合")
        
        # Layer 4: Linear(4096, 4096)
        current = model.classifier[4](current)
        print(f"[4] Linear(4096→4096): {current.shape}")
        print(f"    第2个全连接层，输出 {current.shape[1]} 个神经元")
        
        # Layer 5: IF() - 第2个IF层
        if_layer_2 = model.classifier[5]
        current = if_layer_2(current)
        print(f"[5] IF(): {current.shape}")
        print(f"    🔸 第2个IF层作用于 {current.shape[1]} 个神经元")
        print(f"    🔸 阈值参数: {if_layer_2.thresh.data.item()}")
        print(f"    🔸 每个神经元的输出都受到同一个阈值控制")
        
        # Layer 6: Dropout
        current = model.classifier[6](current)
        print(f"[6] Dropout: {current.shape}")
        
        # Layer 7: Linear(4096, 10)
        current = model.classifier[7](current)
        print(f"[7] Linear(4096→10): {current.shape}")
        print(f"    输出层，10个类别的logits")
        
        print(f"\n📤 最终输出: {current.shape}")


def compare_if_vs_relu():
    """比较IF层与ReLU的区别"""
    print("\n" + "="*80)
    print("🆚 IF层 vs ReLU 详细对比")
    print("="*80)
    
    # 创建测试数据
    test_data = torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0, 8.0, 10.0, 16.0]])
    
    print("📊 测试数据:", test_data.flatten().tolist())
    
    # ReLU激活
    relu = nn.ReLU()
    relu_output = relu(test_data)
    print(f"ReLU输出:   {relu_output.flatten().tolist()}")
    
    # IF层激活 (ANN模式, T=0)
    if_layer = IF(T=0, L=8, thresh=8.0)
    if_output = if_layer(test_data)
    print(f"IF输出(ANN): {if_output.flatten().tolist()}")
    
    print("\n🔍 关键区别分析:")
    print("-" * 50)
    
    print("1️⃣ ReLU函数:")
    print("   • 公式: max(0, x)")
    print("   • 特点: 线性整流，负值归零，正值保持")
    print("   • 输出: 连续值，范围 [0, +∞)")
    
    print("\n2️⃣ IF层 (ANN模式):")
    print("   • 公式: x = x/thresh → clamp(0,1) → quantize → x*thresh")
    print("   • 特点: 分段线性量化")
    print(f"   • 阈值: {if_layer.thresh.data.item()}")
    print(f"   • 量化级别: {if_layer.L}")
    print("   • 输出: 离散化值，有限精度")
    
    print("\n3️⃣ IF层 (SNN模式, T>0):")
    print("   • 功能: 积分发放神经元")
    print("   • 特点: 膜电位积累，达到阈值发放脉冲")
    print("   • 输出: 脉冲序列 (0或thresh)")


def analyze_neuron_count():
    """分析IF层对应的神经元数量"""
    print("\n" + "="*80)
    print("🧠 IF层神经元数量详细分析")
    print("="*80)
    
    model = modelpool('vgg16', 'cifar10')
    
    print("📋 Classifier结构详解:")
    print("-" * 50)
    
    classifier_info = [
        ("Flatten", "将(4,512,1,1)扁平化为(4,512)", "数据预处理"),
        ("Linear(512→4096)", "512个输入特征 → 4096个神经元", "特征扩展"),
        ("IF() #1", "作用于4096个神经元", "激活+量化"),
        ("Dropout", "随机失活4096个神经元中的部分", "正则化"),
        ("Linear(4096→4096)", "4096个输入 → 4096个神经元", "特征变换"),
        ("IF() #2", "作用于4096个神经元", "激活+量化"),
        ("Dropout", "随机失活4096个神经元中的部分", "正则化"),
        ("Linear(4096→10)", "4096个输入 → 10个输出", "分类输出")
    ]
    
    for i, (layer_name, neuron_info, function) in enumerate(classifier_info):
        print(f"[{i}] {layer_name:15} | {neuron_info:25} | {function}")
    
    print("\n🎯 关键理解:")
    print("-" * 50)
    print("🔸 IF层#1: 控制4096个神经元，但只有1个阈值参数")
    print("🔸 IF层#2: 控制4096个神经元，但只有1个阈值参数")
    print("🔸 每个IF层的阈值是全局共享的，不是逐神经元独立的")
    print("🔸 这4096个神经元的激活行为由同一个阈值统一控制")
    
    print("\n💡 设计原理:")
    print("-" * 50)
    print("✅ 优势:")
    print("   • 参数效率: 2个参数 vs 8192个参数 (如果逐神经元)")
    print("   • 硬件友好: 统一阈值易于FPGA/ASIC实现")
    print("   • 生物合理: 符合真实神经元的层级组织")
    print("   • 训练稳定: 减少了参数空间，避免过拟合")
    
    print("\n❌ 传统ReLU网络:")
    print("   • 每个神经元独立激活")
    print("   • 没有统一的激活控制")
    print("   • 难以直接转换为SNN")


def demonstrate_if_modes():
    """演示IF层在不同模式下的行为"""
    print("\n" + "="*80)
    print("🔄 IF层双模式工作演示")
    print("="*80)
    
    # 创建测试输入
    test_input = torch.tensor([[0.5, 2.0, 4.0, 8.0, 12.0, 16.0]])
    
    print(f"📥 测试输入: {test_input.flatten().tolist()}")
    
    # ANN模式 (T=0)
    print("\n1️⃣ ANN模式 (T=0) - 量化激活:")
    print("-" * 40)
    
    if_ann = IF(T=0, L=8, thresh=8.0)
    output_ann = if_ann(test_input)
    
    print(f"   阈值: {if_ann.thresh.data.item()}")
    print(f"   量化级别: {if_ann.L}")
    print(f"   输出: {output_ann.flatten().tolist()}")
    print("   作用: 实现分段线性量化，保持ANN计算")
    
    # SNN模式 (T=4)
    print("\n2️⃣ SNN模式 (T=4) - 脉冲发放:")
    print("-" * 40)
    
    if_snn = IF(T=4, L=8, thresh=8.0)
    if_snn.eval()
    
    # 需要扩展时间维度
    test_input_expanded = test_input.unsqueeze(0).repeat(4, 1, 1)  # [T, batch, features]
    test_input_flattened = test_input_expanded.view(-1, test_input.shape[-1])  # [T*batch, features]
    
    output_snn = if_snn(test_input_flattened)
    output_snn_reshaped = output_snn.view(4, 1, -1)  # [T, batch, features]
    
    print(f"   阈值: {if_snn.thresh.data.item()}")
    print(f"   时间步: {if_snn.T}")
    print("   各时间步输出:")
    for t in range(4):
        print(f"     T={t}: {output_snn_reshaped[t].flatten().tolist()}")
    print("   作用: 积分发放机制，输出脉冲序列")


def main():
    """主函数"""
    # 分析数据流
    analyze_data_flow()
    
    # 比较IF vs ReLU
    compare_if_vs_relu()
    
    # 分析神经元数量
    analyze_neuron_count()
    
    # 演示IF双模式
    demonstrate_if_modes()
    
    print("\n" + "="*80)
    print("📝 总结")
    print("="*80)
    print("🎯 classifier中的两个IF层:")
    print("   • IF#1: 作用于4096个神经元 (Linear 512→4096之后)")
    print("   • IF#2: 作用于4096个神经元 (Linear 4096→4096之后)")
    print("   • 每个IF层只有1个阈值参数，但控制整层的激活行为")
    print("   • 不仅仅是ReLU替代，而是支持ANN-SNN双模式的智能激活函数")
    print("   • QCFS方法的核心创新：量化感知训练 + 脉冲神经网络转换")


if __name__ == "__main__":
    main() 