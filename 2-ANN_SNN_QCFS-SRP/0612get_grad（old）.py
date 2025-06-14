#!/usr/bin/env python3


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import os
import sys
from datetime import datetime
from Models import modelpool
from Preprocess import datapool
from utils import seed_all
import pandas as pd

def get_params_grad(model):
    """
    获取模型参数和对应的梯度
    参考自hessian_weight_importance.py
    """
    params = []
    grads = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        params.append(param)
        grads.append(0. if param.grad is None else param.grad + 0.)
    return params, grads

def print_params_and_gradients(model):
    """打印模型参数和梯度信息"""
    print("="*80)
    print("VGG16模型参数和梯度信息")
    print("="*80)
    
    params, grads = get_params_grad(model)
    
    print(f"总共有 {len(params)} 个需要梯度的参数")
    print("-"*80)
    
    total_params = 0
    total_grad_norm = 0.0
    
    for i, (param, grad) in enumerate(zip(params, grads)):
        param_count = param.numel()
        total_params += param_count
        
        # 梯度统计
        if torch.is_tensor(grad):
            grad_norm = grad.norm().item()
            grad_mean = grad.mean().item()
            grad_std = grad.std().item()
            non_zero_elements = (grad != 0).sum().item()
            total_grad_norm += grad_norm ** 2
        else:
            grad_norm = grad_mean = grad_std = 0.0
            non_zero_elements = 0
        
        print(f"参数 {i+1:2d}: 形状={list(param.shape)}, 数量={param_count:,}")
        print(f"  参数统计: 均值={param.mean().item():.6f}, 标准差={param.std().item():.6f}")
        print(f"  梯度统计: 范数={grad_norm:.6f}, 均值={grad_mean:.6f}, 标准差={grad_std:.6f}")
        print(f"  非零梯度: {non_zero_elements:,}/{param_count:,} ({100*non_zero_elements/param_count:.2f}%)")
        print("-"*40)
    
    total_grad_norm = np.sqrt(total_grad_norm)
    print(f"\n总结: {total_params:,} 个参数, 总梯度范数: {total_grad_norm:.6f}")
    print("="*80)

def print_if_layers_only(model):
    """只打印IF层的参数和梯度信息"""
    print("="*80)
    print("IF层阈值参数和梯度信息")
    print("="*80)
    
    if_layer_count = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # 只显示IF层的阈值参数
        if 'thresh' in name.lower() and param.numel() == 1:
            if_layer_count += 1
            print(f"IF层阈值: {name}")
            print(f"  形状: {list(param.shape)}, 参数数量: {param.numel():,}")
            print(f"  阈值: {param.data.item():.6f}")
            
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_mean = param.grad.mean().item()
                grad_value = param.grad.data.item()
                print(f"  梯度值: {grad_value:.6f}")
                print(f"  梯度范数: {grad_norm:.6f}, 梯度均值: {grad_mean:.6f}")
            else:
                print(f"  梯度: None")
            print("-"*60)
    
    if if_layer_count == 0:
        print("未找到IF层阈值参数")
    else:
        print(f"总共找到 {if_layer_count} 个IF层阈值参数")
    print("="*80)

def print_all_if_module_info(model):
    """打印所有IF模块的详细信息"""
    print("="*80)
    print("IF模块详细信息")
    print("="*80)
    
    from Models.layer import IF
    
    if_module_count = 0
    for name, module in model.named_modules():
        if isinstance(module, IF):
            if_module_count += 1
            print(f"IF模块: {name}")
            print(f"  阈值(thresh): {module.thresh.item():.6f}")
            print(f"  gamma参数: {module.gama}")
            print(f"  时间步数(T): {module.T}")
            print(f"  量化级别(L): {module.L}")
            
            # 打印阈值参数的梯度
            if module.thresh.grad is not None:
                thresh_grad = module.thresh.grad.item()
                print(f"  阈值梯度: {thresh_grad:.6f}")
            else:
                print(f"  阈值梯度: None")
            
            print("-"*60)
    
    if if_module_count == 0:
        print("未找到IF模块")
    else:
        print(f"总共找到 {if_module_count} 个IF模块")
    print("="*80)

class GradientAnalyzer:
    """全连接层梯度分析器（参考gradient_cray.py）"""
    def __init__(self, model):
        self.model = model
        self.gradient_hooks = {}
        self.gradient_records = {}
        
    def register_gradient_hooks(self):
        """为所有全连接层注册梯度记录钩子"""
        print("注册全连接层梯度钩子...")
        
        # 移除现有钩子
        for handle in self.gradient_hooks.values():
            handle.remove()
        self.gradient_hooks = {}
        self.gradient_records = {}
        
        # 查找所有全连接层
        fc_count = 0
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                fc_count += 1
                # 为权重参数注册梯度钩子
                hook = self._gradient_hook(name)
                handle = module.weight.register_hook(hook)
                self.gradient_hooks[name] = handle
                print(f"  - 注册钩子: {name} (输入={module.in_features}, 输出={module.out_features})")
        
        print(f"总共注册了 {fc_count} 个全连接层的梯度钩子")
        
    def _gradient_hook(self, name):
        """创建梯度钩子函数"""
        def hook(grad):
            # 确保梯度有效
            if grad is None:
                return
            
            # 计算每个输出神经元的平均梯度
            if grad.dim() > 1:
                # 全连接层: 对输入维度求平均
                neuron_grads = grad.abs().mean(dim=1)  # [out_features]
            else:
                # 1D情况
                neuron_grads = grad.abs()
            
            # 保存梯度统计信息
            self.gradient_records[name] = neuron_grads.detach().cpu()
        return hook
    
    def analyze_gradients(self, dataloader, criterion, num_batches=5):
        """
        分析全连接层梯度分布（参考gradient_cray.py）
        
        参数:
        dataloader - 数据加载器
        criterion - 损失函数
        num_batches - 分析批次数
        
        返回:
        gradient_stats - 梯度统计信息
        """
        print(f"\n开始分析 {num_batches} 个批次的梯度分布...")
        
        # 注册梯度钩子
        self.register_gradient_hooks()
        
        # 确保模型处于训练模式
        self.model.train()
        
        # 梯度统计收集器
        gradient_stats = {}
        for name in self.gradient_hooks.keys():
            gradient_stats[name] = {'values': []}
        
        # 处理指定批次数据
        batch_count = 0
        data_iter = iter(dataloader)
        
        for batch_idx in range(num_batches):
            try:
                inputs, targets = next(data_iter)
                inputs, targets = inputs.to(next(self.model.parameters()).device), targets.to(next(self.model.parameters()).device)
            except StopIteration:
                print(f"数据不足，只处理了 {batch_idx} 个批次")
                break
                
            # 清空梯度
            self.model.zero_grad()
            
            # 前向传播
            outputs = self.model(inputs)
            
            # 处理SNN输出
            if len(outputs.shape) > 2:
                outputs = outputs.mean(0)  # 对时间维度求平均
            
            # 计算损失
            loss = criterion(outputs, targets)
            
            # 反向传播（触发梯度钩子）
            loss.backward()
            
            # 收集梯度数据
            for name, grads in self.gradient_records.items():
                if grads is not None:
                    gradient_stats[name]['values'].extend(grads.numpy())
            
            batch_count += 1
            print(f"  处理批次 {batch_count}/{num_batches}, 损失: {loss.item():.6f}")
        
        # 计算梯度统计
        print("\n计算梯度统计信息...")
        for name, stats in gradient_stats.items():
            if stats['values']:
                values = np.array(stats['values'])
                stats['mean'] = np.mean(values)
                stats['std'] = np.std(values)
                stats['min'] = np.min(values)
                stats['max'] = np.max(values)
                stats['median'] = np.median(values)
                stats['num_neurons'] = len(values)
                
                # 计算百分位数
                stats['p25'] = np.percentile(values, 25)
                stats['p75'] = np.percentile(values, 75)
                stats['p95'] = np.percentile(values, 95)
        
        return gradient_stats
    
    def get_low_gradient_neurons(self, gradient_stats, ratio=0.1):
        """
        识别低梯度神经元
        
        参数:
        gradient_stats - analyze_gradients返回的统计数据
        ratio - 要识别的神经元比例
        
        返回:
        low_gradient_neurons - 低梯度神经元列表
        """
        low_neurons = []
        
        # 处理每层的梯度统计
        for layer_name, stats in gradient_stats.items():
            if 'values' not in stats or not stats['values']:
                continue
                
            # 对梯度值排序
            grads = np.array(stats['values'])
            sorted_indices = np.argsort(grads)
            
            # 计算低梯度阈值
            num_low = int(len(grads) * ratio)
            
            # 收集低梯度神经元
            for idx in sorted_indices[:num_low]:
                low_neurons.append({
                    'layer': layer_name,
                    'neuron_index': idx,
                    'grad_value': grads[idx],
                    'grad_percentile': (np.searchsorted(np.sort(grads), grads[idx]) + 1) / len(grads)
                })
        
        return low_neurons
    
    def print_gradient_analysis(self, gradient_stats):
        """打印梯度分析结果"""
        print("="*80)
        print("全连接层梯度分布分析")
        print("="*80)
        
        if not gradient_stats:
            print("没有收集到梯度数据")
            return
        
        for layer_name, stats in gradient_stats.items():
            if not stats.get('values'):
                continue
                
            print(f"\n层: {layer_name}")
            print(f"  神经元数量: {stats['num_neurons']:,}")
            print(f"  梯度统计:")
            print(f"    均值: {stats['mean']:.8f}")
            print(f"    标准差: {stats['std']:.8f}")
            print(f"    最小值: {stats['min']:.8f}")
            print(f"    最大值: {stats['max']:.8f}")
            print(f"    中位数: {stats['median']:.8f}")
            print(f"  梯度分布:")
            print(f"    25%分位数: {stats['p25']:.8f}")
            print(f"    75%分位数: {stats['p75']:.8f}")
            print(f"    95%分位数: {stats['p95']:.8f}")
            print("-"*60)
        
        # 分析低梯度神经元
        print("\n低梯度神经元分析:")
        for ratio in [0.05, 0.1, 0.2]:
            low_neurons = self.get_low_gradient_neurons(gradient_stats, ratio)
            print(f"  梯度最低 {ratio*100:.1f}% 的神经元数量: {len(low_neurons)}")
            
            if low_neurons:
                # 按层分组统计
                layer_counts = {}
                for neuron in low_neurons:
                    layer = neuron['layer']
                    if layer not in layer_counts:
                        layer_counts[layer] = 0
                    layer_counts[layer] += 1
                
                for layer, count in layer_counts.items():
                    print(f"    {layer}: {count} 个")
        
        print("="*80)
        
    class GradientAnalyzer:
    def __init__(self, model, prune_ratio=0.1):
        self.model = model
        self.prune_ratio = prune_ratio
        self.gradient_records = defaultdict(list)
        
        # 为所有全连接层注册梯度钩子
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                hook = self.make_hook(name)
                module.weight.register_hook(hook)
    
    def make_hook(self, layer_name):
        """创建梯度钩子"""
        def gradient_hook(grad):
            # 计算每个输出神经元的平均梯度幅度
            # 对于全连接层，对输入维度取平均
            if grad.dim() > 1:
                grad_mag = grad.abs().mean(dim=1)  # 输出形状: [out_features]
            else:
                grad_mag = grad.abs()
            self.gradient_records[layer_name].append(grad_mag.detach().cpu())
        return gradient_hook
    
    def get_low_grad_neurons(self):
        """获取低梯度神经元信息"""
        all_neurons = []
        
        # 遍历所有记录的全连接层
        for layer_name, grad_list in self.gradient_records.items():
            if not grad_list:
                continue
                
            # 合并所有梯度记录
            stacked_grads = torch.stack(grad_list)
            
            # 计算平均梯度 (跨时间和批次)
            if stacked_grads.dim() > 1:
                neuron_avg_grad = stacked_grads.mean(dim=0).mean(dim=0)
            else:
                neuron_avg_grad = stacked_grads.mean()
            
            # 收集该层的神经元梯度信息
            for neuron_idx, grad_value in enumerate(neuron_avg_grad):
                all_neurons.append((layer_name, neuron_idx, grad_value.item()))
        
        # 按梯度值排序
        all_neurons.sort(key=lambda x: x[2])
        
        # 选择梯度最小的神经元
        num_to_prune = int(len(all_neurons) * self.prune_ratio)
        return all_neurons[:num_to_prune]
    
    def prune_neurons(self, neurons_to_prune):
        """执行神经元剪枝"""
        for layer_name, neuron_idx, _ in neurons_to_prune:
            # 找到对应的层
            module = None
            for name, mod in self.model.named_modules():
                if name == layer_name and isinstance(mod, nn.Linear):
                    module = mod
                    break
            
            if module:
                # 执行剪枝：将神经元的权重置零
                with torch.no_grad():
                    module.weight.data[neuron_idx] = 0
                    if module.bias is not None:
                        module.bias.data[neuron_idx] = 0
                    
                    # 可选：剪枝下游连接
                    self._prune_downstream(module, layer_name, neuron_idx)
    
    def _prune_downstream(self, pruned_module, layer_name, neuron_idx):
        """剪枝下游层的连接"""
        # 查找所有使用该层输出的模块
        downstream_layers = []
        
        # 检查所有后续层
        current_layer_found = False
        for name, module in self.model.named_modules():
            if name == layer_name:
                current_layer_found = True
                continue
            
            if current_layer_found and isinstance(module, (nn.Linear, LIFNeuron)):
                # 线性层直接使用该输出
                if isinstance(module, nn.Linear):
                    if module.weight.shape[1] == pruned_module.weight.shape[0]:
                        downstream_layers.append(module)
                # 如果是LIFNeuron，查找其前面的线性层
                elif isinstance(module, LIFNeuron):
                    # 获取前面的线性层
                    for prev_name, prev_module in self.model.named_modules():
                        if prev_name == name.replace("lif", "fc") and isinstance(prev_module, nn.Linear):
                            if prev_module.weight.shape[1] == pruned_module.weight.shape[0]:
                                downstream_layers.append(prev_module)
        
        # 剪枝下游层对应的输入权重
        for down_layer in downstream_layers:
            with torch.no_grad():
                # 剪枝输入连接
                down_layer.weight.data[:, neuron_idx] = 0
        
    
    def cleanup_hooks(self):
        """清理梯度钩子"""
        for handle in self.gradient_hooks.values():
            handle.remove()
        self.gradient_hooks = {}
        self.gradient_records = {}

class OutputRedirector:
    """输出重定向器，同时输出到控制台和文件"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='获取VGG16参数和梯度')
    parser.add_argument('--batch_size', default=32, type=int, help='批次大小')
    parser.add_argument('--device', default='0', type=str, help='设备')
    parser.add_argument('--seed', default=42, type=int, help='随机种子')
    parser.add_argument('--mode', choices=['ann', 'snn'], default='snn', help='模式')
    parser.add_argument('--num_batches', default=5, type=int, help='梯度分析的批次数')

    
    args = parser.parse_args()
    
    # 设置输出重定向（默认保存到文件）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"gradient_analysis_{args.mode}_{timestamp}.txt"
    output_redirector = OutputRedirector(filename)
    sys.stdout = output_redirector
    print(f"输出将保存到文件: {filename}")
    
    # 设置环境
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(args.seed)
    
    print(f"设备: {device}, 随机种子: {args.seed}")
    print(f"分析模式: {args.mode}")
    print(f"梯度分析批次数: {args.num_batches}")
    
    try:
        # 创建模型
        print("创建VGG16模型...")
        model = modelpool('vgg16', 'cifar10')
        
        # 直接加载预训练模型
        model_path = '/root/autodl-tmp/0-ANN2SNN-Allinone/2-ANN_SNN_QCFS-SRP/cifar10-checkpoints/vgg16_wd[0.0005].pth'
        print(f"加载预训练模型: {model_path}")
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        
        # 处理旧版本state_dict的兼容性
        keys = list(state_dict.keys())
        for k in keys:
            if "relu.up" in k:
                state_dict[k[:-7]+'act.thresh'] = state_dict.pop(k)
            elif "up" in k:
                state_dict[k[:-2]+'thresh'] = state_dict.pop(k)
        
        model.load_state_dict(state_dict)
        print("✅ 预训练模型加载成功")
        
        if args.mode == 'snn':
            model.set_T(8)
            model.set_L(4)
            print("设置为SNN模式")
        else:
            model.set_T(0)
            print("设置为ANN模式")
        
        model.to(device)
        model.train()
        
        # 加载数据
        print("加载CIFAR10数据集...")
        train_loader, test_loader = datapool('cifar10', args.batch_size)
        
        # 获取一批数据
        data_iter = iter(train_loader)
        images, labels = next(data_iter)
        images, labels = images.to(device), labels.to(device)
        
        print(f"输入形状: {images.shape}, 标签形状: {labels.shape}")
        
        # 前向传播
        print("执行前向传播...")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # 处理SNN输出
        if len(outputs.shape) > 2:
            outputs = outputs.mean(0)
        
        loss = criterion(outputs, labels)
        print(f"损失: {loss.item():.6f}")
        
        # 反向传播
        print("执行反向传播...")
        loss.backward()
        
        # 只打印IF层信息
        print_if_layers_only(model)
        print_all_if_module_info(model)
        
        # 分析全连接层梯度（默认启用）
        print("\n" + "="*80)
        print("开始全连接层梯度分析")
        print("="*80)
        
        # 创建梯度分析器
        analyzer = GradientAnalyzer(model)
        
        try:
            # 分析梯度分布
            gradient_stats = analyzer.analyze_gradients(
                train_loader, 
                criterion, 
                num_batches=args.num_batches
            )
            
            # 打印分析结果
            analyzer.print_gradient_analysis(gradient_stats)
            
            # 保存各层梯度信息到CSV文件
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            for layer_name, stats in gradient_stats.items():
                if not stats.get('values'):
                    continue
                
                # 获取该层的神经元数量（总梯度数除以batch数）
                num_neurons = len(stats['values']) // args.num_batches
                
                # 重塑数据为 [num_neurons, num_batches] 的形状
                gradient_values = np.array(stats['values']).reshape(num_neurons, args.num_batches)
                
                # 计算每个神经元的平均梯度值
                mean_gradients = np.mean(gradient_values, axis=1)
                
                # 创建DataFrame
                df = pd.DataFrame({
                    'neuron_index': range(num_neurons)
                })
                
                # 添加每个batch的梯度值列
                for i in range(args.num_batches):
                    df[f'gradient_batch_{i+1}'] = gradient_values[:, i]
                
                # 添加平均梯度值列
                df['gradient_mean'] = mean_gradients
                
                # 生成文件名
                filename = f"gradient_analysis_{layer_name}_{timestamp}.csv"
                
                # 保存到CSV
                df.to_csv(filename, index=False)
                print(f"已保存{layer_name}层的梯度信息到: {filename}")
                print(f"  神经元数量: {num_neurons}")
                print(f"  每个神经元包含{args.num_batches}个batch的梯度值和平均值")
            
            # 详细分析低梯度神经元
            print("\n" + "="*80)
            print("低梯度神经元详细分析")
            print("="*80)
            
            for ratio in [0.05, 0.1, 0.15, 0.2]:
                low_neurons = analyzer.get_low_gradient_neurons(gradient_stats, ratio)
                print(f"\n梯度最低 {ratio*100:.1f}% 的神经元详情:")
                
                if low_neurons:
                    # 按层分组显示
                    layer_groups = {}
                    for neuron in low_neurons:
                        layer = neuron['layer']
                        if layer not in layer_groups:
                            layer_groups[layer] = []
                        layer_groups[layer].append(neuron)
                    
                    for layer, neurons in layer_groups.items():
                        print(f"  {layer}: {len(neurons)} 个神经元")
                        # 显示前5个最低梯度的神经元
                        for i, neuron in enumerate(sorted(neurons, key=lambda x: x['grad_value'])[:5]):
                            print(f"    #{i+1}: 神经元{neuron['neuron_index']}, 梯度={neuron['grad_value']:.8f}, 百分位={neuron['grad_percentile']:.3f}")
                else:
                    print("  无数据")
            
        finally:
            # 清理梯度钩子
            analyzer.cleanup_hooks()
        
        print("\n✅ 完成!")
        print("\n💡 使用说明:")
        print("python 0612get_grad.py --mode snn  # SNN模式查看IF层信息和梯度分析")
        print("python 0612get_grad.py --mode ann  # ANN模式查看IF层信息和梯度分析")
        print("python 0612get_grad.py --mode snn --num_batches 10  # 指定分析批次数")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 恢复标准输出并关闭文件
        if output_redirector is not None:
            sys.stdout = output_redirector.terminal
            output_redirector.close()
            print(f"输出已保存到: {filename}")

if __name__ == "__main__":
    main() 