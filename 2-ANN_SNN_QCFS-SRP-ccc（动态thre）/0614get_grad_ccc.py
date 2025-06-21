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
import warnings
# from Models.layer import load_model_compatible

# 设置环境变量抑制cuDNN警告
os.environ['CUDNN_V8_API_DISABLED'] = '1'
warnings.filterwarnings("ignore", category=UserWarning)
# 抑制PyTorch相关警告
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


class GradientAnalyzer:
    """全连接层梯度分析器"""
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
        分析全连接层梯度分布
        
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
            gradient_stats[name] = {'values': None}
        
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
            
            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total = targets.size(0)
            correct = (predicted == targets).sum().item()
            accuracy = 100 * correct / total
            
            # 反向传播（触发梯度钩子）
            loss.backward()
            
            # 收集梯度数据
            for name, grads in self.gradient_records.items():
                if grads is not None:
                    if gradient_stats[name]['values'] is None:
                        # 第一个批次，直接赋值
                        gradient_stats[name]['values'] = grads.numpy()
                    else:
                        # 后续批次，累加梯度
                        gradient_stats[name]['values'] += grads.numpy()
            
            batch_count += 1
        
        # 计算平均梯度
        for name in gradient_stats:
            if gradient_stats[name]['values'] is not None:
                gradient_stats[name]['values'] = gradient_stats[name]['values'] / batch_count
        
        # 计算梯度统计
        print("\n计算梯度统计信息...")
        for name, stats in gradient_stats.items():
            if stats['values'] is not None:
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

    def get_low_gradient_neurons(self, gradient_stats,order='low', ratio=0.1):
        """
        识别低梯度神经元
        
        参数:
        gradient_stats - analyze_gradients返回的统计数据
        order - 排序方式: 'low'(梯度从小到大), 'high'(梯度从大到小), 'index'(按神经元序号), 'random'(随机)
        ratio - 要识别的神经元比例
        
        返回:
        low_gradient_neurons - 低梯度神经元列表
        """
        low_neurons = []
        
        # 处理每层的梯度统计
        for layer_name, stats in gradient_stats.items():
            if layer_name == 'classifier.7':     # 跳过最后一层
                continue
            if 'values' not in stats or stats['values'] is None:
                continue
                
            # 对梯度值排序
            grads = np.array(stats['values'])
            if order == 'low':
                sorted_indices = np.argsort(grads)  # 从小到大排序
            elif order == 'high':
                sorted_indices = np.argsort(grads)[::-1]  # 从大到小排序
            elif order == 'index':
                sorted_indices = np.arange(len(grads))  # 按神经元序号排序
            elif order == 'random':
                # 随机打乱索引
                sorted_indices = np.random.permutation(len(grads))
            else:
                # 默认按梯度从小到大排序
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
            if 'values' not in stats or stats['values'] is None:
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
            low_neurons = self.get_low_gradient_neurons(gradient_stats,'low', ratio)
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
    
    def prune_neurons(self, neurons_to_prune):
        """执行神经元剪枝"""
        # 统计每层剪枝的神经元数量
        layer_prune_count = {}
        
        for neuron_info in neurons_to_prune:
            layer_name = neuron_info['layer']
            neuron_idx = neuron_info['neuron_index']
            
            # 更新统计信息
            if layer_name not in layer_prune_count:
                layer_prune_count[layer_name] = 0
            layer_prune_count[layer_name] += 1
            
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
        
        # 打印每层剪枝统计信息
        print("\n剪枝统计信息:")
        print("="*60)
        print(f"{'层名称':<30} {'剪枝神经元数量':<15} {'总神经元数量':<15}")
        print("-"*60)
        
        total_pruned = 0
        for layer_name, count in layer_prune_count.items():
            # 获取该层的总神经元数量
            for name, module in self.model.named_modules():
                if name == layer_name and isinstance(module, nn.Linear):
                    total_neurons = module.out_features
                    print(f"{layer_name:<30} {count:<15} {total_neurons:<15}")
                    total_pruned += count
                    break
        
        print("-"*60)
        print(f"总计剪枝神经元数量: {total_pruned}")
        print("="*60)
    
    def cleanup_hooks(self):
        """清理梯度钩子"""
        for handle in self.gradient_hooks.values():
            handle.remove()
        self.gradient_hooks = {}
        self.gradient_records = {}


    def save_neuronidx_weight_grad(self, model, gradient_stats, timestamp, before_pruning_state=None):
        """
        保存权重分析信息到CSV文件
        
        参数:
        model - 模型
        gradient_stats - 梯度统计信息
        timestamp - 时间戳
        before_pruning_state - 剪枝前的模型状态
        """
        print("\n开始保存权重分析信息...")
        
        # 确保log目录存在
        log_dir = "log_weight_grad"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            print(f"创建目录: {log_dir}")
        
        # 遍历所有全连接层
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                print(f"分析层: {name}")
                
                # 获取剪枝后的权重数据
                weight_data = module.weight.data
                out_features, in_features = weight_data.shape
                
                # 计算剪枝后的平均权重
                pruned_weights = weight_data.abs().mean(dim=1).cpu().numpy()
                
                # 获取剪枝前的平均权重
                if before_pruning_state is not None and f"{name}.weight" in before_pruning_state:
                    before_weight_data = before_pruning_state[f"{name}.weight"]
                    avg_befor_weights = before_weight_data.abs().mean(dim=1).cpu().numpy()
                else:
                    # 如果没有剪枝前状态，抛出异常
                    raise ValueError(f"层 {name} 没有剪枝前状态信息，无法保存权重分析数据")
                
                # 获取对应的梯度值
                if name not in gradient_stats or gradient_stats[name]['values'] is None:
                    raise ValueError(f"层 {name} 没有梯度信息，无法保存权重分析数据")
                
                gradient_values = gradient_stats[name]['values']
                
                # 创建DataFrame
                df = pd.DataFrame({
                    'neuron_index': range(out_features),
                    'avg_befor_weight': avg_befor_weights,
                    'pruned_weight': pruned_weights,
                    'gradient_value': gradient_values
                })
                
                # 生成文件名（简化格式）
                filename = f"{name}_weight_grad_{timestamp}.csv"
                filepath = os.path.join(log_dir, filename)
                
                # 保存到CSV
                df.to_csv(filepath, index=False)
        #         print(f"  已保存权重信息到: {filepath}")
        #         print(f"  神经元数量: {out_features}")
        #         print(f"  剪枝前权重统计:")
        #         print(f"    均值: {np.mean(avg_befor_weights):.8f}")
        #         print(f"    标准差: {np.std(avg_befor_weights):.8f}")
        #         print(f"    最小值: {np.min(avg_befor_weights):.8f}")
        #         print(f"    最大值: {np.max(avg_befor_weights):.8f}")
        #         print(f"    中位数: {np.median(avg_befor_weights):.8f}")
        #         print(f"  剪枝后权重统计:")
        #         print(f"    均值: {np.mean(pruned_weights):.8f}")
        #         print(f"    标准差: {np.std(pruned_weights):.8f}")
        #         print(f"    最小值: {np.min(pruned_weights):.8f}")
        #         print(f"    最大值: {np.max(pruned_weights):.8f}")
        #         print(f"    中位数: {np.median(pruned_weights):.8f}")
        #         print(f"  梯度值统计:")
        #         print(f"    均值: {np.mean(gradient_values):.8f}")
        #         print(f"    标准差: {np.std(gradient_values):.8f}")
        #         print(f"    最小值: {np.min(gradient_values):.8f}")
        #         print(f"    最大值: {np.max(gradient_values):.8f}")
        #         print(f"    中位数: {np.median(gradient_values):.8f}")
        
        # print("权重分析信息保存完成!")

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

def evaluate_model(model, test_loader, criterion, device, seed=42):
    """评估模型性能"""
    # 设置随机种子，确保数据加载顺序一致
    seed_all(seed)
    
    # 保存模型原始状态
    original_state = {
        'training': model.training,
        'state_dict': model.state_dict().copy()
    }
    
    model.eval()  # 确保模型在评估模式
    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    
    with torch.no_grad():  # 禁用梯度计算
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(images)
            
            # 处理SNN输出
            if len(outputs.shape) > 2:
                outputs = outputs.mean(0)  # 对时间维度求平均
            
            # 计算损失
            loss = criterion(outputs, labels)
            
            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total = labels.size(0)
            correct = (predicted == labels).sum().item()
            
            # 累加统计信息
            total_correct += correct
            total_samples += total
            total_loss += loss.item()
    
    # 计算平均准确率和损失
    avg_accuracy = 100 * total_correct / total_samples
    avg_loss = total_loss / len(test_loader)
    # print(f"\n评估完成:")
    print(f"  总样本数: {total_samples}")
    print(f"  平均准确率: {avg_accuracy:.2f}% ({total_correct}/{total_samples})")
    print(f"  平均损失: {avg_loss:.6f}")
    
    # 恢复模型原始状态
    model.load_state_dict(original_state['state_dict'])
    if original_state['training']:
        model.train()
    else:
        model.eval()
    
    return avg_accuracy, avg_loss

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='获取VGG16参数和梯度')
    parser.add_argument('--batch_size', default=200, type=int, help='批次大小')
    parser.add_argument('--device', default='0', type=str, help='设备')
    parser.add_argument('--seed', default=42, type=int, help='随机种子')
    parser.add_argument('--mode', choices=['ann', 'snn'], default='snn', help='模式')
    parser.add_argument('--num_batches', default=5, type=int, help='梯度分析的批次数')
    parser.add_argument('-r','--pruning_ratio', default=0.5, type=float, help='剪枝比例')
    parser.add_argument('--dataset', choices=['cifar10', 'cifar100'], default='cifar10', help='数据集')
    parser.add_argument('--order', default='random', type=str, help='low/high/index/random 梯度从小到大/从大到小/按神经元序号/随机排序')
    
    args = parser.parse_args()
    
    # 设置输出重定向（默认保存到文件）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"gradient_analysis_{args.mode}_{timestamp}.txt"
    output_redirector = OutputRedirector(filename)
    sys.stdout = output_redirector
    # print(f"输出将保存到文件: {filename}")
    
    # 设置环境
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(args.seed)
    
    # print(f"设备: {device}, 随机种子: {args.seed}")
    # print(f"分析模式: {args.mode}")
    # print(f"梯度分析批次数: {args.num_batches}")
    print(f"剪枝比例: {args.pruning_ratio}")
    # print(f"数据集: {args.dataset}")
    print(f"梯度排序方式: {args.order}")
    
    # 创建模型
    model = modelpool('vgg16', args.dataset)
    
    # 直接加载预训练模型
    model_path = '/root/autodl-tmp/0-ANN2SNN-Allinone/2-ANN_SNN_QCFS-SRP/cifar10-checkpoints/vgg16_wd[0.0005].pth'
    # model_path = '/root/autodl-tmp/0-ANN2SNN-Allinone/2-ANN_SNN_QCFS-SRP/cifar100-checkpoints/vgg16_L[4].pth'
    
    # print(f"加载预训练模型: {model_path}")
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    model.load_state_dict(state_dict)
    # 使用兼容性加载函数
    # try:
    #     load_model_compatible(model, state_dict)
    # except Exception as e:
    #     print(f"兼容性加载失败，尝试常规加载: {e}")
    #     model.load_state_dict(state_dict, strict=False)  # 使用非严格模式作为备选

    # print("✅ 预训练模型加载成功")
    
    if args.mode == 'snn':
        model.set_T(4)
        model.set_L(4)
        # print("设置为SNN模式")
    else:
        model.set_T(0)
        model.set_L(4)
        # print("设置为ANN模式")
    
    model.to(device)
    
    # 加载数据
    # print(f"加载{args.dataset}测试数据集...")
    train_loader, test_loader = datapool(args.dataset, args.batch_size)
    
    # 使用测试集进行评估
    criterion = nn.CrossEntropyLoss()
    
    # 保存模型初始状态（深拷贝）
    initial_state = {}
    for key, value in model.state_dict().items():
        initial_state[key] = value.clone().detach()
    
    # 剪枝前评估
    print("\n剪枝前评估:")
    pre_accuracy, pre_loss = evaluate_model(model, test_loader, criterion, device, args.seed)
    
    # 分析全连接层梯度
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
        # analyzer.print_gradient_analysis(gradient_stats)    
        
        # 获取低梯度神经元
        low_gradient_neurons = analyzer.get_low_gradient_neurons(
            gradient_stats, 
            order=args.order,
            ratio=args.pruning_ratio
        )
        
        # 执行剪枝
        analyzer.prune_neurons(low_gradient_neurons)
        
        # 剪枝后评估
        print("\n剪枝后评估:")
        post_accuracy, post_loss = evaluate_model(model, test_loader, criterion, device, args.seed)
        
        # 保存各层梯度信息到CSV文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for layer_name, stats in gradient_stats.items():
            if 'values' not in stats or stats['values'] is None:
                continue
            
            # 获取该层的神经元数量
            num_neurons = stats['num_neurons']
            
            # 创建DataFrame
            df = pd.DataFrame({
                'neuron_index': range(num_neurons),
                'gradient_value': stats['values']
            })
            
            # 生成文件名
            # filename = f"gradient_analysis_{layer_name}_{timestamp}.csv"
            
            # 保存到CSV
            # df.to_csv(filename, index=False)
            # print(f"已保存{layer_name}层的梯度信息到: {filename}")
            # print(f"  神经元数量: {num_neurons}")
            # print(f"  每个神经元包含1个平均梯度值")
                  
    finally:
        # 保存权重分析信息
        analyzer.save_neuronidx_weight_grad(model, gradient_stats, timestamp, initial_state)
        
        # 清理梯度钩子
        analyzer.cleanup_hooks()
    
    print("\n✅ 完成!")

if __name__ == "__main__":
    main() 