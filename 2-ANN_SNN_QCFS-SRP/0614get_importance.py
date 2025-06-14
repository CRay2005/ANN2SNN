#!/usr/bin/env python3
"""
改进的Hessian权重重要性计算器
解决数值精度问题和采样不足问题
"""

import torch
import torch.nn as nn
import numpy as np
import time
from tqdm import tqdm
from Models import modelpool
from Preprocess import datapool
import pandas as pd
from datetime import datetime
import argparse
import os
import sys
from utils import seed_all

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

def get_params_grad(model):
    """获取模型参数和对应的梯度"""
    params = []
    grads = []
    for param in model.parameters():
        if not param.requires_grad or param.grad is None:
            continue
        params.append(param)
        grads.append(param.grad)
    return params, grads


def hessian_vector_product(gradsH, params, v, stop_criterion=False):
    """计算Hessian向量乘积 Hv，使用高精度"""
    hv = torch.autograd.grad(gradsH, params, grad_outputs=v, 
                            only_inputs=True, retain_graph=not stop_criterion)
    return hv


class ImprovedHessianWeightImportance:
    """
    改进的Hessian权重重要性计算器
    专注于全连接层的Hessian分析
    """
    
    def __init__(self, model, device='cuda', n_samples=500, use_double_precision=True):
        self.model = model
        self.device = device
        self.n_samples = n_samples
        self.use_double_precision = use_double_precision
        
        # 存储结果
        self.layer_params = []
        self.hessian_traces = {}
        self.weight_importance = {}
        
        # 准备模型参数
        self._prepare_model()
    
    def _prepare_model(self):
        """准备全连接层参数"""
        print("准备全连接层参数...")
        
        # 处理全连接层
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # 保存参数信息
                self.layer_params.append((name, module.weight))
                print(f"注册全连接层: {name} (输入={module.in_features}, 输出={module.out_features})")
    
    def compute_hessian_trace_hutchinson(self, data_loader, criterion):
        """使用改进的Hutchinson方法计算Hessian trace"""
        print(f"使用改进的Hutchinson方法 (n_samples={self.n_samples}, 高精度={self.use_double_precision})...")
        
        # 设置为训练模式
        self.model.train()
        
        # 确保所有参数可训练
        for param in self.model.parameters():
            param.requires_grad = True
        
        # 如果使用双精度，转换模型
        if self.use_double_precision:
            print("转换为双精度模式...")
            self.model = self.model.double()
        
        # 获取多个batch提高稳定性
        all_traces = {}
        for name, _ in self.layer_params:
            all_traces[name] = []
        
        batch_count = 0
        for batch_idx, (data, target) in enumerate(data_loader):
            if batch_idx >= 3:  # 使用前3个batch
                break
                
            batch_count += 1
            data, target = data.to(self.device), target.to(self.device)
            
            if self.use_double_precision:
                data = data.double()
            
            print(f"\n处理第 {batch_idx + 1} 个batch...")
            
            # 前向传播
            self.model.zero_grad()
            output = self.model(data)
            
            # 处理SNN输出
            if len(output.shape) > 2:
                output = output.mean(0)
            
            # 计算损失和梯度
            loss = criterion(output, target)
            loss.backward(create_graph=True)
            
            # 获取参数和梯度
            params = []
            gradsH = []
            valid_layers = []
            
            for name, param in self.layer_params:
                if param.grad is not None:
                    params.append(param)
                    gradsH.append(param.grad)
                    valid_layers.append((name, param))
                    # print(f"层 {name} 的梯度范数: {param.grad.norm().item():.8f}")
                else:
                    print(f"警告: 层 {name} 没有梯度")
            
            # Hutchinson采样 - 增加采样数
            batch_traces = {}
            for name, _ in valid_layers:
                batch_traces[name] = []
            
            print(f"开始Hutchinson采样 (批次 {batch_idx + 1})...")
            
            for sample_idx in range(self.n_samples):
                # 使用不同的随机分布提高采样质量
                if sample_idx % 2 == 0:
                    # Rademacher分布
                    v = [torch.randint_like(p, high=2, device=self.device).float() * 2 - 1 
                         for p in params]
                else:
                    # 标准正态分布
                    v = [torch.randn_like(p, device=self.device) for p in params]
                
                if self.use_double_precision:
                    v = [vi.double() for vi in v]
                
                try:
                    # 计算Hessian-向量乘积
                    Hv = hessian_vector_product(gradsH, params, v, 
                                              stop_criterion=(sample_idx == self.n_samples - 1))
                    
                    # 计算trace并保持高精度
                    for i, (name, param) in enumerate(valid_layers):
                        if i >= len(Hv):
                            continue
                        
                        # 直接计算 v^T * H * v
                        trace_val = torch.sum(v[i] * Hv[i]).item()
                        batch_traces[name].append(trace_val)
                    
                    # 清理中间变量
                    del Hv
                    del v
                    
                except Exception as e:
                    print(f"  样本 {sample_idx} 计算失败: {e}")
                    continue
                
                # if (sample_idx + 1) % 100 == 0:
                #     print(f"  完成 {sample_idx + 1}/{self.n_samples} 次采样")
            
            # 收集这个batch的结果
            for name in batch_traces:
                if len(batch_traces[name]) > 0:
                    avg_trace = np.mean(batch_traces[name])
                    all_traces[name].append(avg_trace)
                    # print(f"  批次 {batch_idx + 1} - 层 {name}: 平均trace = {avg_trace:.8f}")
        
        # 计算所有batch的最终平均trace
        print(f"\n计算 {batch_count} 个batch的最终平均...")
        for name in all_traces:
            if len(all_traces[name]) > 0:
                final_trace = np.mean(all_traces[name])
                trace_std = np.std(all_traces[name])
                self.hessian_traces[name] = [final_trace]
                # print(f"层 {name}: 最终trace = {final_trace:.8f} ± {trace_std:.8f}")
            else:
                self.hessian_traces[name] = [0.0]
                print(f"层 {name}: 无有效trace，设为0")
    
    def collect_neuron_importance(self):
        """
        收集每个神经元的重要性数据
        
        返回:
        neuron_importance_list - 包含每个神经元重要性信息的列表
        """
        print("\n收集神经元重要性数据...")
        neuron_importance_list = []
        
        for name, param in self.layer_params:
            if name not in self.hessian_traces:
                print(f"警告: {name} 没有Hessian trace")
                continue
            
            # 检查梯度
            if param.grad is None:
                print(f"警告: {name} 没有梯度")
                continue
                
            # 检查Hessian trace
            hessian_trace = self.hessian_traces[name][0]
            if abs(hessian_trace) < 1e-10:
                print(f"警告: {name} 的Hessian trace接近0")
            
            weight_norm_sq = param.norm(p=2) ** 2
            param_count = param.numel()
            
            # 计算每个神经元的重要性
            if param_count > 0:
                base_importance = hessian_trace * weight_norm_sq.item() / param_count
                
                # 对每个神经元计算重要性
                for neuron_idx in range(param.size(0)):  # 遍历输出神经元
                    # 获取该神经元的权重范数
                    neuron_weight_norm = param[neuron_idx].norm(p=2) ** 2
                    # 计算该神经元的重要性
                    neuron_importance = base_importance * neuron_weight_norm.item()
                    
                    # 添加到列表
                    neuron_importance_list.append({
                        'layer': name,
                        'neuron_id': neuron_idx,
                        'importance': neuron_importance
                    })
                    
                    # if neuron_idx < 5:  # 只打印前5个神经元的信息作为示例
                    #     print(f"层 {name} 神经元 {neuron_idx}: 重要性 = {neuron_importance:.8f}")
        
        # 按重要性排序
        neuron_importance_list.sort(key=lambda x: x['importance'], reverse=True)
        
        print(f"\n总共收集了 {len(neuron_importance_list)} 个神经元的重要性数据")
        return neuron_importance_list
    
    def get_pruning_candidates(self, neuron_importance_list, pruning_ratio=0.5):
        """
        获取剪枝候选，基于神经元重要性排序
        
        参数:
        neuron_importance_list - 神经元重要性列表
        pruning_ratio - 要剪枝的神经元比例
        
        返回:
        pruning_candidates - 剪枝候选列表
        """
        print(f"\n生成剪枝候选 (剪枝比例: {pruning_ratio})...")
        
        # 按重要性排序（从小到大）
        sorted_neurons = sorted(neuron_importance_list, key=lambda x: x['importance'])
        
        # 选择要剪枝的神经元
        num_to_prune = int(len(sorted_neurons) * pruning_ratio)
        pruning_candidates = sorted_neurons[:num_to_prune]
        
        # 按层分组统计
        layer_counts = {}
        for neuron in pruning_candidates:
            layer = neuron['layer']
            if layer not in layer_counts:
                layer_counts[layer] = 0
            layer_counts[layer] += 1
        
        print(f"选择了 {len(pruning_candidates)} 个神经元进行剪枝:")
        for layer, count in layer_counts.items():
            print(f"  {layer}: {count} 个神经元")
        
        return pruning_candidates

    def run_full_analysis(self, data_loader, criterion, pruning_ratio):
        """运行完整分析"""
        print("="*80)
        print("🚀 全连接层Hessian重要性分析")
        print("="*80)
        
        start_time = time.time()
        
        try:
            # 计算Hessian trace
            print("\n1. 计算Hessian trace...")
            self.compute_hessian_trace_hutchinson(data_loader, criterion)
            
            # 收集神经元重要性数据
            print("\n2. 收集神经元重要性数据...")
            neuron_importance_list = self.collect_neuron_importance()
            
            # 生成剪枝候选
            print("\n3. 生成剪枝候选...")
            pruning_candidates = self.get_pruning_candidates(neuron_importance_list,pruning_ratio)
            
            total_time = time.time() - start_time
            print(f"\n✅ 分析完成！总耗时: {total_time:.2f}秒")
            print("="*80)
            
            return {
                'neuron_importance_list': neuron_importance_list,
                'hessian_traces': self.hessian_traces,
                'pruning_candidates': pruning_candidates
            }
        finally:
            # 清理
            if self.use_double_precision:
                self.model = self.model.float()


def evaluate_model(model, test_loader, criterion, device, seed=None):
    """评估模型性能"""
    if seed is not None:
        seed_all(seed)
        
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(images)
            
            # 处理SNN输出
            if len(outputs.shape) > 2:
                outputs = outputs.mean(0)
            
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
    print(f"\n评估完成:")
    print(f"  总样本数: {total_samples}")
    print(f"  平均准确率: {avg_accuracy:.2f}% ({total_correct}/{total_samples})")
    print(f"  平均损失: {avg_loss:.6f}")
    
    return avg_accuracy, avg_loss

def prune_neurons(model, pruning_candidates):
    """执行神经元剪枝"""
    print("\n执行神经元剪枝...")
    for neuron_info in pruning_candidates:
        layer_name = neuron_info['layer']
        neuron_idx = neuron_info['neuron_id']
        
        # 找到对应的层
        module = None
        for name, mod in model.named_modules():
            if name == layer_name and isinstance(mod, nn.Linear):
                module = mod
                break
        
        if module:
            # 执行剪枝：将神经元的权重置零
            with torch.no_grad():
                module.weight.data[neuron_idx] = 0
                if module.bias is not None:
                    module.bias.data[neuron_idx] = 0

def save_results_to_csv(neuron_importance_list, timestamp):
    """保存神经元重要性数据到CSV文件"""
    print("\n保存神经元重要性数据...")
    
    # 按层分组
    layer_data = {}
    for neuron in neuron_importance_list:
        layer = neuron['layer']
        if layer not in layer_data:
            layer_data[layer] = []
        layer_data[layer].append({
            'neuron_id': neuron['neuron_id'],
            'importance': neuron['importance']
        })
    
    # 为每层创建CSV文件
    for layer_name, neurons in layer_data.items():
        # 创建DataFrame
        df = pd.DataFrame(neurons)
        
        # 生成文件名
        filename = f"importance_analysis_{layer_name}_{timestamp}.csv"
        
        # 保存到CSV
        df.to_csv(filename, index=False)
        print(f"已保存{layer_name}层的重要性信息到: {filename}")
        print(f"  神经元数量: {len(neurons)}")
        print(f"  每个神经元包含1个重要性值")

def main():
    """主函数"""
    # 添加命令行参数
    parser = argparse.ArgumentParser(description='神经元重要性分析与剪枝')
    parser.add_argument('--batch_size', default=200, type=int, help='批次大小')
    parser.add_argument('--device', default='0', type=str, help='设备')
    parser.add_argument('--seed', default=42, type=int, help='随机种子')
    parser.add_argument('--mode', choices=['ann', 'snn'], default='snn', help='模式')
    parser.add_argument('--n_samples', default=200, type=int, help='Hessian采样数量')
    parser.add_argument('-r','--pruning_ratio', default=0.5, type=float, help='剪枝比例')
    parser.add_argument('--dataset', choices=['cifar10', 'cifar100'], default='cifar100', help='数据集')
    parser.add_argument('--use_double_precision', action='store_true', help='是否使用双精度计算')
    
    args = parser.parse_args()
    
    # 设置输出重定向（默认保存到文件）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"./log/importance_analysis_{args.mode}_{timestamp}.txt"
    output_redirector = OutputRedirector(filename)
    sys.stdout = output_redirector
    print(f"输出将保存到文件: {filename}")
    
    # 设置环境
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(args.seed)
    
    print(f"设备: {device}, 随机种子: {args.seed}")
    print(f"分析模式: {args.mode}")
    print(f"Hessian采样数量: {args.n_samples}")
    print(f"剪枝比例: {args.pruning_ratio}")
    print(f"数据集: {args.dataset}")
    print(f"使用双精度: {args.use_double_precision}")
    
    # 创建模型
    print("创建VGG16模型...")
    model = modelpool('vgg16', args.dataset)
    

    
    # 加载预训练模型
    if args.dataset == 'cifar10':
        model_path = '/root/autodl-tmp/0-ANN2SNN-Allinone/2-ANN_SNN_QCFS-SRP/cifar10-checkpoints/vgg16_wd[0.0005].pth'
    else:
        model_path = '/root/autodl-tmp/0-ANN2SNN-Allinone/2-ANN_SNN_QCFS-SRP/cifar100-checkpoints/vgg16_L[4].pth'
    
    print(f"加载预训练模型: {model_path}")
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    print("✅ 预训练模型加载成功")

    # 设置模型模式
    if args.mode == 'snn':
        model.set_T(8)
        model.set_L(4)
        print("设置为SNN模式")
    else:
        model.set_T(0)
        model.set_L(4)
        print("设置为ANN模式")


    model.to(device)


    
    # 加载数据
    print(f"加载{args.dataset}数据集...")
    train_loader, test_loader = datapool(args.dataset, args.batch_size)

    criterion = nn.CrossEntropyLoss()

    # 剪枝前评估
    print("\n剪枝前评估:")
    pre_accuracy, pre_loss = evaluate_model(model, test_loader, criterion, device, args.seed)
        
    # 创建改进的计算器
    hessian_calc = ImprovedHessianWeightImportance(
        model=model,
        device=device,
        n_samples=args.n_samples,
        use_double_precision=args.use_double_precision
    )
    
    # 运行分析

    results = hessian_calc.run_full_analysis(train_loader, criterion,args.pruning_ratio)
    
    # 获取剪枝候选
    pruning_candidates = results['pruning_candidates']
    
    # 按层分组统计
    layer_counts = {}
    for neuron in pruning_candidates:
        layer = neuron['layer']
        if layer not in layer_counts:
            layer_counts[layer] = 0
        layer_counts[layer] += 1
    
    print("\n建议剪枝的神经元统计:")
    for layer, count in layer_counts.items():
        print(f"  {layer}: {count} 个神经元")
    

    # 执行剪枝
    prune_neurons(model, pruning_candidates)
    
    # 剪枝后评估
    print("\n剪枝后评估:")
    post_accuracy, post_loss = evaluate_model(model, test_loader, criterion, device, args.seed)
    
    # # 保存结果
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # save_results_to_csv(results['neuron_importance_list'], timestamp)
    
    # # 保存评估结果
    # eval_results = {
    #     'pre_pruning': {
    #         'accuracy': pre_accuracy,
    #         'loss': pre_loss
    #     },
    #     'post_pruning': {
    #         'accuracy': post_accuracy,
    #         'loss': post_loss
    #     }
    # }
    
    # 保存到文件
    # torch.save(eval_results, f'evaluation_results_{timestamp}.pt')
    # print(f"\n📁 评估结果已保存到 evaluation_results_{timestamp}.pt")
    
    return results


if __name__ == "__main__":
    results = main() 