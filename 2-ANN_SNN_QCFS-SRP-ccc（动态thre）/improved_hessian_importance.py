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
from Models.layer import IF


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
    解决数值精度和采样不足问题
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
        """准备模型参数，只选择IF层的参数"""
        print("准备IF层参数...")
        
        for name, module in self.model.named_modules():
            if isinstance(module, IF):
                for param_name, param in module.named_parameters():
                    if param.requires_grad:
                        full_name = f"{name}.{param_name}"
                        self.layer_params.append((full_name, param))
                        print(f"注册IF层参数: {full_name} - {param.shape}")
    
    def compute_hessian_trace_hutchinson(self, data_loader, criterion):
        """使用改进的Hutchinson方法计算Hessian trace"""
        print(f"使用改进的Hutchinson方法 (n_samples={self.n_samples}, 高精度={self.use_double_precision})...")
        
        self.model.eval()
        
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
                
                if (sample_idx + 1) % 100 == 0:
                    print(f"  完成 {sample_idx + 1}/{self.n_samples} 次采样")
            
            # 收集这个batch的结果
            for name in batch_traces:
                if len(batch_traces[name]) > 0:
                    avg_trace = np.mean(batch_traces[name])
                    all_traces[name].append(avg_trace)
                    print(f"  批次 {batch_idx + 1} - 层 {name}: 平均trace = {avg_trace:.8f}")
        
        # 计算所有batch的最终平均trace
        print(f"\n计算 {batch_count} 个batch的最终平均...")
        for name in all_traces:
            if len(all_traces[name]) > 0:
                final_trace = np.mean(all_traces[name])
                trace_std = np.std(all_traces[name])
                self.hessian_traces[name] = [final_trace]  # IF层只有一个参数
                print(f"层 {name}: 最终trace = {final_trace:.8f} ± {trace_std:.8f}")
            else:
                self.hessian_traces[name] = [0.0]
                print(f"层 {name}: 无有效trace，设为0")
    
    def compute_weight_importance(self):
        """计算IF神经元的重要性"""
        print("\n计算IF神经元重要性...")
        
        for name, param in self.layer_params:
            if name not in self.hessian_traces:
                continue
            
            hessian_trace = self.hessian_traces[name][0]
            
            # 改进的重要性计算公式
            weight_norm_sq = param.norm(p=2) ** 2
            param_count = param.numel()
            
            # 使用归一化的重要性计算
            if param_count > 0:
                importance = hessian_trace * weight_norm_sq.item() / param_count
            else:
                importance = 0.0
            
            self.weight_importance[name] = [importance]
            
            print(f"IF层 {name}:")
            print(f"  Hessian trace: {hessian_trace:.8f}")
            print(f"  权重范数²: {weight_norm_sq:.6f}")
            print(f"  参数数量: {param_count}")
            print(f"  重要性: {importance:.8f}")
    
    def analyze_importance_distribution(self):
        """分析重要性分布"""
        print("\n📊 重要性分布分析:")
        print("-" * 60)
        
        all_importances = []
        valid_importances = []
        
        for name, importance_list in self.weight_importance.items():
            importance = importance_list[0]
            all_importances.append(importance)
            
            print(f"{name}: {importance:.8f}")
            
            if abs(importance) > 1e-10:  # 非零重要性
                valid_importances.append(importance)
        
        print(f"\n统计结果:")
        print(f"总IF层数: {len(all_importances)}")
        print(f"非零重要性层数: {len(valid_importances)}")
        print(f"非零比例: {len(valid_importances)/len(all_importances)*100:.2f}%")
        
        if len(valid_importances) > 0:
            print(f"非零重要性统计:")
            print(f"  均值: {np.mean(valid_importances):.8f}")
            print(f"  标准差: {np.std(valid_importances):.8f}")
            print(f"  范围: [{np.min(valid_importances):.8f}, {np.max(valid_importances):.8f}]")
        else:
            print("⚠️  所有重要性值仍为0或接近0")
            
        return all_importances, valid_importances
    
    def get_pruning_candidates(self, pruning_ratio=0.3):
        """获取剪枝候选，基于重要性排序"""
        print(f"\n生成剪枝候选 (剪枝比例: {pruning_ratio})...")
        
        # 收集所有重要性
        layer_importances = []
        for name, importance_list in self.weight_importance.items():
            layer_importances.append((name, importance_list[0]))
        
        # 按重要性排序（从小到大）
        layer_importances.sort(key=lambda x: x[1])
        
        # 选择要剪枝的层
        num_to_prune = int(len(layer_importances) * pruning_ratio)
        pruning_candidates = layer_importances[:num_to_prune]
        
        print(f"选择了 {len(pruning_candidates)} 个IF层进行剪枝:")
        for name, importance in pruning_candidates:
            print(f"  {name}: {importance:.8f}")
        
        return pruning_candidates
    
    def run_full_analysis(self, data_loader, criterion):
        """运行完整分析"""
        print("="*80)
        print("🚀 改进的Hessian权重重要性分析")
        print("="*80)
        
        start_time = time.time()
        
        # 计算Hessian trace
        self.compute_hessian_trace_hutchinson(data_loader, criterion)
        
        # 计算重要性
        self.compute_weight_importance()
        
        # 分析分布
        all_imp, valid_imp = self.analyze_importance_distribution()
        
        # 生成剪枝候选
        pruning_candidates = self.get_pruning_candidates(0.3)
        
        total_time = time.time() - start_time
        print(f"\n✅ 分析完成！总耗时: {total_time:.2f}秒")
        print("="*80)
        
        return {
            'hessian_traces': self.hessian_traces,
            'weight_importance': self.weight_importance,
            'all_importances': all_imp,
            'valid_importances': valid_imp,
            'pruning_candidates': pruning_candidates
        }


# 测试脚本
if __name__ == "__main__":
    print("测试改进的Hessian重要性计算器...")
    
    from Models import modelpool
    from Preprocess import datapool
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建模型
    model = modelpool('vgg16', 'cifar10')
    model.set_L(8)
    model.set_T(0)  # ANN模式
    model.to(device)
    
    # 加载数据
    train_loader, _ = datapool('cifar10', 16)
    
    # 创建改进的计算器
    hessian_calc = ImprovedHessianWeightImportance(
        model=model,
        device=device,
        n_samples=200,  # 增加采样数
        use_double_precision=True  # 使用双精度
    )
    
    # 运行分析
    criterion = nn.CrossEntropyLoss()
    results = hessian_calc.run_full_analysis(train_loader, criterion)
    
    print("\n🎯 分析结果总结:")
    print(f"有效重要性层数: {len(results['valid_importances'])}")
    if len(results['valid_importances']) > 0:
        print("✅ 成功计算出非零重要性值！")
    else:
        print("❌ 重要性值仍为0，需要进一步调试") 