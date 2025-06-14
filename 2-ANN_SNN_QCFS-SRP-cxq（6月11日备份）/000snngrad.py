
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

class SNNGradientAnalyzer:
    """SNN梯度分析工具：模拟梯度反向传播与梯度统计"""
    def __init__(self, model, surrogate_grad_type='sigmoid', grad_scale=5.0):
        """
        初始化SNN梯度分析器
        
        参数:
        model - 已训练好的SNN模型
        surrogate_grad_type - 模拟梯度类型: 'sigmoid', 'arctan', 'triangle'
        grad_scale - 模拟梯度缩放因子
        """
        self.model = model
        self.surrogate_grad_type = surrogate_grad_type
        self.grad_scale = grad_scale
        self.gradient_hooks = {}
        self.gradient_records = defaultdict(list)
        self.forward_cache = {}
        
        # 为所有激活层(IF神经元)注册前向钩子
        self.register_activation_hooks()
    
    def apply_surrogate_grad(self, x):
        """应用模拟梯度函数"""
        if self.surrogate_grad_type == 'sigmoid':
            sg = torch.sigmoid(self.grad_scale * x)
            return sg * (1 - sg)
        elif self.surrogate_grad_type == 'arctan':
            return 1 / (1 + (self.grad_scale * np.pi * x)**2)
        elif self.surrogate_grad_type == 'triangle':
            return torch.clamp(1 - self.grad_scale * torch.abs(x), min=0)
        else:
            # 默认使用sigmoid
            sg = torch.sigmoid(self.grad_scale * x)
            return sg * (1 - sg)
    
    def register_activation_hooks(self):
        """为所有激活层注册前向钩子"""
        for name, module in self.model.named_modules():
            # 识别SNN中的激活层（IF神经元）
            if 'IF' in module.__class__.__name__ or 'LIF' in module.__class__.__name__:
                # 注册前向钩子记录激活值
                hook = self._forward_hook(name)
                handle = module.register_forward_hook(hook)
                self.gradient_hooks[name] = handle
    
    def _forward_hook(self, name):
        """创建前向钩子函数"""
        def hook(module, input, output):
            # 记录激活值(模拟梯度计算依据)
            self.forward_cache[name] = output.detach().clone()
        return hook
    
    def backward_with_surrogate(self, output, target, criterion):
        """使用模拟梯度进行反向传播"""
        # 清空前向缓存
        self.forward_cache = {}
        
        # 前向传播（激活钩子将被触发）
        pred = output if output.dim() <= 2 else output.mean(2)  # 处理时序输出
        loss = criterion(pred, target)
        
        # 获取梯度函数
        grad_fn = loss.grad_fn
        
        # 修改梯度函数链
        def custom_grad(grad_output):
            # 原始梯度
            orig_grad = grad_output.clone()
            
            # 在梯度回传过程中应用模拟梯度
            for name, module in self.model.named_modules():
                # 只在激活层应用模拟梯度
                if name in self.forward_cache:
                    # 获取原始梯度
                    if orig_grad.grad_fn.next_functions:
                        # 获取对应层的梯度
                        grad_input = orig_grad.grad_fn.next_functions[0][0]
                        
                        # 应用模拟梯度
                        cached_value = self.forward_cache[name]
                        surrogate_grad = self.apply_surrogate_grad(cached_value)
                        
                        # 修改梯度
                        modified_grad = grad_input * surrogate_grad
                        
                        # 替换为修改后的梯度
                        grad_input.copy_(modified_grad)
            
            return orig_grad
        
        # 将自定义梯度函数附加到原始梯度上
        grad_fn.register_hook(custom_grad)
        
        # 执行反向传播（将触发自定义梯度计算）
        loss.backward()
        
        return loss
    
    def register_fc_gradient_hooks(self):
        """为所有全连接层注册梯度钩子"""
        # 清除现有钩子
        for handle in self.gradient_hooks.values():
            handle.remove()
        self.gradient_hooks = {}
        self.gradient_records.clear()
        
        # 为所有全连接层注册梯度钩子
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # 注册梯度钩子
                hook = self._gradient_hook(name)
                handle = module.weight.register_hook(hook)
                self.gradient_hooks[name] = handle
    
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
            self.gradient_records[name].append(neuron_grads.detach().cpu().numpy())
        return hook
    
    def analyze_fc_gradients(self, dataloader, criterion):
        """
        分析全连接层梯度
        
        参数:
        dataloader - 数据加载器
        criterion - 损失函数
        
        返回:
        layer_grads - 每层的平均梯度统计
        """
        # 确保梯度钩子已注册
        self.register_fc_gradient_hooks()
        
        # 清除梯度记录
        self.gradient_records = defaultdict(list)
        
        # 配置模型为评估模式但保留梯度
        self.model.eval()
        
        # 处理整个数据集
        with torch.set_grad_enabled(True):
            for inputs, targets in dataloader:
                # 清空梯度
                self.model.zero_grad()
                
                # 前向传播
                outputs = self.model(inputs)
                
                # 使用模拟梯度进行反向传播
                loss = self.backward_with_surrogate(outputs, targets, criterion)
                
                # 打印进度
                if hasattr(dataloader, '__len__'):
                    progress = len(self.gradient_records) / len(dataloader) * 100
                    print(f"Processing: {progress:.1f}%", end='\r')
        
        # 计算平均梯度
        layer_grads = {}
        for name, grad_list in self.gradient_records.items():
            if grad_list:
                # 合并所有批次的梯度
                all_grads = np.concatenate(grad_list, axis=0)
                
                # 计算每个神经元的平均梯度
                neuron_avg_grads = np.mean(all_grads, axis=0)
                
                # 存储结果
                layer_grads[name] = {
                    'mean_grad': np.mean(neuron_avg_grads),
                    'min_grad': np.min(neuron_avg_grads),
                    'max_grad': np.max(neuron_avg_grads),
                    'neuron_grads': neuron_avg_grads,
                    'num_neurons': len(neuron_avg_grads)
                }
        
        return layer_grads
    
    def visualize_gradients(self, layer_grads):
        """可视化梯度分布"""
        plt.figure(figsize=(15, 10))
        
        # 为每层创建子图
        for i, (layer_name, grad_info) in enumerate(layer_grads.items()):
            plt.subplot(3, 3, i+1)
            
            # 获取神经元梯度
            neuron_grads = grad_info['neuron_grads']
            
            # 绘制直方图
            plt.hist(neuron_grads, bins=30, alpha=0.7)
            
            # 添加统计信息
            plt.title(f"{layer_name}\nMean: {grad_info['mean_grad']:.4f}")
            plt.xlabel("Gradient Magnitude")
            plt.ylabel("Number of Neurons")
            
            # 最多显示9层
            if i+1 >= 9:
                break
        
        plt.tight_layout()
        plt.savefig("snn_gradient_distribution.png")
        plt.close()
        
        # 打印整体统计
        print("\n=== SNN Gradient Analysis Summary ===")
        for layer_name, grad_info in layer_grads.items():
            print(f"Layer: {layer_name}")
            print(f"  Neurons: {grad_info['num_neurons']}")
            print(f"  Mean Gradient: {grad_info['mean_grad']:.6f}")
            print(f"  Min Gradient: {grad_info['min_grad']:.6f}")
            print(f"  Max Gradient: {grad_info['max_grad']:.6f}")
            print("-" * 40)
        
        return layer_grads