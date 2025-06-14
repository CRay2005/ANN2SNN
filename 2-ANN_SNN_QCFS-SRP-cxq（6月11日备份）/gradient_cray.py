import torch
import torch.nn as nn
import torch.nn.functional as F

class IFNeuron(nn.Module):
    """具有内置反向模拟梯度的脉冲神经元"""
    def __init__(self, threshold=1.0, surrogate_grad='sigmoid', scale=5.0, reset='soft'):
        """
        初始化IF神经元
        
        参数:
        threshold - 脉冲发放阈值
        surrogate_grad - 模拟梯度类型: 'sigmoid', 'arctan', 'triangle'
        scale - 模拟梯度缩放因子
        reset - 重置模式: 'soft' (漏电), 'hard' (归零)
        """
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(threshold))
        self.scale = scale
        self.surrogate_grad = surrogate_grad
        self.reset_mode = reset
        
        # 神经元状态
        self.mem_pot = None
        self.spike = None
        self.prev_spike = None  # 用于记录上次脉冲状态
        
        # 初始化状态
        self.reset_state()
    
    def reset_state(self):
        """重置神经元状态"""
        self.mem_pot = torch.tensor(0.0)
        self.spike = torch.tensor(0.0)
        self.prev_spike = torch.tensor(0.0)
    
    def forward(self, x):
        """前向传播：膜电位更新与脉冲发放"""
        # 更新膜电位（考虑漏电或重置）
        if self.reset_mode == 'soft':
            # 软重置：根据上次发放状态衰减膜电位
            self.mem_pot = self.mem_pot * (1 - self.prev_spike) + x
        else:
            # 硬重置：脉冲后归零
            self.mem_pot = self.mem_pot + x
            
        # 保存上次脉冲状态用于下次更新
        self.prev_spike = self.spike.clone()
        
        # 计算脉冲（不可导）
        spike = (self.mem_pot >= self.threshold).float()
        self.spike = spike.detach()  # 阻断原始梯度
        
        # === 反向传播模拟梯度关键部分 ===
        # 计算与阈值的差值作为模拟梯度依据
        delta = self.mem_pot - self.threshold
        
        # 创建可导的脉冲输出（使用前向值但自定义后向梯度）
        # 这是PyTorch自定义反向传播的标准技巧
        spike_out = spike + delta - delta.detach()
        
        # 脉冲后重置膜电位（硬重置）
        if self.reset_mode == 'hard':
            self.mem_pot = self.mem_pot * (1 - spike)
        
        return spike_out
    
    @staticmethod
    def backward(ctx, grad_output):
        """自定义反向传播：应用模拟梯度"""
        # 获取前向传播保存的值
        delta = ctx.saved_tensors[0]
        scale = ctx.scale
        surrogate_grad = ctx.surrogate_grad
        
        # 计算模拟梯度
        if surrogate_grad == 'sigmoid':
            sg = torch.sigmoid(scale * delta)
            surrogate_grad = sg * (1 - sg) * scale
        elif surrogate_grad == 'arctan':
            surrogate_grad = scale / (1 + (scale * np.pi * delta) ** 2)
        elif surrogate_grad == 'triangle':
            surrogate_grad = torch.clamp(1 - scale * torch.abs(delta), min=0)
        else:  # 默认使用sigmoid
            sg = torch.sigmoid(scale * delta)
            surrogate_grad = sg * (1 - sg) * scale
        
        # 应用模拟梯度到上游梯度
        modified_grad = grad_output * surrogate_grad
        
        return modified_grad, None, None, None, None
    
    def extra_repr(self):
        """返回模块的额外信息表示"""
        return (f"threshold={self.threshold.item()}, "
                f"grad='{self.surrogate_grad}', "
                f"scale={self.scale}, reset={self.reset_mode}")
        
        

class SNNWithCustomNeurons(nn.Module):
    """使用内置模拟梯度神经元的SNN模型"""
    def __init__(self, input_size, hidden_size, output_size, T=8, reset_mode='soft'):
        super().__init__()
        self.T = T  # 时间步长
        self.reset_mode = reset_mode
        
        # 网络层定义
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = IFNeuron(threshold=1.0, surrogate_grad='sigmoid', reset=reset_mode)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.lif2 = IFNeuron(threshold=1.0, surrogate_grad='arctan', reset=reset_mode)
        
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.lif3 = IFNeuron(threshold=1.0, surrogate_grad='triangle', reset=reset_mode)
        
        # 梯度记录器
        self.gradient_hooks = {}
        self.gradient_records = {}
    
    def set_T(self, T):
        """设置时间步长"""
        self.T = T
    
    def reset_states(self):
        """重置所有神经元状态"""
        self.lif1.reset_state()
        self.lif2.reset_state()
        self.lif3.reset_state()
    
    def register_gradient_hooks(self):
        """为所有全连接层注册梯度记录钩子"""
        # 移除现有钩子
        for handle in self.gradient_hooks.values():
            handle.remove()
        self.gradient_hooks = {}
        self.gradient_records = {}
        
        # 为所有全连接层注册钩子
        for name, module in [('fc1', self.fc1), ('fc2', self.fc2), ('fc3', self.fc3)]:
            # 为权重参数注册梯度钩子
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
            self.gradient_records[name] = neuron_grads.detach().cpu()
        return hook
    
    def forward(self, x):
        """前向传播处理时域信息"""
        # 初始化输出
        outputs = []
        
        # 重置神经元状态
        self.reset_states()
        
        # 时间步展开
        for t in range(self.T):
            # 第一层
            x_t = x[:, :, t] if self.T > 1 else x
            h1 = self.fc1(x_t)
            s1 = self.lif1(h1)
            
            # 第二层
            h2 = self.fc2(s1)
            s2 = self.lif2(h2)
            
            # 输出层
            h3 = self.fc3(s2)
            out = self.lif3(h3)
            
            # 收集输出
            outputs.append(out)
        
        # 时间轴处理
        if self.T > 1:
            return torch.stack(outputs, dim=2)  # [B, C, T]
        else:
            return outputs[-1]  # [B, C]
    
    def analyze_gradients(self, dataloader, criterion, num_batches=10):
        """
        分析全连接层梯度分布
        
        参数:
        dataloader - 数据加载器
        criterion - 损失函数
        num_batches - 分析批次数
        
        返回:
        gradient_stats - 梯度统计信息
        """
        # 注册梯度钩子
        self.register_gradient_hooks()
        
        # 配置模型为评估模式但保留梯度
        self.train()
        
        # 梯度统计收集器
        gradient_stats = {}
        for name in ['fc1', 'fc2', 'fc3']:
            gradient_stats[name] = {'values': []}
        
        # 处理指定批次数据
        batch_count = 0
        for inputs, targets in dataloader:
            if batch_count >= num_batches:
                break
                
            # 清空梯度和状态
            self.zero_grad()
            self.reset_states()
            
            # 前向传播（触发梯度钩子）
            outputs = self(inputs)
            
            # 处理时序输出
            if self.T > 1:
                # 使用最后一个时间步的输出
                loss = criterion(outputs[:, :, -1], targets)
            else:
                loss = criterion(outputs, targets)
            
            # 反向传播（使用内置模拟梯度）
            loss.backward()
            
            # 收集梯度数据
            for name, grads in self.gradient_records.items():
                gradient_stats[name]['values'].extend(grads.numpy())
            
            batch_count += 1
        
        # 计算梯度统计
        for name, stats in gradient_stats.items():
            if stats['values']:
                values = np.array(stats['values'])
                stats['mean'] = np.mean(values)
                stats['std'] = np.std(values)
                stats['min'] = np.min(values)
                stats['max'] = np.max(values)
                stats['num_neurons'] = len(values)
        
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
    
    def prune_neurons(self, neurons_to_prune):
        """根据分析结果剪枝神经元"""
        for neuron in neurons_to_prune:
            layer_name = neuron['layer']
            idx = neuron['neuron_index']
            
            # 找到对应层
            layer = getattr(self, layer_name, None)
            if layer is None or not isinstance(layer, nn.Linear):
                continue
                
            # 执行剪枝（将对应神经元的权重置零）
            with torch.no_grad():
                # 剪枝输出权重
                layer.weight.data[idx] = 0
                
                # 如果有偏置项，剪枝偏置
                if layer.bias is not None:
                    layer.bias.data[idx] = 0
                
                print(f"Pruned neuron {idx} in layer {layer_name} (grad={neuron['grad_value']:.6f})")
        
        print(f"Total neurons pruned: {len(neurons_to_prune)}")