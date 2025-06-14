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
        
        
        
class SNNGradientAnalyzer:
    def __init__(self, model, prune_ratio=0.1):
        self.model = model
        self.prune_ratio = prune_ratio
        self.gradient_records = defaultdict(list)  # 存储全连接层梯度
        
        # 只为全连接层注册梯度钩子
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):  # 仅处理Linear层
                # 获取全连接层的权重参数
                weight_param = module.weight
                # 为每个全连接层创建唯一的标识符
                layer_id = f"fc_{name}"  # 使用层名作为唯一标识
                
                # 创建钩子函数
                hook = self.make_hook(layer_id)
                weight_param.register_hook(hook)
    
    def make_hook(self, layer_id):
        def gradient_hook(grad):
            # 计算每个输出神经元的平均梯度
            # 梯度形状: [out_features, in_features]
            # 计算方式: 对每个输出神经元求其在输入特征上的梯度均值
            grad_mag = grad.abs().mean(dim=1)  # 输出形状: [out_features]
            self.gradient_records[layer_id].append(grad_mag.detach().cpu())
        return gradient_hook
    
    def get_low_grad_neurons(self):
        all_neurons = []  # 存储要剪枝的神经元信息
        
        # 遍历记录的所有全连接层
        for layer_id, grad_list in self.gradient_records.items():
            if not grad_list:  # 跳过空记录
                continue
                
            # 合并所有梯度记录
            stacked_grads = torch.stack(grad_list)  # [timesteps, batch, neurons]
            
            # 计算时间维度和批次维度的平均梯度
            # 结果为每个神经元的平均梯度幅度
            neuron_avg_grad = stacked_grads.mean(dim=0).mean(dim=0)  # [neurons]
            
            # 收集该层的神经元梯度信息
            for neuron_idx, grad_value in enumerate(neuron_avg_grad):
                # 格式: (层标识符, 神经元索引, 梯度值)
                all_neurons.append((layer_id, neuron_idx, grad_value.item()))
        
        # 如果没有记录任何神经元，返回空列表
        if not all_neurons:
            return []
        
        # 按梯度值从小到大排序
        all_neurons.sort(key=lambda x: x[2])
        
        # 选择梯度最小的前 prune_ratio 比例的神经元
        num_to_select = int(len(all_neurons) * self.prune_ratio)
        selected_neurons = all_neurons[:num_to_select]
        
        return selected_neurons