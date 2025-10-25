import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import random
import os
import logging
# 导入IF类用于类型检查
from Models.layer_copy import IF

def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

def train(model, device, train_loader, criterion, optimizer, T):
    running_loss = 0
    model.train()
    M = len(train_loader)
    total = 0
    correct = 0
    for i, (images, labels) in enumerate((train_loader)):
        optimizer.zero_grad()
        labels = labels.to(device)
        images = images.to(device)
        if T > 0:
            outputs = model(images).mean(0)
        else:
            outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        loss.mean().backward()
        optimizer.step()
        total += float(labels.size(0))
        _, predicted = outputs.cpu().max(1)
        correct += float(predicted.eq(labels.cpu()).sum().item())
    return running_loss, 100 * correct / total


def train_with_thre(model, device, train_loader, criterion, optimizer, T=None, scale_factor=1.0, layer_weights=None, adjust_interval=100, return_thre_grads=False, probe_num_batches=5, probe_loader=None, adjust_scale_factor=0.01):
    """
    简化的SNN阈值自适应训练函数 - 只调整全连接层后的IF层阈值
    
    参数说明:
    - T: 时间步数（当前版本未使用，保留用于兼容性）
    - scale_factor: 缩放因子（当前版本未使用，保留用于兼容性）
    - probe_num_batches: 探测批次数
    """
    running_loss = 0
    model.train()
    M = len(train_loader)
    total = 0
    correct = 0
    
    # 初始化thre_grads_history列表
    thre_grads_history = []
    
    # 如果没有提供layer_weights，创建默认的权重
    if layer_weights is None:
        # 根据模型类型确定IF层数量
        model_name = model.__class__.__name__.lower()
        if 'vgg' in model_name:
            layer_weights = torch.ones(2, device=device, requires_grad=True)  # VGG有2个全连接层后的IF层
        else:
            # 其他模型：自动计算IF层数量
            all_if_layers = [name for name, m in model.named_modules() if isinstance(m, IF)]
            layer_weights = torch.ones(len(all_if_layers), device=device, requires_grad=True)

    # 根据模型类型动态选择IF层名称
    model_name = model.__class__.__name__.lower()
    if 'vgg' in model_name:
        # VGG模型：选择全连接层后的IF层
        fc_if_layers = ['classifier.2', 'classifier.5']
        print(f"[INFO] 检测到VGG模型，选择IF层: {fc_if_layers}")
    elif 'resnet' in model_name:
        # ResNet模型：选择最后两个IF层
        fc_if_layers = ['conv4_x.2.act', 'conv3_x.2.act']
        print(f"[INFO] 检测到ResNet模型，选择IF层: {fc_if_layers}")
    else:
        # 其他模型：自动选择最后两个IF层
        all_if_layers = [name for name, m in model.named_modules() if isinstance(m, IF)]
        fc_if_layers = all_if_layers[-2:] if len(all_if_layers) >= 2 else all_if_layers
        print(f"[INFO] 检测到{model_name}模型，自动选择最后两个IF层: {fc_if_layers}")

    # 基于 named_modules() 仅为目标 IF 层构建有序列表与 name->index 映射
    
    targeted_if_layers = [(name, m) for name, m in model.named_modules() if isinstance(m, IF) and name in fc_if_layers]
    
    name_to_if_index = {name: idx for idx, (name, _m) in enumerate(targeted_if_layers)}

    for i, (images, labels) in enumerate((train_loader)):
        # ===== ANN 常规训练步 =====
        optimizer.zero_grad()
        labels = labels.to(device)
        images = images.to(device)

        # 强制以ANN模式运行（T=0）
        #original_T = getattr(model, 'T', 0)
        original_T = T
        model.set_T(0)

        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        loss.mean().backward()
        optimizer.step()
        total += float(labels.size(0))
        _, predicted = outputs.cpu().max(1)
        correct += float(predicted.eq(labels.cpu()).sum().item())

        # ===== 周期性：调整 SNN 阈值梯度（简化版） =====
        if adjust_interval > 0 and ((i + 1) % adjust_interval == 0):
            
            model.eval()  #===================================================            
            model.set_T(original_T)
            
            # 内存优化：清理梯度缓存   #===================================
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # 只对全连接层后的IF层进行梯度探测
            named_if_modules = []
            thresh_tensors = []
            for name, m in model.named_modules():
                if isinstance(m, IF) and name in fc_if_layers and hasattr(m, 'thresh') and m.thresh is not None:
                    named_if_modules.append((name, m))
                    thresh_tensors.append(m.thresh)

            # ===== 直接对当前batch进行阈值更新 =====
            with torch.no_grad():
                # 运行SNN模式获取最后时刻的mem
                if model.T is not None and model.T > 0:
                    # 重置所有IF层的spike_count_tensor，确保从0开始累积
                    for name, m in model.named_modules():
                        if isinstance(m, IF) and name in fc_if_layers:
                            if hasattr(m, 'spike_count_tensor') and m.spike_count_tensor is not None:
                                m.spike_count_tensor.zero_()
                    
                    # 运行SNN模式，让spike_count_tensor正确累积
                    _ = model(images)
                    
                    # 内存优化：清理SNN前向传播的中间结果   #====================
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
                    # 调试：检查SNN模式运行后mem的状态
                    print(f"[DEBUG] SNN模式运行后，检查mem状态:")
                    for name, m in model.named_modules():
                        if isinstance(m, IF) and name in fc_if_layers:
                            mem = getattr(m, 'mem', None)
                            spike_cnt = getattr(m, 'spike_count_tensor', None)
                            print(f"[DEBUG] {name}: mem类型={type(mem)}, spike_cnt类型={type(spike_cnt)}")
                            if isinstance(mem, torch.Tensor):
                                print(f"[DEBUG] {name}: mem形状={mem.shape}")
                            if isinstance(spike_cnt, torch.Tensor):
                                print(f"[DEBUG] {name}: spike_cnt形状={spike_cnt.shape}")
                
                # 计算每层的update_value并更新阈值
                layer_update_values = []
                
                for name, m in named_if_modules:
                    
                    update_value_scalar = 0.0  # 默认值
                    
                    if isinstance(m, IF) and name in fc_if_layers:
                        mem = getattr(m, 'mem', None)
                        spike_cnt = getattr(m, 'spike_count_tensor', None)
                        if isinstance(mem, torch.Tensor) and isinstance(spike_cnt, torch.Tensor):
                            try:
                                prod = mem * spike_cnt
                                mem_spike_avg = prod.mean(dim=0) if prod.dim() >= 1 else prod
                                
                                # 计算update_value并更新阈值
                                # 使用 prod * predict_spike 的平均（对所有元素），得到每层对应的标量
                                prod_predict = prod * m.predict_spike.detach().abs().to(prod.device)
                                update_value_scalar = prod_predict.mean().item()  # 对所有元素求平均，得到标量
                                mem_spike_avg_scalar = mem_spike_avg.mean().item() if mem_spike_avg.dim() > 0 else mem_spike_avg.item()
                                update_value_scalar = - mem_spike_avg_scalar + scale_factor * update_value_scalar
                                
                                print(f"[DEBUG] {name}: 成功计算，update_value_scalar={update_value_scalar:.6f}")
                                
                                if layer_weights is not None:
                                    idx = name_to_if_index.get(name, None)
                                    if idx is not None and idx < len(layer_weights):
                                        lw_scalar = layer_weights[idx].item() if torch.is_tensor(layer_weights[idx]) else float(layer_weights[idx])
                                        # 使用in-place操作更新阈值，保持梯度连接
                                        threshold_update = lw_scalar * update_value_scalar * adjust_scale_factor
                                        m.thresh.data.sub_(threshold_update)
                                
                                spike_cnt.zero_()
                            except Exception as e:
                                print(f"[DEBUG] {name}: 异常 {e}，使用默认值0.0")
                                update_value_scalar = 0.0
                        else:
                            print(f"[DEBUG] {name}: mem或spike_cnt不是tensor，使用默认值0.0")
                            update_value_scalar = 0.0
                    else:
                        print(f"[DEBUG] {name}: 不是IF或不在fc_if_layers中，使用默认值0.0")
                        update_value_scalar = 0.0
                    
                    # 每个层只添加一次值
                    layer_update_values.append(update_value_scalar)
                
                # 保存avg_update_values用于后续layer_weights更新
                avg_update_values = torch.tensor(layer_update_values, device=device)
                
            if len(thresh_tensors) > 0 and probe_num_batches > 0:
                # 使用训练数据迭代器
                data_iter = iter(train_loader)
                
                # 累计梯度求平均
                sum_grads = [torch.zeros_like(t, device=t.device) for t in thresh_tensors]
                used_batches = 0
                sum_update_values = 0.0  # 累计update_value用于后续计算
                
                # 内存优化：减少探测批次大小  #====================
                probe_batch_size = min(32, images.shape[0])  # 限制探测批次大小
                
                for _ in range(probe_num_batches):
                    try:
                        p_images, p_labels = next(data_iter)
                    except StopIteration:
                        break
                    
                    # 内存优化：如果批次太大，只使用部分数据   #====================
                    if p_images.shape[0] > probe_batch_size:
                        indices = torch.randperm(p_images.shape[0])[:probe_batch_size]
                        p_images = p_images[indices]
                        p_labels = p_labels[indices]
                    
                    p_images = p_images.to(device)
                    p_labels = p_labels.to(device)

                    # 前向传播
                    p_outputs = model(p_images)
                    if p_outputs.dim() > 2:
                        p_outputs = p_outputs.mean(0)
                    p_loss = criterion(p_outputs, p_labels)

                    # 计算阈值梯度
                    grads = torch.autograd.grad(
                        p_loss,
                        thresh_tensors,
                        retain_graph=False,
                        create_graph=False,
                        allow_unused=True
                    )

                    for idx, g in enumerate(grads):
                        if g is not None:
                            sum_grads[idx] += g.detach()
                    used_batches += 1
                    
                    # 内存优化：及时清理中间变量   #===================================
                    del p_images, p_labels, p_outputs, p_loss
                    if 'grads' in locals():
                        del grads
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None

                # 计算平均梯度
                if used_batches > 0:
                    avg_grads = torch.stack([s / float(used_batches) for s in sum_grads])
                    
                    if return_thre_grads:
                        # 记录梯度历史
                        avg_record = {}
                        for (layer_name, _m), avg_g in zip(named_if_modules, avg_grads):
                            avg_record[layer_name] = avg_g.detach().cpu().clone()
                        thre_grads_history.append({
                            'step': i + 1,
                            'grads': avg_record,
                            'used_batches': used_batches
                        })
                
                    # 更新layer_weights
                    if layer_weights is not None and layer_weights.requires_grad:
                        # 确保avg_grads的形状与layer_weights匹配
                        if avg_grads.shape != layer_weights.shape:
                            avg_grads = avg_grads.squeeze()  # 移除多余的维度
                        
                        # 使用更稳定的更新公式，避免数值发散
                        update_step = adjust_scale_factor * avg_grads
                        # 限制更新步长，防止发散
                        max_update = 0.001  # 最大更新步长
                        update_step = torch.clamp(update_step, -max_update, max_update)
                        
                        # 使用in-place操作更新权重
                        layer_weights.data.sub_(update_step)
                        
    # 训练结束后恢复为ANN模式
    model.set_T(0)
    model.train()  #===================================================    
    
    if return_thre_grads:
        return running_loss, 100 * correct / total, thre_grads_history
    return running_loss, 100 * correct / total

import os
import datetime

def val(model, test_loader, device, T,optimize_thre_flag=False):
    correct = 0
    total = 0
    model.eval()

    for m in model.modules():
        if isinstance(m, IF):
            m.optimize_thre_flag = optimize_thre_flag

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate((test_loader)):
            inputs = inputs.to(device)
            #if T is not None and T > 0:
            if getattr(model, 'T', 0) > 0:
                outputs = model(inputs).mean(0)
            else:
                outputs = model(inputs)
            _, predicted = outputs.cpu().max(1)
            total += float(targets.size(0))
            correct += float(predicted.eq(targets).sum().item())

            # # 采集最优thre，运行指定batchsize就退出
            # if optimize_thre_flag and batch_idx == 1:
            #     # 保存每个IF层的阈值数据
            #     if_count = 0  # 用于记录IF层的索引
            #     for m in model.modules():
            #         if isinstance(m, IF):
            #             os.makedirs('log', exist_ok=True)
            #             # 为每个IF层创建独立的CSV文件
            #             save_path = os.path.join('log', f'IF_{if_count}_thresholds.csv')
            #             m.save_thresholds_to_csv(save_path)
            #             if_count += 1
            #     print("保存每个IF层的阈值数据完成！")
            #     # print(f"batch:{batch_idx}")
            #     # print(f"final_acc:{100 * correct / total}")
            #     break

        final_acc = 100 * correct / total
    return final_acc


def mem_distribution(model, data_loader, device, targets, max_batches=None, bins=50, save_dir='mem_plots', show=False, x_range=None):
    """
    在给定数据集上，收集并绘制选定 IF 层中神经元的膜电位分布直方图。

    参数:
    - model: 已包含 IF 层的模型
    - data_loader: 数据加载器（运行数据集）
    - device: 设备
    - targets: 字典，键为 IF 层在 named_modules 中的名字，值为需要统计的神经元索引列表
              例如: { 'classifier.2': [0, 5, 10], 'classifier.5': [3] }
              注意: 若 IF 层输出不是二维 (batch, features)，则会在特征维度上展平后索引
    - max_batches: 最多遍历的 batch 数（None 表示全量）
    - bins: 直方图分箱数
    - save_dir: 图片保存目录
    - show: 是否在保存后显示图像

    返回:
    - collected: { layer_name: { neuron_index: 1D tensor of collected membrane potentials } }
    """
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    # 仅一次性获取目标 IF 层引用，减少循环开销
    target_layers = set(targets.keys())
    named_if = {name: m for name, m in model.named_modules() if name in target_layers and isinstance(m, IF)}

    # 预分配存储
    collected = {name: {idx: [] for idx in idxs} for name, idxs in targets.items()}

    with torch.no_grad():
        for b_idx, (images, _labels) in enumerate(data_loader):
            if max_batches is not None and b_idx >= max_batches:
                break
            images = images.to(device)
            _ = model(images)

            for layer_name, m in named_if.items():
                mem = getattr(m, 'mem', None)
                if not isinstance(mem, torch.Tensor):
                    continue
                mem_flat = mem.detach().view(mem.size(0), -1)
                feat = mem_flat.size(1)
                for idx in targets[layer_name]:
                    if 0 <= idx < feat:
                        collected[layer_name][idx].append(mem_flat[:, idx].cpu())

    # 绘图
    for layer_name, idx_dict in collected.items():
        for neuron_idx, chunks in idx_dict.items():
            if not chunks:
                continue
            values = torch.cat(chunks, dim=0).numpy()
            
            # 打印统计信息
            print(f"\n{layer_name} - neuron {neuron_idx} 膜电位统计:")
            print(f"  样本数: {len(values)}")
            print(f"  均值: {values.mean():.4f}")
            print(f"  标准差: {values.std():.4f}")
            print(f"  最小值: {values.min():.4f}")
            print(f"  最大值: {values.max():.4f}")
            print(f"  25%分位数: {np.percentile(values, 25):.4f}")
            print(f"  50%分位数: {np.percentile(values, 50):.4f}")
            print(f"  75%分位数: {np.percentile(values, 75):.4f}")
            
            plt.figure(figsize=(8, 5))
            # 固定x轴范围与统一分桶（若提供x_range）
            if x_range is not None and isinstance(x_range, (tuple, list)) and len(x_range) == 2:
                fixed_bins = np.linspace(x_range[0], x_range[1], bins + 1)
                plt.hist(values, bins=fixed_bins, color='#4C72B0', alpha=0.85, edgecolor='black')
                plt.xlim(x_range[0], x_range[1])
            else:
                plt.hist(values, bins=bins, color='#4C72B0', alpha=0.85, edgecolor='black')
            plt.title(f'Membrane Potential Distribution\n{layer_name} - neuron {neuron_idx}\nMean: {values.mean():.3f}, Std: {values.std():.3f}')
            plt.xlabel('Membrane potential (mem)')
            plt.ylabel('Count')
            plt.tight_layout()
            out_path = os.path.join(save_dir, f"{layer_name.replace('.', '_')}_neuron{neuron_idx}_hist.png")
            plt.savefig(out_path, dpi=150)
            if show:
                plt.show()
            plt.close()

    # 拼接为张量后返回
    for layer_name, idx_dict in collected.items():
        for neuron_idx, chunks in idx_dict.items():
            collected[layer_name][neuron_idx] = torch.cat(chunks, dim=0) if chunks else torch.empty(0)

    return collected


def evaluate_mem_distribution_similarity(collected_data, layer_name, neuron1_idx, neuron2_idx, methods=['kl', 'js', 'wasserstein', 'correlation']):
    """
    评估两个神经元膜电位分布的相似程度
    
    参数:
    - collected_data: mem_distribution函数返回的数据
    - layer_name: 层名称
    - neuron1_idx: 第一个神经元索引
    - neuron2_idx: 第二个神经元索引
    - methods: 要计算的相似度方法列表 ['kl', 'js', 'wasserstein', 'correlation']
    
    返回:
    - similarity_results: 包含各种相似度指标的字典
    """
    import numpy as np
    from scipy import stats
    from scipy.spatial.distance import jensenshannon
    from scipy.stats import wasserstein_distance
    
    # 获取膜电位数据
    mem1 = collected_data[layer_name][neuron1_idx].numpy()
    mem2 = collected_data[layer_name][neuron2_idx].numpy()
    
    if len(mem1) == 0 or len(mem2) == 0:
        raise ValueError("神经元膜电位数据为空")
    
    similarity_results = {
        'layer_name': layer_name,
        'neuron1_idx': neuron1_idx,
        'neuron2_idx': neuron2_idx,
        'neuron1_stats': {
            'count': len(mem1),
            'mean': mem1.mean(),
            'std': mem1.std(),
            'min': mem1.min(),
            'max': mem1.max()
        },
        'neuron2_stats': {
            'count': len(mem2),
            'mean': mem2.mean(),
            'std': mem2.std(),
            'min': mem2.min(),
            'max': mem2.max()
        }
    }
    
    # 计算KL散度
    if 'kl' in methods:
        try:
            # 创建统一的bins
            min_val = min(mem1.min(), mem2.min())
            max_val = max(mem1.max(), mem2.max())
            bins = np.linspace(min_val, max_val, 50)
            
            # 计算直方图
            hist1, _ = np.histogram(mem1, bins=bins, density=True)
            hist2, _ = np.histogram(mem2, bins=bins, density=True)
            
            # 避免零值
            hist1 = hist1 + 1e-10
            hist2 = hist2 + 1e-10
            
            # 归一化
            hist1 = hist1 / hist1.sum()
            hist2 = hist2 / hist2.sum()
            
            # KL散度 (P||Q)
            kl_div_1_2 = stats.entropy(hist1, hist2)
            kl_div_2_1 = stats.entropy(hist2, hist1)
            
            similarity_results['kl_divergence'] = {
                'kl_1_to_2': kl_div_1_2,
                'kl_2_to_1': kl_div_2_1,
                'kl_symmetric': (kl_div_1_2 + kl_div_2_1) / 2
            }
        except Exception as e:
            similarity_results['kl_divergence'] = {'error': str(e)}
    
    # 计算JS散度 (Jensen-Shannon divergence)
    if 'js' in methods:
        try:
            # 创建统一的bins
            min_val = min(mem1.min(), mem2.min())
            max_val = max(mem1.max(), mem2.max())
            bins = np.linspace(min_val, max_val, 50)
            
            # 计算直方图
            hist1, _ = np.histogram(mem1, bins=bins, density=True)
            hist2, _ = np.histogram(mem2, bins=bins, density=True)
            
            # 避免零值
            hist1 = hist1 + 1e-10
            hist2 = hist2 + 1e-10
            
            # 归一化
            hist1 = hist1 / hist1.sum()
            hist2 = hist2 / hist2.sum()
            
            # JS散度
            js_div = jensenshannon(hist1, hist2)
            
            similarity_results['js_divergence'] = {
                'js_distance': js_div,
                'js_similarity': 1 - js_div  # 转换为相似度 (0-1)
            }
        except Exception as e:
            similarity_results['js_divergence'] = {'error': str(e)}
    
    # 计算Wasserstein距离 (Earth Mover's Distance)
    if 'wasserstein' in methods:
        try:
            wasserstein_dist = wasserstein_distance(mem1, mem2)
            similarity_results['wasserstein_distance'] = {
                'distance': wasserstein_dist,
                'normalized_similarity': 1 / (1 + wasserstein_dist)  # 转换为相似度
            }
        except Exception as e:
            similarity_results['wasserstein_distance'] = {'error': str(e)}
    
    # 计算皮尔逊相关系数
    if 'correlation' in methods:
        try:
            # 如果两个分布长度不同，需要处理
            min_len = min(len(mem1), len(mem2))
            if min_len > 0:
                # 随机采样到相同长度
                idx1 = np.random.choice(len(mem1), min_len, replace=False)
                idx2 = np.random.choice(len(mem2), min_len, replace=False)
                
                corr_coef, p_value = stats.pearsonr(mem1[idx1], mem2[idx2])
                
                similarity_results['correlation'] = {
                    'pearson_correlation': corr_coef,
                    'p_value': p_value,
                    'sample_size': min_len
                }
        except Exception as e:
            similarity_results['correlation'] = {'error': str(e)}
    
    return similarity_results


def compare_multiple_neurons(collected_data, layer_name, neuron_indices, method='js'):
    """
    比较多个神经元的膜电位分布相似度
    
    参数:
    - collected_data: mem_distribution函数返回的数据
    - layer_name: 层名称
    - neuron_indices: 神经元索引列表
    - method: 相似度计算方法 ('kl', 'js', 'wasserstein', 'correlation')
    
    返回:
    - comparison_matrix: 相似度矩阵
    """
    import numpy as np
    
    n_neurons = len(neuron_indices)
    comparison_matrix = np.zeros((n_neurons, n_neurons))
    
    for i in range(n_neurons):
        for j in range(n_neurons):
            if i == j:
                comparison_matrix[i, j] = 1.0  # 自己与自己的相似度为1
            else:
                try:
                    result = evaluate_mem_distribution_similarity(
                        collected_data, layer_name, 
                        neuron_indices[i], neuron_indices[j], 
                        methods=[method]
                    )
                    
                    if method == 'kl':
                        similarity = 1 / (1 + result['kl_divergence']['kl_symmetric'])
                    elif method == 'js':
                        similarity = result['js_divergence']['js_similarity']
                    elif method == 'wasserstein':
                        similarity = result['wasserstein_distance']['normalized_similarity']
                    elif method == 'correlation':
                        similarity = abs(result['correlation']['pearson_correlation'])
                    
                    comparison_matrix[i, j] = similarity
                    
                except Exception as e:
                    print(f"计算神经元 {neuron_indices[i]} 和 {neuron_indices[j]} 相似度时出错: {e}")
                    comparison_matrix[i, j] = 0.0
    
    return comparison_matrix, neuron_indices







