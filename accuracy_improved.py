import matplotlib.pyplot as plt
import numpy as np
import re

def parse_data_file(file_path):
    """解析数据文件并返回结构化数据"""
    data = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 按不同测试场景分割
    sections = content.split('===========')
    
    for section in sections:
        if not section.strip():
            continue
            
        lines = section.strip().split('\n')
        if not lines:
            continue
            
        # 提取测试场景信息
        header = lines[0]
        if 'cifar10,T=4' in header:
            dataset, timestep = 'cifar10', 'T4'
        elif 'cifar10,T=8' in header:
            dataset, timestep = 'cifar10', 'T8'
        elif 'cifar100,T=4' in header:
            dataset, timestep = 'cifar100', 'T4'
        elif 'cifar100,T=8' in header:
            dataset, timestep = 'cifar100', 'T8'
        else:
            continue
        
        # 解析gradient方法数据
        gradient_acc = None
        gradient_loss = None
        improved_acc = None
        improved_loss = None
        
        i = 1
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith('==Gradient'):
                # 下一行是准确率，再下一行是loss
                if i + 2 < len(lines):
                    acc_line = lines[i + 1].strip()
                    loss_line = lines[i + 2].strip()
                    
                    # 解析准确率数据
                    gradient_acc = [float(x) for x in acc_line.split('\t') if x.strip()]
                    
                    # 解析loss数据
                    gradient_loss = [float(x) for x in loss_line.split('\t') if x.strip()]
                    
                i += 3
                continue
                
            elif line.startswith('==Improved method'):
                # 寻找准确率和loss数据
                j = i + 1
                while j < len(lines):
                    current_line = lines[j].strip()
                    
                    # 跳过ratio行（0.1 0.2 ...）
                    if current_line and not current_line.startswith('0.1'):
                        # 检查是否为准确率行（包含%）
                        if '%' in current_line:
                            # 解析准确率数据（去除%符号）
                            acc_values = [float(x.replace('%', '')) for x in current_line.split() if x.strip()]
                            improved_acc = acc_values
                        else:
                            # 检查是否为数字行（loss数据）
                            try:
                                loss_values = [float(x) for x in current_line.split() if x.strip()]
                                if len(loss_values) == 9:  # 确保是完整的数据行
                                    improved_loss = loss_values
                                    break
                            except ValueError:
                                pass
                    j += 1
                break
                
            i += 1
        
        # 存储数据
        ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        if gradient_acc and len(gradient_acc) == 9:
            key = f'{dataset}_{timestep}_Gradient'
            data[key] = {
                'ratio': ratio,
                'accuracy': gradient_acc,
                'loss': gradient_loss if gradient_loss else [0] * 9
            }
        
        if improved_acc and len(improved_acc) == 9:
            key = f'{dataset}_{timestep}_Improved'
            data[key] = {
                'ratio': ratio,
                'accuracy': improved_acc,
                'loss': improved_loss if improved_loss else [0] * 9
            }
    
    return data

# 读取并解析数据
file_path = '/root/autodl-tmp/0-ANN2SNN-Allinone/0000weight_grad_in_out.txt'
data = parse_data_file(file_path)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建图形 - 准确率对比图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 定义4种不同的颜色来区分4个测试场景
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # 蓝色、橘色、绿色、红色

# 定义线型：实线用于Gradient，虚线用于Improved
line_styles = ['-', '--']

# 左图：准确率对比
for i, (key, values) in enumerate(data.items()):
    # 根据测试场景确定颜色
    if 'cifar10_T4' in key:
        color = colors[0]  # 蓝色
    elif 'cifar10_T8' in key:
        color = colors[1]  # 橘色
    elif 'cifar100_T4' in key:
        color = colors[2]  # 绿色
    elif 'cifar100_T8' in key:
        color = colors[3]  # 红色
    
    # 根据方法确定线型
    if 'Gradient' in key:
        linestyle = line_styles[0]  # 实线
    else:  # Improved
        linestyle = line_styles[1]  # 虚线
    
    ax1.plot(values['ratio'], values['accuracy'], 
             color=color, 
             linestyle=linestyle,
             linewidth=2, 
             marker='o', 
             markersize=6,
             label=key)

# 设置左图属性
ax1.set_xlabel('Ratio', fontsize=14)
ax1.set_ylabel('Accuracy (%)', fontsize=14)
ax1.set_title('Accuracy vs Ratio: Gradient vs Improved Method', fontsize=16, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 100)

# 右图：Loss对比
for i, (key, values) in enumerate(data.items()):
    # 根据测试场景确定颜色
    if 'cifar10_T4' in key:
        color = colors[0]  # 蓝色
    elif 'cifar10_T8' in key:
        color = colors[1]  # 橘色
    elif 'cifar100_T4' in key:
        color = colors[2]  # 绿色
    elif 'cifar100_T8' in key:
        color = colors[3]  # 红色
    
    # 根据方法确定线型
    if 'Gradient' in key:
        linestyle = line_styles[0]  # 实线
    else:  # Improved
        linestyle = line_styles[1]  # 虚线
    
    ax2.plot(values['ratio'], values['loss'], 
             color=color, 
             linestyle=linestyle,
             linewidth=2, 
             marker='s', 
             markersize=6,
             label=key)

# 设置右图属性
ax2.set_xlabel('Ratio', fontsize=14)
ax2.set_ylabel('Loss', fontsize=14)
ax2.set_title('Loss vs Ratio: Gradient vs Improved Method', fontsize=16, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 5)

# 添加图例到左图
legend1 = ax1.legend(loc='lower left', fontsize=9, handlelength=4)
for line in legend1.get_lines():
    line.set_linewidth(2)
    line.set_markersize(4)

# 添加图例到右图
legend2 = ax2.legend(loc='upper left', fontsize=9, handlelength=4)
for line in legend2.get_lines():
    line.set_linewidth(2)
    line.set_markersize(4)

# 调整布局
plt.tight_layout()

# 保存图片
plt.savefig('/root/autodl-tmp/0-ANN2SNN-Allinone/accuracy_improved_plot.png', dpi=300, bbox_inches='tight')
plt.savefig('/root/autodl-tmp/0-ANN2SNN-Allinone/accuracy_improved_plot.pdf', bbox_inches='tight')

# 显示图形
plt.show()

# 创建单独的准确率图
plt.figure(figsize=(12, 8))

# 绘制准确率对比图
for i, (key, values) in enumerate(data.items()):
    # 根据测试场景确定颜色
    if 'cifar10_T4' in key:
        color = colors[0]  # 蓝色
    elif 'cifar10_T8' in key:
        color = colors[1]  # 橘色
    elif 'cifar100_T4' in key:
        color = colors[2]  # 绿色
    elif 'cifar100_T8' in key:
        color = colors[3]  # 红色
    
    # 根据方法确定线型
    if 'Gradient' in key:
        linestyle = line_styles[0]  # 实线
    else:  # Improved
        linestyle = line_styles[1]  # 虚线
    
    plt.plot(values['ratio'], values['accuracy'], 
             color=color, 
             linestyle=linestyle,
             linewidth=2, 
             marker='o', 
             markersize=6,
             label=key)

# 设置图表属性
plt.xlabel('Ratio', fontsize=14)
plt.ylabel('Accuracy (%)', fontsize=14)
plt.title('Accuracy Comparison: Gradient vs Improved Method', fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3)

# 将图例放在左下角
legend = plt.legend(loc='lower left', fontsize=10, handlelength=4)
for line in legend.get_lines():
    line.set_linewidth(2)
    line.set_markersize(4)

# 设置坐标轴范围
plt.xlim(0, 1)
plt.ylim(0, 100)

# 调整布局
plt.tight_layout()

# 保存单独的准确率图
plt.savefig('/root/autodl-tmp/0-ANN2SNN-Allinone/accuracy_only_plot.png', dpi=300, bbox_inches='tight')
plt.savefig('/root/autodl-tmp/0-ANN2SNN-Allinone/accuracy_only_plot.pdf', bbox_inches='tight')

plt.show()

# 输出数据统计信息
print("数据解析完成！")
print(f"共解析了 {len(data)} 个测试场景的数据")
print("\n解析的数据包括：")
for key in data.keys():
    print(f"- {key}")

print("\n图表已生成并保存为：")
print("- accuracy_improved_plot.png/pdf (准确率和Loss对比)")
print("- accuracy_only_plot.png/pdf (仅准确率对比)") 