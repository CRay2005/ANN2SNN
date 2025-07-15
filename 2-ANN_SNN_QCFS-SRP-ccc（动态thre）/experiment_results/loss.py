import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 读取数据 - 按测试场景重新组织
data = {
    'cifar10_T4_Random': {
        'ratio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        'loss': [0.176, 0.183, 0.230, 0.407, 0.681, 1.119, 1.594, 1.883, 2.281]
    },
    'cifar10_T4_Gradian': {
        'ratio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        'loss': [0.190, 0.188, 0.187, 0.199, 0.244, 0.397, 0.725, 1.306, 1.900]
    },
    'cifar10_T8_Random': {
        'ratio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        'loss': [0.144, 0.159, 0.227, 0.402, 0.742, 1.187, 1.601, 1.943, 2.189]
    },
    'cifar10_T8_Gradient': {
        'ratio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        'loss': [0.151, 0.151, 0.156, 0.176, 0.263, 0.418, 0.772, 1.346, 1.901]
    },
    'cifar100_T4_Random': {
        'ratio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        'loss': [1.109, 1.087, 1.226, 1.562, 2.156, 2.814, 3.481, 4.141, 4.592]
    },
    'cifar100_T4_Gradient': {
        'ratio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        'loss': [1.220, 1.220, 1.220, 1.203, 1.196, 1.359, 2.031, 3.025, 3.969]
    },
    'cifar100_T8_Random': {
        'ratio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        'loss': [0.929, 0.945, 1.131, 1.530, 2.212, 2.911, 3.593, 4.116, 4.491]
    },
    'cifar100_T8_Gradient': {
        'ratio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        'loss': [1.004, 1.004, 1.004, 0.993, 0.988, 1.106, 1.617, 2.775, 3.865]
    }
}

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建图形
plt.figure(figsize=(12, 8))

# 定义4种不同的颜色来区分4个测试场景
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # 蓝色、橘色、绿色、红色

# 定义线型：实线用于Random，虚线用于Gradient
line_styles = ['-', '--']

# 绘制每条线
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
    if 'Random' in key:
        linestyle = line_styles[0]  # 实线
    else:  # Gradient
        linestyle = line_styles[1]  # 虚线
    
    plt.plot(values['ratio'], values['loss'], 
             color=color, 
             linestyle=linestyle,
             linewidth=2, 
             marker='o', 
             markersize=6,
             label=key)

# 设置图表属性
plt.xlabel('Ratio', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title('Loss vs Ratio for Different Test Conditions', fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3)

# 在y=1.37处添加横线，表示gradient剪枝的界限，只在0.1-0.9范围内显示
plt.axhline(y=1.37, color='black', linestyle=':', linewidth=2, alpha=0.7, xmin=0.1, xmax=0.9, label='Gradient Pruning Threshold')

# 将图例放在左上角，并调整图例的线条显示效果
legend = plt.legend(loc='upper left', fontsize=10, handlelength=4)
# 调整图例中线条的显示效果
for line in legend.get_lines():
    line.set_linewidth(2)  # 调整图例中线条的宽度
    line.set_markersize(4)  # 减小图例中标记点的大小

# 设置坐标轴范围
plt.xlim(0, 1)
plt.ylim(0, 5)

# 调整布局
plt.tight_layout()

# 保存图片
plt.savefig('loss_plot.png', dpi=300, bbox_inches='tight')
plt.savefig('loss_plot.pdf', bbox_inches='tight')

# 显示图形
plt.show()

print("Loss plot has been generated and saved as 'loss_plot.png' and 'loss_plot.pdf'") 