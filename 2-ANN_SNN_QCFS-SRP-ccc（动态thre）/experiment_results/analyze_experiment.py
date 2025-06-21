#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验数据分析脚本
解析experiment_summary.txt文件，生成统计表格
"""

import re
import pandas as pd
import numpy as np

def parse_experiment_file(filename):
    """解析实验总结文件"""
    experiments = []
    
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 分割每个实验
    experiment_blocks = content.split('----------------------------------------')
    
    for block in experiment_blocks:
        if not block.strip():
            continue
            
        # 提取参数
        param_match = re.search(r'参数: ratio=([\d.]+), order=(\w+), sort_by=(\w+)', block)
        if not param_match:
            continue
            
        ratio = float(param_match.group(1))
        order = param_match.group(2)
        sort_by = param_match.group(3)
        
        # 提取结果
        acc_before_match = re.search(r'剪枝前准确率: ([\d.]+)%', block)
        loss_before_match = re.search(r'剪枝前损失: ([\d.]+)', block)
        acc_after_match = re.search(r'剪枝后准确率: ([\d.]+)%', block)
        loss_after_match = re.search(r'剪枝后损失: ([\d.]+)', block)
        
        if all([acc_before_match, loss_before_match, acc_after_match, loss_after_match]):
            experiment = {
                'ratio': ratio,
                'order': order,
                'sort_by': sort_by,
                'acc_after': float(acc_after_match.group(1)),
                'loss_after': float(loss_after_match.group(1))
            }
            experiments.append(experiment)
    
    return experiments

def create_combined_table(experiments):
    """创建准确率和损失组合表格"""
    df = pd.DataFrame(experiments)
    
    # 创建透视表，同时显示准确率和损失
    acc_pivot = df.pivot_table(
        values='acc_after', 
        index=['order', 'sort_by'], 
        columns='ratio', 
        aggfunc='mean'
    )
    
    loss_pivot = df.pivot_table(
        values='loss_after', 
        index=['order', 'sort_by'], 
        columns='ratio', 
        aggfunc='mean'
    )
    
    return acc_pivot, loss_pivot

def save_combined_table_to_file(acc_pivot, loss_pivot, filename):
    """保存组合表格到文件"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("实验数据分析结果 - 准确率与损失对比\n")
        f.write("=" * 60 + "\n\n")
        
        # 创建组合表格格式
        f.write("准确率与损失对比表 (order&sort_by vs ratio)\n")
        f.write("-" * 60 + "\n")
        
        # 获取所有ratio值
        ratios = sorted(acc_pivot.columns)
        
        # 表头
        header = f"{'配置':<20}"
        for ratio in ratios:
            header += f"  ratio={ratio:<8}"
        f.write(header + "\n")
        
        # 分隔线
        f.write("-" * (20 + 9 * len(ratios)) + "\n")
        
        # 数据行
        for idx in acc_pivot.index:
            order, sort_by = idx
            config_name = f"{order}_{sort_by}"
            row = f"{config_name:<20}"
            
            for ratio in ratios:
                acc_val = acc_pivot.loc[idx, ratio]
                loss_val = loss_pivot.loc[idx, ratio]
                cell = f"{acc_val:.2f}%/{loss_val:.3f}"
                row += f"  {cell:<8}"
            
            f.write(row + "\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("格式说明: 准确率%/损失值\n")

def main():
    # 解析实验文件
    experiments = parse_experiment_file('experiment_summary.txt')
    print(f"解析到 {len(experiments)} 个实验")
    
    # 创建组合表格
    acc_pivot, loss_pivot = create_combined_table(experiments)
    
    # 保存结果
    save_combined_table_to_file(acc_pivot, loss_pivot, 'experiment_analysis_results.txt')
    
    # 打印结果
    print("\n准确率与损失对比表:")
    
    # 打印组合表格
    ratios = sorted(acc_pivot.columns)
    header = f"{'配置':<20}"
    for ratio in ratios:
        header += f"  ratio={ratio:<8}"
    print(header)
    print("-" * (20 + 9 * len(ratios)))
    
    for idx in acc_pivot.index:
        order, sort_by = idx
        config_name = f"{order}_{sort_by}"
        row = f"{config_name:<20}"
        
        for ratio in ratios:
            acc_val = acc_pivot.loc[idx, ratio]
            loss_val = loss_pivot.loc[idx, ratio]
            cell = f"{acc_val:.2f}%/{loss_val:.3f}"
            row += f"  {cell:<8}"
        
        print(row)
    
    print(f"\n格式说明: 准确率%/损失值")
    print("分析结果已保存到 experiment_analysis_results.txt")

if __name__ == "__main__":
    main() 