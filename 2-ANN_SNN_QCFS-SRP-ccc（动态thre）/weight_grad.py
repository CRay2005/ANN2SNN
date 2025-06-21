#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from pathlib import Path

# 设置英文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def load_csv_data(csv_path):
    """加载CSV文件数据"""
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded data: {csv_path}")
        print(f"Data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"Failed to load CSV file: {e}")
        return None

def analyze_weight_gradient_relationship(df):
    """分析权重和梯度的关系"""
    print("\n=== Weight and Gradient Relationship Analysis ===")
    
    # 基本统计信息
    print(f"Total neurons: {len(df)}")
    print(f"Average weight value: {df['avg_befor_weight'].mean():.6f}")
    print(f"Average gradient value: {df['gradient_value'].mean():.6f}")
    
    # 计算相关系数
    correlation = df['avg_befor_weight'].corr(df['gradient_value'])
    print(f"Correlation between weight and gradient: {correlation:.6f}")
    
    # 计算重要性分数
    df['importance_score'] = df['avg_befor_weight'] * df['gradient_value']
    print(f"Average importance score: {df['importance_score'].mean():.6f}")
    
    return df

def create_visualizations(df, output_dir, layer_name):
    """创建可视化图表"""
    print(f"\n=== Creating Visualizations ===")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置图表样式
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{layer_name} - Weight and Gradient Relationship Analysis', fontsize=16, fontweight='bold')
    
    # 1. 散点图：权重 vs 梯度
    axes[0, 0].scatter(df['avg_befor_weight'], df['gradient_value'], alpha=0.6, s=20, color='orange')
    axes[0, 0].set_xlabel('Weight Value')
    axes[0, 0].set_ylabel('Gradient Value')
    axes[0, 0].set_title('Weight vs Gradient Scatter Plot')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 添加趋势线
    z = np.polyfit(df['avg_befor_weight'], df['gradient_value'], 1)
    p = np.poly1d(z)
    axes[0, 0].plot(df['avg_befor_weight'], p(df['avg_befor_weight']), "b--", alpha=0.8)
    
    # 2. 权重分布直方图
    axes[0, 1].hist(df['avg_befor_weight'], bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 1].set_xlabel('Weight Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Weight Distribution Histogram')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 梯度分布直方图
    axes[0, 2].hist(df['gradient_value'], bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[0, 2].set_xlabel('Gradient Value')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Gradient Distribution Histogram')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. 重要性分数分布
    axes[1, 0].hist(df['importance_score'], bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[1, 0].set_xlabel('Importance Score')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Importance Score Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. 权重 vs 重要性分数
    axes[1, 1].scatter(df['avg_befor_weight'], df['importance_score'], alpha=0.6, s=20, color='purple')
    axes[1, 1].set_xlabel('Weight Value')
    axes[1, 1].set_ylabel('Importance Score')
    axes[1, 1].set_title('Weight vs Importance Score')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. 梯度 vs 重要性分数
    axes[1, 2].scatter(df['gradient_value'], df['importance_score'], alpha=0.6, s=20, color='orange')
    axes[1, 2].set_xlabel('Gradient Value')
    axes[1, 2].set_ylabel('Importance Score')
    axes[1, 2].set_title('Gradient vs Importance Score')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    output_path = os.path.join(output_dir, f'{layer_name}_weight_grad_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Chart saved: {output_path}")
    plt.show()

def analyze_pruning_strategies(df, pruning_ratios=[0.1, 0.2, 0.3, 0.5, 0.7]):
    """分析不同剪枝策略的效果"""
    print(f"\n=== Pruning Strategy Analysis ===")
    
    strategies = {
        'weight_only': df['avg_befor_weight'],
        'gradient_only': df['gradient_value'],
        'importance_score': df['importance_score'],
        'weight_gradient_sum': df['avg_befor_weight'] + df['gradient_value'],
        'weight_gradient_product': df['avg_befor_weight'] * df['gradient_value']
    }
    
    results = {}
    
    for ratio in pruning_ratios:
        print(f"\nPruning ratio: {ratio*100}%")
        results[ratio] = {}
        
        for strategy_name, scores in strategies.items():
            # 按分数排序，保留高分数的神经元
            sorted_indices = np.argsort(scores)[::-1]  # 从大到小排序
            num_keep = int(len(df) * (1 - ratio))
            kept_indices = sorted_indices[:num_keep]
            
            # 计算保留神经元的平均权重和梯度
            kept_weights = df.iloc[kept_indices]['avg_befor_weight'].mean()
            kept_gradients = df.iloc[kept_indices]['gradient_value'].mean()
            
            results[ratio][strategy_name] = {
                'avg_weight': kept_weights,
                'avg_gradient': kept_gradients,
                'importance_score': kept_weights * kept_gradients
            }
            
            print(f"  {strategy_name:20s}: weight={kept_weights:.6f}, gradient={kept_gradients:.6f}")
    
    return results

def create_pruning_comparison_plot(results, output_dir, layer_name):
    """创建剪枝策略对比图"""
    print(f"\n=== Creating Pruning Strategy Comparison Plot ===")
    
    pruning_ratios = list(results.keys())
    strategies = list(results[pruning_ratios[0]].keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{layer_name} - Pruning Strategy Comparison', fontsize=16, fontweight='bold')
    
    # 1. 平均权重对比
    for strategy in strategies:
        weights = [results[ratio][strategy]['avg_weight'] for ratio in pruning_ratios]
        axes[0, 0].plot(pruning_ratios, weights, marker='o', label=strategy)
    
    axes[0, 0].set_xlabel('Pruning Ratio')
    axes[0, 0].set_ylabel('Average Weight of Kept Neurons')
    axes[0, 0].set_title('Average Weight Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 平均梯度对比
    for strategy in strategies:
        gradients = [results[ratio][strategy]['avg_gradient'] for ratio in pruning_ratios]
        axes[0, 1].plot(pruning_ratios, gradients, marker='s', label=strategy)
    
    axes[0, 1].set_xlabel('Pruning Ratio')
    axes[0, 1].set_ylabel('Average Gradient of Kept Neurons')
    axes[0, 1].set_title('Average Gradient Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 重要性分数对比
    for strategy in strategies:
        importance = [results[ratio][strategy]['importance_score'] for ratio in pruning_ratios]
        axes[1, 0].plot(pruning_ratios, importance, marker='^', label=strategy)
    
    axes[1, 0].set_xlabel('Pruning Ratio')
    axes[1, 0].set_ylabel('Importance Score of Kept Neurons')
    axes[1, 0].set_title('Importance Score Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 综合评分对比（权重和梯度的加权平均）
    for strategy in strategies:
        weights = [results[ratio][strategy]['avg_weight'] for ratio in pruning_ratios]
        gradients = [results[ratio][strategy]['avg_gradient'] for ratio in pruning_ratios]
        # 归一化后加权平均
        norm_weights = np.array(weights) / max(weights)
        norm_gradients = np.array(gradients) / max(gradients)
        combined_score = 0.5 * norm_weights + 0.5 * norm_gradients
        axes[1, 1].plot(pruning_ratios, combined_score, marker='d', label=strategy)
    
    axes[1, 1].set_xlabel('Pruning Ratio')
    axes[1, 1].set_ylabel('Combined Score')
    axes[1, 1].set_title('Combined Score Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    output_path = os.path.join(output_dir, f'{layer_name}_pruning_strategies_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Pruning strategy comparison chart saved: {output_path}")
    plt.show()

def recommend_pruning_strategy(results):
    """推荐最佳剪枝策略"""
    print(f"\n=== Pruning Strategy Recommendation ===")
    
    # 计算每个策略的综合得分
    strategy_scores = {}
    
    for strategy in list(results[list(results.keys())[0]].keys()):
        total_score = 0
        for ratio in results.keys():
            # 综合考虑权重、梯度和重要性分数
            weight_score = results[ratio][strategy]['avg_weight']
            gradient_score = results[ratio][strategy]['avg_gradient']
            importance_score = results[ratio][strategy]['importance_score']
            
            # 归一化评分
            normalized_score = (weight_score + gradient_score + importance_score) / 3
            total_score += normalized_score
        
        strategy_scores[strategy] = total_score / len(results)
    
    # 排序并推荐
    sorted_strategies = sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True)
    
    print("Recommended pruning strategies (ranked by effectiveness):")
    for i, (strategy, score) in enumerate(sorted_strategies, 1):
        print(f"{i}. {strategy:20s}: combined score = {score:.6f}")
    
    return sorted_strategies[0][0]  # 返回最佳策略

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Analyze weight and gradient relationships, recommend pruning strategies')
    parser.add_argument('--csv_path', type=str, required=True, help='CSV file path')
    parser.add_argument('--output_dir', type=str, default='analysis_results', help='Output directory')
    parser.add_argument('--pruning_ratios', nargs='+', type=float, default=[0.1, 0.2, 0.3, 0.5, 0.7], help='List of pruning ratios')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.csv_path):
        print(f"Error: File does not exist - {args.csv_path}")
        return
    
    # 加载数据
    df = load_csv_data(args.csv_path)
    if df is None:
        return
    
    # 检查必要的列是否存在
    required_columns = ['neuron_index', 'avg_befor_weight', 'gradient_value']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing required columns - {missing_columns}")
        return
    
    # 获取层名称
    layer_name = Path(args.csv_path).stem.replace('_weight_grad_low', '')
    
    # 分析权重和梯度关系
    df = analyze_weight_gradient_relationship(df)
    
    # 创建可视化图表
    create_visualizations(df, args.output_dir, layer_name)
    
    # 分析剪枝策略
    results = analyze_pruning_strategies(df, args.pruning_ratios)
    
    # 创建剪枝策略对比图
    create_pruning_comparison_plot(results, args.output_dir, layer_name)
    
    # 推荐最佳策略
    best_strategy = recommend_pruning_strategy(results)
    
    print(f"\n=== Analysis Complete ===")
    print(f"Recommended best pruning strategy: {best_strategy}")
    print(f"Analysis results saved to: {args.output_dir}")

if __name__ == "__main__":
    main() 