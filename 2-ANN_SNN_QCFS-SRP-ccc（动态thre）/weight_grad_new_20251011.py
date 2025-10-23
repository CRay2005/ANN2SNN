#!/usr/bin/env python3
"""
FC Layer Weight and Gradient Relationship Analysis Script
Plot relationships between FC layer weights and FC weight gradients, IF output gradients, IF input gradients
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path
from scipy import stats

# Set font and style
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 8)

def load_csv_data(csv_path):
    """Load CSV file data"""
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Successfully loaded data: {csv_path}")
        print(f"📊 Data shape: {df.shape}")
        print(f"📋 Columns: {list(df.columns)}")
        print(f"📈 Data preview:")
        print(df.head())
        return df
    except Exception as e:
        print(f"❌ Failed to load CSV file: {e}")
        return None

def analyze_weight_gradient_relationships(df, analysis_type="all"):
    """Analyze relationships between FC layer weights and various gradients"""
    print("\n" + "="*80)
    print(f"FC Layer Weight and Gradient Relationship Analysis ({analysis_type.title()} Neurons)")
    print("="*80)
    
    # Check if pruning_status column exists and filter data accordingly
    if 'pruning_status' in df.columns:
        if analysis_type == "kept":
            analysis_df = df[df['pruning_status'] == 1]
            print(f"📊 Analyzing kept neurons only (pruning_status = 1)")
        elif analysis_type == "pruned":
            analysis_df = df[df['pruning_status'] == 0]
            print(f"📊 Analyzing pruned neurons only (pruning_status = 0)")
        else:
            analysis_df = df
            print(f"📊 Analyzing all neurons")
        
        print(f"📊 Pruning status breakdown:")
        print(f"   Total neurons: {len(df)}")
        print(f"   Kept neurons: {len(df[df['pruning_status'] == 1])}")
        print(f"   Pruned neurons: {len(df[df['pruning_status'] == 0])}")
        print(f"   Analyzing: {len(analysis_df)} neurons")
    else:
        analysis_df = df
        print(f"📊 No pruning_status column found, analyzing all {len(df)} neurons")
    
    # Basic statistics
    print(f"\n📊 FC layer weight statistics:")
    print(f"   Mean: {analysis_df['fc_weight'].mean():.8f}")
    print(f"   Std: {analysis_df['fc_weight'].std():.8f}")
    print(f"   Min: {analysis_df['fc_weight'].min():.8f}")
    print(f"   Max: {analysis_df['fc_weight'].max():.8f}")
    
    print(f"📊 FC layer weight gradient statistics:")
    print(f"   Mean: {analysis_df['fc_weight_grad'].mean():.8f}")
    print(f"   Std: {analysis_df['fc_weight_grad'].std():.8f}")
    print(f"   Min: {analysis_df['fc_weight_grad'].min():.8f}")
    print(f"   Max: {analysis_df['fc_weight_grad'].max():.8f}")
    
    print(f"📊 IF layer output gradient statistics:")
    print(f"   Mean: {analysis_df['if_output_grad'].mean():.8f}")
    print(f"   Std: {analysis_df['if_output_grad'].std():.8f}")
    print(f"   Min: {analysis_df['if_output_grad'].min():.8f}")
    print(f"   Max: {analysis_df['if_output_grad'].max():.8f}")
    
    print(f"📊 IF layer input gradient statistics:")
    print(f"   Mean: {analysis_df['if_input_grad'].mean():.8f}")
    print(f"   Std: {analysis_df['if_input_grad'].std():.8f}")
    print(f"   Min: {analysis_df['if_input_grad'].min():.8f}")
    print(f"   Max: {analysis_df['if_input_grad'].max():.8f}")
    
    # Calculate correlation coefficients
    print(f"\n🔗 Correlation analysis:")
    corr_fc_weight_grad = analysis_df['fc_weight'].corr(analysis_df['fc_weight_grad'])
    corr_fc_if_output = analysis_df['fc_weight'].corr(analysis_df['if_output_grad'])
    corr_fc_if_input = analysis_df['fc_weight'].corr(analysis_df['if_input_grad'])
    
    print(f"   FC Weight vs FC Weight Gradient: {corr_fc_weight_grad:.6f}")
    print(f"   FC Weight vs IF Output Gradient: {corr_fc_if_output:.6f}")
    print(f"   FC Weight vs IF Input Gradient: {corr_fc_if_input:.6f}")
    
    # Calculate Spearman rank correlation coefficients
    print(f"\n🔗 Spearman rank correlation analysis:")
    spearman_fc_weight_grad, _ = stats.spearmanr(analysis_df['fc_weight'], analysis_df['fc_weight_grad'])
    spearman_fc_if_output, _ = stats.spearmanr(analysis_df['fc_weight'], analysis_df['if_output_grad'])
    spearman_fc_if_input, _ = stats.spearmanr(analysis_df['fc_weight'], analysis_df['if_input_grad'])
    
    print(f"   FC Weight vs FC Weight Gradient: {spearman_fc_weight_grad:.6f}")
    print(f"   FC Weight vs IF Output Gradient: {spearman_fc_if_output:.6f}")
    print(f"   FC Weight vs IF Input Gradient: {spearman_fc_if_input:.6f}")
    
    return {
        'analysis_type': analysis_type,
        'neuron_count': len(analysis_df),
        'fc_weight_grad_corr': corr_fc_weight_grad,
        'fc_if_output_corr': corr_fc_if_output,
        'fc_if_input_corr': corr_fc_if_input,
        'spearman_fc_weight_grad': spearman_fc_weight_grad,
        'spearman_fc_if_output': spearman_fc_if_output,
        'spearman_fc_if_input': spearman_fc_if_input
    }

def create_scatter_plots(df, output_dir, layer_name):
    """Create scatter plots: FC layer weight relationships with various gradients"""
    print(f"\n📊 Creating scatter plots...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if pruning_status column exists
    has_pruning_status = 'pruning_status' in df.columns
    if has_pruning_status:
        # Filter data for kept neurons only (pruning_status = 1)
        kept_df = df[df['pruning_status'] == 1]
        pruned_df = df[df['pruning_status'] == 0]
        pruned_count = len(pruned_df)
        kept_count = len(kept_df)
        print(f"📊 Pruning status analysis:")
        print(f"   Total neurons: {len(df)}")
        print(f"   Kept neurons: {kept_count}")
        print(f"   Pruned neurons: {pruned_count}")
        print(f"   Creating 6 plots: 3 for kept neurons + 3 for pruned neurons...")
        
        # Use both kept and pruned neurons for analysis
        plot_df_kept = kept_df
        plot_df_pruned = pruned_df
        plot_title_suffix = " (Kept vs Pruned Neurons)"
    else:
        print("⚠️ No pruning_status column found, using all neurons")
        plot_df_kept = df
        plot_df_pruned = df
        plot_title_suffix = " (All Neurons)"
    
    # Set plot style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'{layer_name} - FC Layer Weight and Gradient Relationship Analysis{plot_title_suffix}', fontsize=18, fontweight='bold')
    
    # Define colors for kept and pruned neurons
    kept_color = 'blue'
    pruned_color = 'red'
    
    # Calculate global axis limits for consistent scaling across all panels
    # FC Weight limits
    fc_weight_min = min(plot_df_kept['fc_weight'].min(), plot_df_pruned['fc_weight'].min())
    fc_weight_max = max(plot_df_kept['fc_weight'].max(), plot_df_pruned['fc_weight'].max())
    
    # FC Weight Gradient limits
    fc_weight_grad_min = min(plot_df_kept['fc_weight_grad'].min(), plot_df_pruned['fc_weight_grad'].min())
    fc_weight_grad_max = max(plot_df_kept['fc_weight_grad'].max(), plot_df_pruned['fc_weight_grad'].max())
    
    # IF Output Gradient limits
    if_output_grad_min = min(plot_df_kept['if_output_grad'].min(), plot_df_pruned['if_output_grad'].min())
    if_output_grad_max = max(plot_df_kept['if_output_grad'].max(), plot_df_pruned['if_output_grad'].max())
    
    # IF Input Gradient limits
    if_input_grad_min = min(plot_df_kept['if_input_grad'].min(), plot_df_pruned['if_input_grad'].min())
    if_input_grad_max = max(plot_df_kept['if_input_grad'].max(), plot_df_pruned['if_input_grad'].max())
    
    # Add small margins to the limits
    margin_factor = 0.05
    fc_weight_range = fc_weight_max - fc_weight_min
    fc_weight_grad_range = fc_weight_grad_max - fc_weight_grad_min
    if_output_grad_range = if_output_grad_max - if_output_grad_min
    if_input_grad_range = if_input_grad_max - if_input_grad_min
    
    fc_weight_min -= fc_weight_range * margin_factor
    fc_weight_max += fc_weight_range * margin_factor
    fc_weight_grad_min -= fc_weight_grad_range * margin_factor
    fc_weight_grad_max += fc_weight_grad_range * margin_factor
    if_output_grad_min -= if_output_grad_range * margin_factor
    if_output_grad_max += if_output_grad_range * margin_factor
    if_input_grad_min -= if_input_grad_range * margin_factor
    if_input_grad_max += if_input_grad_range * margin_factor
    
    print(f"📊 Global axis limits calculated:")
    print(f"   FC Weight: [{fc_weight_min:.6f}, {fc_weight_max:.6f}]")
    print(f"   FC Weight Gradient: [{fc_weight_grad_min:.6f}, {fc_weight_grad_max:.6f}]")
    print(f"   IF Output Gradient: [{if_output_grad_min:.6f}, {if_output_grad_max:.6f}]")
    print(f"   IF Input Gradient: [{if_input_grad_min:.6f}, {if_input_grad_max:.6f}]")
    
    # Row 1: Kept Neurons (pruning_status = 1)
    # 1. FC layer weight vs FC layer weight gradient (Kept)
    axes[0, 0].scatter(plot_df_kept['fc_weight'], plot_df_kept['fc_weight_grad'], alpha=0.6, s=20, color=kept_color, label='Kept')
    axes[0, 0].set_xlabel('FC Layer Weight Value')
    axes[0, 0].set_ylabel('FC Layer Weight Gradient Value')
    axes[0, 0].set_title('FC Weight vs FC Weight Gradient (Kept Neurons)')
    axes[0, 0].grid(True, alpha=0.3)
    # Set consistent axis limits
    axes[0, 0].set_xlim(fc_weight_min, fc_weight_max)
    axes[0, 0].set_ylim(fc_weight_grad_min, fc_weight_grad_max)
    
    # Add trend line
    if len(plot_df_kept) > 1:
        z = np.polyfit(plot_df_kept['fc_weight'], plot_df_kept['fc_weight_grad'], 1)
        p = np.poly1d(z)
        axes[0, 0].plot(plot_df_kept['fc_weight'], p(plot_df_kept['fc_weight']), "r--", alpha=0.8, linewidth=2)
    
    # Add correlation coefficient
    corr = plot_df_kept['fc_weight'].corr(plot_df_kept['fc_weight_grad'])
    axes[0, 0].text(0.05, 0.95, f'Correlation: {corr:.4f}', transform=axes[0, 0].transAxes, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 2. FC layer weight vs IF layer output gradient (Kept)
    axes[0, 1].scatter(plot_df_kept['fc_weight'], plot_df_kept['if_output_grad'], alpha=0.6, s=20, color=kept_color, label='Kept')
    axes[0, 1].set_xlabel('FC Layer Weight Value')
    axes[0, 1].set_ylabel('IF Layer Output Gradient Value')
    axes[0, 1].set_title('FC Weight vs IF Output Gradient (Kept Neurons)')
    axes[0, 1].grid(True, alpha=0.3)
    # Set consistent axis limits
    axes[0, 1].set_xlim(fc_weight_min, fc_weight_max)
    axes[0, 1].set_ylim(if_output_grad_min, if_output_grad_max)
    
    # Add trend line
    if len(plot_df_kept) > 1:
        z = np.polyfit(plot_df_kept['fc_weight'], plot_df_kept['if_output_grad'], 1)
        p = np.poly1d(z)
        axes[0, 1].plot(plot_df_kept['fc_weight'], p(plot_df_kept['fc_weight']), "r--", alpha=0.8, linewidth=2)
    
    # Add correlation coefficient
    corr = plot_df_kept['fc_weight'].corr(plot_df_kept['if_output_grad'])
    axes[0, 1].text(0.05, 0.95, f'Correlation: {corr:.4f}', transform=axes[0, 1].transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 3. FC layer weight vs IF layer input gradient (Kept)
    axes[0, 2].scatter(plot_df_kept['fc_weight'], plot_df_kept['if_input_grad'], alpha=0.6, s=20, color=kept_color, label='Kept')
    axes[0, 2].set_xlabel('FC Layer Weight Value')
    axes[0, 2].set_ylabel('IF Layer Input Gradient Value')
    axes[0, 2].set_title('FC Weight vs IF Input Gradient (Kept Neurons)')
    axes[0, 2].grid(True, alpha=0.3)
    # Set consistent axis limits
    axes[0, 2].set_xlim(fc_weight_min, fc_weight_max)
    axes[0, 2].set_ylim(if_input_grad_min, if_input_grad_max)
    
    # Add trend line
    if len(plot_df_kept) > 1:
        z = np.polyfit(plot_df_kept['fc_weight'], plot_df_kept['if_input_grad'], 1)
        p = np.poly1d(z)
        axes[0, 2].plot(plot_df_kept['fc_weight'], p(plot_df_kept['fc_weight']), "r--", alpha=0.8, linewidth=2)
    
    # Add correlation coefficient
    corr = plot_df_kept['fc_weight'].corr(plot_df_kept['if_input_grad'])
    axes[0, 2].text(0.05, 0.95, f'Correlation: {corr:.4f}', transform=axes[0, 2].transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Row 2: Pruned Neurons (pruning_status = 0)
    # 4. FC layer weight vs FC layer weight gradient (Pruned)
    axes[1, 0].scatter(plot_df_pruned['fc_weight'], plot_df_pruned['fc_weight_grad'], alpha=0.6, s=20, color=pruned_color, label='Pruned')
    axes[1, 0].set_xlabel('FC Layer Weight Value')
    axes[1, 0].set_ylabel('FC Layer Weight Gradient Value')
    axes[1, 0].set_title('FC Weight vs FC Weight Gradient (Pruned Neurons)')
    axes[1, 0].grid(True, alpha=0.3)
    # Set consistent axis limits
    axes[1, 0].set_xlim(fc_weight_min, fc_weight_max)
    axes[1, 0].set_ylim(fc_weight_grad_min, fc_weight_grad_max)
    
    # Add trend line
    if len(plot_df_pruned) > 1:
        z = np.polyfit(plot_df_pruned['fc_weight'], plot_df_pruned['fc_weight_grad'], 1)
        p = np.poly1d(z)
        axes[1, 0].plot(plot_df_pruned['fc_weight'], p(plot_df_pruned['fc_weight']), "r--", alpha=0.8, linewidth=2)
    
    # Add correlation coefficient
    corr = plot_df_pruned['fc_weight'].corr(plot_df_pruned['fc_weight_grad'])
    axes[1, 0].text(0.05, 0.95, f'Correlation: {corr:.4f}', transform=axes[1, 0].transAxes, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 5. FC layer weight vs IF layer output gradient (Pruned)
    axes[1, 1].scatter(plot_df_pruned['fc_weight'], plot_df_pruned['if_output_grad'], alpha=0.6, s=20, color=pruned_color, label='Pruned')
    axes[1, 1].set_xlabel('FC Layer Weight Value')
    axes[1, 1].set_ylabel('IF Layer Output Gradient Value')
    axes[1, 1].set_title('FC Weight vs IF Output Gradient (Pruned Neurons)')
    axes[1, 1].grid(True, alpha=0.3)
    # Set consistent axis limits
    axes[1, 1].set_xlim(fc_weight_min, fc_weight_max)
    axes[1, 1].set_ylim(if_output_grad_min, if_output_grad_max)
    
    # Add trend line
    if len(plot_df_pruned) > 1:
        z = np.polyfit(plot_df_pruned['fc_weight'], plot_df_pruned['if_output_grad'], 1)
        p = np.poly1d(z)
        axes[1, 1].plot(plot_df_pruned['fc_weight'], p(plot_df_pruned['fc_weight']), "r--", alpha=0.8, linewidth=2)
    
    # Add correlation coefficient
    corr = plot_df_pruned['fc_weight'].corr(plot_df_pruned['if_output_grad'])
    axes[1, 1].text(0.05, 0.95, f'Correlation: {corr:.4f}', transform=axes[1, 1].transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 6. FC layer weight vs IF layer input gradient (Pruned)
    axes[1, 2].scatter(plot_df_pruned['fc_weight'], plot_df_pruned['if_input_grad'], alpha=0.6, s=20, color=pruned_color, label='Pruned')
    axes[1, 2].set_xlabel('FC Layer Weight Value')
    axes[1, 2].set_ylabel('IF Layer Input Gradient Value')
    axes[1, 2].set_title('FC Weight vs IF Input Gradient (Pruned Neurons)')
    axes[1, 2].grid(True, alpha=0.3)
    # Set consistent axis limits
    axes[1, 2].set_xlim(fc_weight_min, fc_weight_max)
    axes[1, 2].set_ylim(if_input_grad_min, if_input_grad_max)
    
    # Add trend line
    if len(plot_df_pruned) > 1:
        z = np.polyfit(plot_df_pruned['fc_weight'], plot_df_pruned['if_input_grad'], 1)
        p = np.poly1d(z)
        axes[1, 2].plot(plot_df_pruned['fc_weight'], p(plot_df_pruned['fc_weight']), "r--", alpha=0.8, linewidth=2)
    
    # Add correlation coefficient
    corr = plot_df_pruned['fc_weight'].corr(plot_df_pruned['if_input_grad'])
    axes[1, 2].text(0.05, 0.95, f'Correlation: {corr:.4f}', transform=axes[1, 2].transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    suffix = "_kept_vs_pruned" if has_pruning_status else "_all_neurons"
    output_path = os.path.join(output_dir, f'{layer_name}_fc_weight_gradient_relationships_6plots{suffix}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 6 scatter plots saved: {output_path}")
    plt.show()
    
    return plot_df_kept


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Analyze FC layer weight and gradient relationships, generate 6-panel visualization plots')
    parser.add_argument('--csv_path', type=str, 
                       default='/root/autodl-tmp/0-ANN2SNN-Allinone/2-ANN_SNN_QCFS-SRP-ccc（动态thre）/weight_gradient_analysis_20251011/fc_if_correlation_classifier.1_classifier.2_20251011_233541_pruned_0.5.csv',
                       help='CSV file path')
    parser.add_argument('--output_dir', type=str, default='weight_gradient_analysis_20251011', help='Output directory')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.csv_path):
        print(f"❌ Error: File does not exist - {args.csv_path}")
        print("💡 Please run the main program first to generate data:")
        print("python 0614get_grad_ccc_20251011.py --mode snn --save_analysis")
        return
    
    # Load data
    df = load_csv_data(args.csv_path)
    if df is None:
        return
    
    # Check if required columns exist
    required_columns = ['fc_weight', 'fc_weight_grad', 'if_output_grad', 'if_input_grad']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"❌ Error: Missing required columns - {missing_columns}")
        print(f"📋 Available columns: {list(df.columns)}")
        return
    
    # Get layer name
    layer_name = Path(args.csv_path).stem.replace('fc_if_correlation_', '').replace('_20251011_215151', '')
    
    print(f"\n🎯 Analyzing layer: {layer_name}")
    print(f"🎯 Creating 6-panel comparison plot (Kept vs Pruned Neurons)")
    
    # Analyze weight and gradient relationships for both kept and pruned neurons
    if 'pruning_status' in df.columns:
        print(f"\n📊 Analyzing kept neurons...")
        kept_results = analyze_weight_gradient_relationships(df, "kept")
        
        print(f"\n📊 Analyzing pruned neurons...")
        pruned_results = analyze_weight_gradient_relationships(df, "pruned")
    else:
        print(f"\n📊 Analyzing all neurons (no pruning status available)...")
        all_results = analyze_weight_gradient_relationships(df, "all")
    
    # Create 6-panel scatter plots
    plot_df = create_scatter_plots(df, args.output_dir, layer_name)
    
    print(f"\n✅ Analysis completed!")
    print(f"📁 6-panel scatter plots saved to: {args.output_dir}")
    print(f"🎯 Analyzed layer: {layer_name}")
    
    # Output key findings
    if 'pruning_status' in df.columns:
        print(f"\n🔍 Key findings comparison:")
        print(f"📊 Kept Neurons ({kept_results['neuron_count']} neurons):")
        print(f"   FC Weight vs FC Weight Gradient correlation: {kept_results['fc_weight_grad_corr']:.4f}")
        print(f"   FC Weight vs IF Output Gradient correlation: {kept_results['fc_if_output_corr']:.4f}")
        print(f"   FC Weight vs IF Input Gradient correlation: {kept_results['fc_if_input_corr']:.4f}")
        
        print(f"\n📊 Pruned Neurons ({pruned_results['neuron_count']} neurons):")
        print(f"   FC Weight vs FC Weight Gradient correlation: {pruned_results['fc_weight_grad_corr']:.4f}")
        print(f"   FC Weight vs IF Output Gradient correlation: {pruned_results['fc_if_output_corr']:.4f}")
        print(f"   FC Weight vs IF Input Gradient correlation: {pruned_results['fc_if_input_corr']:.4f}")
    else:
        print(f"\n🔍 Key findings:")
        print(f"   FC Weight vs FC Weight Gradient correlation: {all_results['fc_weight_grad_corr']:.4f}")
        print(f"   FC Weight vs IF Output Gradient correlation: {all_results['fc_if_output_corr']:.4f}")
        print(f"   FC Weight vs IF Input Gradient correlation: {all_results['fc_if_input_corr']:.4f}")

if __name__ == "__main__":
    main()
