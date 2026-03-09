#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

def analyze_ddr_ratios():
    """Analyze DDR read ratios and plot 3D visualization"""
    
    script_dir = Path(__file__).parent
    parsed_results_dir = script_dir / "parsed_results"
    combined_csv = parsed_results_dir / "ncu_results_combined.csv"
    
    if not combined_csv.exists():
        print(f"Error: {combined_csv} not found")
        return
    
    # Read the combined results
    df = pd.read_csv(combined_csv)
    
    # Calculate DDR read ratio (actual / theoretical)
    # Use m_ddr_read for analysis
    df['ddr_read_ratio_m'] = df['m_ddr_read_bytes_actual'] / df['ddr_read_bytes_theoretical']
    df['ddr_read_ratio_n'] = df['n_ddr_read_bytes_actual'] / df['ddr_read_bytes_theoretical']
    
    # Use the average of m and n ratios
    df['ddr_read_ratio'] = (df['ddr_read_ratio_m'] + df['ddr_read_ratio_n']) / 2
    
    # Get unique K values sorted
    k_values = sorted(df['K'].unique())
    
    print(f"Found {len(k_values)} K values: {k_values}")
    print(f"\nAnalyzing DDR read ratios...\n")
    
    # Find inflection points for each K value
    inflection_points = {}
    last_normal_points = {}
    
    for k in k_values:
        df_k = df[df['K'] == k].sort_values('M')
        
        # Find the last point where ratio <= 1.05 (within 5% of theoretical)
        normal_threshold = 1.05
        normal_points = df_k[df_k['ddr_read_ratio'] <= normal_threshold]
        
        if len(normal_points) > 0:
            last_normal = normal_points.iloc[-1]
            m_val = int(last_normal['M'])
            n_val = int(last_normal['N'])
            theoretical_bytes = last_normal['ddr_read_bytes_theoretical']
            theoretical_mb = theoretical_bytes / (1024 * 1024)
            
            last_normal_points[k] = {
                'M': m_val,
                'N': n_val,
                'ratio': last_normal['ddr_read_ratio'],
                'theoretical_bytes': theoretical_bytes,
                'theoretical_mb': theoretical_mb
            }
        
        # Find where ratio exceeds 1.1 (10% over theoretical)
        threshold = 1.1
        inflection_idx = df_k[df_k['ddr_read_ratio'] > threshold].index
        
        if len(inflection_idx) > 0:
            inflection_row = df_k.loc[inflection_idx[0]]
            m_val = int(inflection_row['M'])
            n_val = int(inflection_row['N'])
            theoretical_bytes = inflection_row['ddr_read_bytes_theoretical']
            theoretical_mb = theoretical_bytes / (1024 * 1024)
            
            inflection_points[k] = {
                'M': m_val,
                'N': n_val,
                'ratio': inflection_row['ddr_read_ratio'],
                'theoretical_bytes': theoretical_bytes,
                'theoretical_mb': theoretical_mb
            }
            
            # Print both points
            if k in last_normal_points:
                ln = last_normal_points[k]
                print(f"K={k:4d}: Last normal at M={ln['M']:4d}, ratio={ln['ratio']:.3f}, "
                      f"Size={ln['theoretical_mb']:.2f} MB | "
                      f"Jump to M={m_val:4d}, ratio={inflection_row['ddr_read_ratio']:.3f}, "
                      f"Size={theoretical_mb:.2f} MB")
            else:
                print(f"K={k:4d}: Inflection at M={m_val:4d}, N={n_val:4d}, "
                      f"ratio={inflection_row['ddr_read_ratio']:.3f}, "
                      f"Theoretical Read Size={theoretical_mb:.2f} MB")
        else:
            if k in last_normal_points:
                ln = last_normal_points[k]
                print(f"K={k:4d}: All normal, max at M={ln['M']:4d}, ratio={ln['ratio']:.3f}, Size={ln['theoretical_mb']:.2f} MB")
            else:
                print(f"K={k:4d}: No data")
    
    # Create visualizations
    fig = plt.figure(figsize=(20, 6))
    
    # 1. 3D plot
    ax1 = fig.add_subplot(131, projection='3d')
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(k_values)))
    
    for i, k in enumerate(k_values):
        df_k = df[df['K'] == k].sort_values('M')
        x = df_k['M'].values
        y = np.full_like(x, k)
        z = df_k['ddr_read_ratio'].values
        
        ax1.plot(x, y, z, color=colors[i], linewidth=2, label=f'K={k}')
        
        # Mark last normal point (green)
        if k in last_normal_points:
            ln = last_normal_points[k]
            ax1.scatter([ln['M']], [k], [ln['ratio']], 
                       color='green', s=100, marker='s', zorder=5)
        
        # Mark inflection point (red)
        if k in inflection_points:
            ip = inflection_points[k]
            ax1.scatter([ip['M']], [k], [ip['ratio']], 
                       color='red', s=100, marker='o', zorder=5)
    
    ax1.set_xlabel('M (Matrix Size)', fontsize=10)
    ax1.set_ylabel('K Value', fontsize=10)
    ax1.set_zlabel('DDR Read Ratio (Actual/Theoretical)', fontsize=10)
    ax1.set_title('3D View: DDR Read Ratio vs Matrix Size and K', fontsize=12)
    ax1.view_init(elev=20, azim=45)
    
    # Add a plane at z=1
    xx, yy = np.meshgrid(df['M'].unique(), k_values)
    zz = np.ones_like(xx)
    ax1.plot_surface(xx, yy, zz, alpha=0.2, color='gray')
    
    # 2. 2D plot - All K curves
    ax2 = fig.add_subplot(132)
    
    for i, k in enumerate(k_values):
        df_k = df[df['K'] == k].sort_values('M')
        ax2.plot(df_k['M'], df_k['ddr_read_ratio'], 
                color=colors[i], linewidth=2, marker='o', markersize=4, label=f'K={k}')
        
        # Mark last normal point (green square)
        if k in last_normal_points:
            ln = last_normal_points[k]
            ax2.scatter([ln['M']], [ln['ratio']], 
                       color='green', s=100, marker='s', zorder=5)
        
        # Mark inflection point (red star)
        if k in inflection_points:
            ip = inflection_points[k]
            ax2.scatter([ip['M']], [ip['ratio']], 
                       color='red', s=100, marker='*', zorder=5)
    
    ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Ideal (1.0x)')
    ax2.axhline(y=1.05, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Normal (1.05x)')
    ax2.axhline(y=1.1, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Inflection (1.1x)')
    ax2.set_xlabel('M (Matrix Size)', fontsize=10)
    ax2.set_ylabel('DDR Read Ratio (Actual/Theoretical)', fontsize=10)
    ax2.set_title('2D View: DDR Read Ratio for All K Values', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # 3. Last normal points vs inflection points plot
    ax3 = fig.add_subplot(133)
    
    if last_normal_points:
        k_list = list(last_normal_points.keys())
        m_list = [last_normal_points[k]['M'] for k in k_list]
        
        ax3.plot(k_list, m_list, 's-', color='green', linewidth=2, markersize=8, label='Last Normal (≤1.05x)')
        
    if inflection_points:
        k_list = list(inflection_points.keys())
        m_list = [inflection_points[k]['M'] for k in k_list]
        
        ax3.plot(k_list, m_list, 'o-', color='red', linewidth=2, markersize=8, label='Inflection (>1.1x)')
        
    ax3.set_xlabel('K Value', fontsize=10)
    ax3.set_ylabel('M (Matrix Size)', fontsize=10)
    ax3.set_title('Last Normal vs Inflection Points', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=9)
    
    plt.tight_layout()
    
    # Save figure
    output_file = parsed_results_dir / "ddr_ratio_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_file}")
    
    # Create separate detailed plots for each K value
    fig2, axes = plt.subplots(3, 5, figsize=(20, 12))
    axes = axes.flatten()
    
    for i, k in enumerate(k_values):
        if i >= len(axes):
            break
            
        ax = axes[i]
        df_k = df[df['K'] == k].sort_values('M')
        
        ax.plot(df_k['M'], df_k['ddr_read_ratio_m'], 
               'o-', linewidth=2, markersize=4, label='kernel_m', color='blue')
        ax.plot(df_k['M'], df_k['ddr_read_ratio_n'], 
               'o-', linewidth=2, markersize=4, label='kernel_n', color='green')
        ax.plot(df_k['M'], df_k['ddr_read_ratio'], 
               'o-', linewidth=2, markersize=4, label='average', color='red')
        
        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.axhline(y=1.05, color='green', linestyle='--', linewidth=1, alpha=0.3)
        ax.axhline(y=1.1, color='red', linestyle='--', linewidth=1, alpha=0.3)
        
        # Mark last normal point
        if k in last_normal_points:
            ln = last_normal_points[k]
            ax.axvline(x=ln['M'], color='green', linestyle=':', linewidth=2, alpha=0.5)
        
        # Mark inflection point
        if k in inflection_points:
            ip = inflection_points[k]
            ax.axvline(x=ip['M'], color='red', linestyle=':', linewidth=2, alpha=0.5)
        
        ax.set_xlabel('M', fontsize=9)
        ax.set_ylabel('Ratio', fontsize=9)
        ax.set_title(f'K={k}', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    
    plt.tight_layout()
    
    output_file2 = parsed_results_dir / "ddr_ratio_detailed.png"
    plt.savefig(output_file2, dpi=300, bbox_inches='tight')
    print(f"Detailed plots saved to: {output_file2}")
    
    # Save last normal points to CSV
    if last_normal_points:
        last_normal_df = pd.DataFrame([
            {
                'K': k, 
                'Last_Normal_M': v['M'], 
                'Last_Normal_N': v['N'],
                'Ratio': v['ratio'],
                'Theoretical_Read_Bytes': v['theoretical_bytes'],
                'Theoretical_Read_MB': v['theoretical_mb']
            }
            for k, v in last_normal_points.items()
        ])
        last_normal_csv = parsed_results_dir / "last_normal_points.csv"
        last_normal_df.to_csv(last_normal_csv, index=False)
        print(f"\nLast normal points saved to: {last_normal_csv}")
        
        print("\n" + "="*80)
        print("LAST NORMAL POINTS (ratio <= 1.05)")
        print("="*80)
        print(f"{'K':<6} {'M':<6} {'N':<6} {'Theoretical Read (MB)':<25} {'Ratio':<8}")
        print("-"*80)
        for k in sorted(last_normal_points.keys()):
            ln = last_normal_points[k]
            print(f"{k:<6} {ln['M']:<6} {ln['N']:<6} {ln['theoretical_mb']:<25.2f} {ln['ratio']:<8.3f}")
        
        # Calculate statistics
        avg_size_mb = last_normal_df['Theoretical_Read_MB'].mean()
        median_size_mb = last_normal_df['Theoretical_Read_MB'].median()
        min_size_mb = last_normal_df['Theoretical_Read_MB'].min()
        max_size_mb = last_normal_df['Theoretical_Read_MB'].max()
        
        print(f"\nAverage theoretical read size at last normal: {avg_size_mb:.2f} MB")
        print(f"Median theoretical read size at last normal: {median_size_mb:.2f} MB")
        print(f"Min theoretical read size: {min_size_mb:.2f} MB")
        print(f"Max theoretical read size: {max_size_mb:.2f} MB")
    
    # Save inflection points to CSV (for comparison)
    if inflection_points:
        inflection_df = pd.DataFrame([
            {
                'K': k, 
                'Inflection_M': v['M'], 
                'Inflection_N': v['N'],
                'Ratio_at_inflection': v['ratio'],
                'Theoretical_Read_Bytes': v['theoretical_bytes'],
                'Theoretical_Read_MB': v['theoretical_mb']
            }
            for k, v in inflection_points.items()
        ])
        inflection_csv = parsed_results_dir / "inflection_points.csv"
        inflection_df.to_csv(inflection_csv, index=False)
        print(f"\nInflection points (ratio > 1.1) saved to: {inflection_csv}")
    
    plt.show()


if __name__ == "__main__":
    analyze_ddr_ratios()
