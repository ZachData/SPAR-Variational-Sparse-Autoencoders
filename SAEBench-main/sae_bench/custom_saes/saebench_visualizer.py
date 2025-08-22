import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import pandas as pd

def load_saebench_results(json_files):
    """Load multiple SAEBench result files"""
    results = []
    for file_path in json_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            results.append(data)
    return results

def extract_metrics(results):
    """Extract key metrics from SAEBench results"""
    metrics_data = []
    
    for result in results:
        # Extract SAE info from sae_lens_release_id
        release_id = result.get('sae_lens_release_id', '')
        eval_type = result.get('eval_type_id', 'unknown')
        
        # Parse model type and k value from release_id
        # Check for vSAE/VSAE in both release_id and potential filename patterns
        is_vsae = ('vSAE' in release_id or 'VSAE' in release_id or 
                   'VSAETopK' in release_id or 'vsae' in release_id.lower())
        
        if 'TopK' in release_id or 'topk' in release_id.lower():
            model_type = 'vSAE TopK' if is_vsae else 'TopK'
            # Extract k value - try multiple patterns
            k_value = 64  # default
            if '_k' in release_id:
                k_match = release_id.split('_k')[1].split('_')[0]
                try:
                    k_value = int(k_match)
                except:
                    k_value = 64
            elif 'k64' in release_id:
                k_value = 64
            elif 'k128' in release_id:
                k_value = 128
            elif 'k256' in release_id:
                k_value = 256
            elif 'k512' in release_id:
                k_value = 512
        else:
            model_type = 'Unknown'
            k_value = 0
            
        metrics = result['eval_result_metrics']
        
        row = {
            'model_type': model_type,
            'k_value': k_value,
            'release_id': release_id,
            'eval_type': eval_type,
        }
        
        # Handle different evaluation types
        if eval_type == 'core':
            row.update({
                # Model behavior preservation
                'kl_div_score': metrics['model_behavior_preservation']['kl_div_score'],
                'kl_div_with_sae': metrics['model_behavior_preservation']['kl_div_with_sae'],
                
                # Model performance preservation  
                'ce_loss_score': metrics['model_performance_preservation']['ce_loss_score'],
                'ce_loss_with_sae': metrics['model_performance_preservation']['ce_loss_with_sae'],
                'ce_loss_without_sae': metrics['model_performance_preservation']['ce_loss_without_sae'],
                
                # Reconstruction quality
                'explained_variance': metrics['reconstruction_quality']['explained_variance'],
                'mse': metrics['reconstruction_quality']['mse'],
                'cossim': metrics['reconstruction_quality']['cossim'],
                
                # Sparsity
                'l0': metrics['sparsity']['l0'],
                'l1': metrics['sparsity']['l1'],
                
                # Shrinkage
                'l2_ratio': metrics['shrinkage']['l2_ratio'],
                
                # Misc metrics
                'frac_alive': metrics['misc_metrics']['frac_alive'],
                'freq_over_1_percent': metrics['misc_metrics']['freq_over_1_percent'],
            })
            
        elif eval_type == 'scr':
            scr_metrics = metrics['scr_metrics']
            # Extract SCR metrics for different thresholds
            thresholds = [2, 5, 10, 20, 50, 100, 500]
            for threshold in thresholds:
                row[f'scr_metric_threshold_{threshold}'] = scr_metrics[f'scr_metric_threshold_{threshold}']
                row[f'scr_dir1_threshold_{threshold}'] = scr_metrics[f'scr_dir1_threshold_{threshold}']
                row[f'scr_dir2_threshold_{threshold}'] = scr_metrics[f'scr_dir2_threshold_{threshold}']
                
        elif eval_type == 'sparse_probing':
            # Extract sparse probing metrics
            llm_metrics = metrics['llm']
            sae_metrics = metrics['sae']
            
            # Overall accuracies
            row['llm_test_accuracy'] = llm_metrics['llm_test_accuracy']
            row['sae_test_accuracy'] = sae_metrics['sae_test_accuracy']
            
            # Top-k accuracies
            for k in [1, 2, 5]:
                row[f'llm_top_{k}_test_accuracy'] = llm_metrics[f'llm_top_{k}_test_accuracy']
                row[f'sae_top_{k}_test_accuracy'] = sae_metrics[f'sae_top_{k}_test_accuracy']
            
            # Calculate performance preservation ratios
            row['accuracy_preservation'] = sae_metrics['sae_test_accuracy'] / llm_metrics['llm_test_accuracy']
            for k in [1, 2, 5]:
                if llm_metrics[f'llm_top_{k}_test_accuracy'] > 0:
                    row[f'top_{k}_preservation'] = sae_metrics[f'sae_top_{k}_test_accuracy'] / llm_metrics[f'llm_top_{k}_test_accuracy']
                else:
                    row[f'top_{k}_preservation'] = 0
                    
        elif eval_type == 'tpp':
            tpp_metrics = metrics['tpp_metrics']
            # Extract TPP metrics for different thresholds
            thresholds = [2, 5, 10, 20, 50, 100, 500]
            for threshold in thresholds:
                row[f'tpp_threshold_{threshold}_total_metric'] = tpp_metrics[f'tpp_threshold_{threshold}_total_metric']
                row[f'tpp_threshold_{threshold}_intended_diff_only'] = tpp_metrics[f'tpp_threshold_{threshold}_intended_diff_only']
                row[f'tpp_threshold_{threshold}_unintended_diff_only'] = tpp_metrics[f'tpp_threshold_{threshold}_unintended_diff_only']
        
        metrics_data.append(row)
    
    return pd.DataFrame(metrics_data)

def create_comparison_plots(df, save_path='saebench_plots'):
    """Create comprehensive comparison plots for core metrics"""
    core_df = df[df['eval_type'] == 'core'] if 'eval_type' in df.columns else df
    
    if core_df.empty:
        print("No core evaluation data found for comparison plots")
        return
        
    Path(save_path).mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    colors = ['#2E86AB', '#A23B72']  # Blue for TopK, Pink for vSAE TopK
    
    # Key metrics to plot
    metrics = [
        ('kl_div_score', 'KL Divergence Score', 'Higher is better'),
        ('ce_loss_score', 'CE Loss Score', 'Higher is better'), 
        ('explained_variance', 'Explained Variance', 'Higher is better'),
        ('cossim', 'Cosine Similarity', 'Higher is better'),
        ('l0', 'L0 Norm (Sparsity)', 'Context-dependent'),
        ('frac_alive', 'Fraction Alive Features', 'Higher is better')
    ]
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (metric, title, note) in enumerate(metrics):
        ax = axes[idx]
        
        # Plot for each model type
        for i, model_type in enumerate(['TopK', 'vSAE TopK']):
            data = core_df[core_df['model_type'] == model_type]
            if not data.empty:
                ax.plot(data['k_value'], data[metric], 'o-', 
                       color=colors[i], linewidth=2, markersize=8,
                       label=model_type)
        
        ax.set_xlabel('K Value')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{title}\n({note})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log', base=2)
        ax.set_xticks([64, 128, 256, 512])
        ax.set_xticklabels(['64', '128', '256', '512'])
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/core_comparison_overview.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_pareto_plot(df, save_path='saebench_plots'):
    """Create Pareto frontier plot for sparsity vs reconstruction quality"""
    core_df = df[df['eval_type'] == 'core'] if 'eval_type' in df.columns else df
    
    if core_df.empty:
        print("No core evaluation data found for Pareto plot")
        return
        
    plt.figure(figsize=(12, 8))
    
    colors = {'TopK': '#2E86AB', 'vSAE TopK': '#A23B72'}
    
    for model_type in ['TopK', 'vSAE TopK']:
        data = core_df[core_df['model_type'] == model_type]
        if not data.empty:
            plt.scatter(data['l0'], data['explained_variance'], 
                       c=colors[model_type], s=100, alpha=0.7, 
                       label=model_type, edgecolors='black', linewidth=1)
            
            # Add k value annotations
            for _, row in data.iterrows():
                plt.annotate(f'k={int(row["k_value"])}', 
                           (row['l0'], row['explained_variance']),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=9, alpha=0.8)
    
    plt.xlabel('L0 Norm (Sparsity)')
    plt.ylabel('Explained Variance')
    plt.title('Sparsity vs Reconstruction Quality\nPareto Frontier Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{save_path}/pareto_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

def get_color_for_k(model_type, k_value):
    """Get color based on model type and k value with gradients"""
    # Define color gradients: light to dark as k increases
    if model_type == 'TopK':
        # Blue to teal gradient
        color_map = {64: '#87CEEB', 128: '#4682B4', 256: '#2E86AB', 512: '#008B8B'}
    elif model_type == 'vSAE TopK':
        # Magenta to purple gradient  
        color_map = {64: '#FF69B4', 128: '#DA70D6', 256: '#A23B72', 512: '#8B008B'}
    else:
        return '#999999'  # Default gray for unknown
    
    return color_map.get(k_value, '#999999')

def get_marker_for_model(model_type):
    """Get marker shape based on model type"""
    if model_type == 'TopK':
        return 'o'  # Circle
    elif model_type == 'vSAE TopK':
        return '^'  # Triangle
    else:
        return 's'  # Square for unknown

def create_scr_plots(df, save_path='saebench_plots'):
    """Create SCR-specific visualization plots"""
    scr_df = df[df['eval_type'] == 'scr']
    if scr_df.empty:
        print("No SCR data found for visualization")
        return
    
    Path(save_path).mkdir(exist_ok=True)
    
    thresholds = [2, 5, 10, 20, 50, 100, 500]
    
    # SCR Performance across thresholds
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Overall SCR metric performance
    for model_type in ['TopK', 'vSAE TopK']:
        model_data = scr_df[scr_df['model_type'] == model_type]
        if not model_data.empty:
            for _, row in model_data.iterrows():
                scr_values = [row[f'scr_metric_threshold_{t}'] for t in thresholds]
                ax1.plot(thresholds, scr_values, marker=get_marker_for_model(model_type), linestyle='-',
                        color=get_color_for_k(model_type, row['k_value']), alpha=0.8,
                        linewidth=2, markersize=8,
                        label=f'{model_type} k={int(row["k_value"])}')
    
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('SCR Metric')
    ax1.set_title('SCR Performance Across Thresholds')
    ax1.set_xscale('log', base=2)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Direction 1 performance
    for model_type in ['TopK', 'vSAE TopK']:
        model_data = scr_df[scr_df['model_type'] == model_type]
        if not model_data.empty:
            for _, row in model_data.iterrows():
                dir1_values = [row[f'scr_dir1_threshold_{t}'] for t in thresholds]
                ax2.plot(thresholds, dir1_values, marker=get_marker_for_model(model_type), linestyle='-',
                        color=get_color_for_k(model_type, row['k_value']), alpha=0.8,
                        linewidth=2, markersize=8,
                        label=f'{model_type} k={int(row["k_value"])}')
    
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('SCR Direction 1')
    ax2.set_title('SCR Direction 1 Performance')
    ax2.set_xscale('log', base=2)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Direction 2 performance
    for model_type in ['TopK', 'vSAE TopK']:
        model_data = scr_df[scr_df['model_type'] == model_type]
        if not model_data.empty:
            for _, row in model_data.iterrows():
                dir2_values = [row[f'scr_dir2_threshold_{t}'] for t in thresholds]
                ax3.plot(thresholds, dir2_values, marker=get_marker_for_model(model_type), linestyle='-',
                        color=get_color_for_k(model_type, row['k_value']), alpha=0.8,
                        linewidth=2, markersize=8,
                        label=f'{model_type} k={int(row["k_value"])}')
    
    ax3.set_xlabel('Threshold')
    ax3.set_ylabel('SCR Direction 2')
    ax3.set_title('SCR Direction 2 Performance')
    ax3.set_xscale('log', base=2)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Heatmap of SCR performance by k-value and threshold - FIXED VERSION
    if len(scr_df) > 1:
        pivot_data = []
        for _, row in scr_df.iterrows():
            for threshold in thresholds:
                pivot_data.append({
                    'model_k': f'{row["model_type"]} k={int(row["k_value"])}',
                    'threshold': threshold,
                    'scr_metric': row[f'scr_metric_threshold_{threshold}']
                })
        
        pivot_df = pd.DataFrame(pivot_data)
        
        # Check for and handle duplicates before pivoting
        if pivot_df.duplicated(subset=['model_k', 'threshold']).any():
            print("Warning: Found duplicate entries in SCR data. Taking mean values.")
            # Group by model_k and threshold, take mean of scr_metric
            pivot_df = pivot_df.groupby(['model_k', 'threshold'], as_index=False)['scr_metric'].mean()
        
        heatmap_data = pivot_df.pivot(index='model_k', columns='threshold', values='scr_metric')
        
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                   center=0, ax=ax4, cbar_kws={'label': 'SCR Metric'})
        ax4.set_title('SCR Performance Heatmap')
        ax4.set_xlabel('Threshold')
        ax4.set_ylabel('Model Configuration')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/scr_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_sparse_probing_plots(df, save_path='saebench_plots'):
    """Create sparse probing-specific visualization plots"""
    sp_df = df[df['eval_type'] == 'sparse_probing']
    if sp_df.empty:
        print("No sparse probing data found for visualization")
        return
    
    Path(save_path).mkdir(exist_ok=True)
    
    # Create comprehensive sparse probing analysis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Overall accuracy comparison (SAE vs LLM)
    for model_type in ['TopK', 'vSAE TopK']:
        model_data = sp_df[sp_df['model_type'] == model_type]
        if not model_data.empty:
            for _, row in model_data.iterrows():
                ax1.scatter(row['llm_test_accuracy'], row['sae_test_accuracy'], 
                           c=get_color_for_k(model_type, row['k_value']), s=120, alpha=0.8,
                           marker=get_marker_for_model(model_type),
                           edgecolors='black', linewidth=1,
                           label=f'{model_type} k={int(row["k_value"])}' if _ == model_data.index[0] else "")
                
                # Add k value annotation
                ax1.annotate(f'k={int(row["k_value"])}', 
                           (row['llm_test_accuracy'], row['sae_test_accuracy']),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=9, alpha=0.8)
    
    # Perfect preservation line
    ax1.plot([0.8, 1.0], [0.8, 1.0], 'k--', alpha=0.5, label='Perfect Preservation')
    ax1.set_xlabel('LLM Test Accuracy')
    ax1.set_ylabel('SAE Test Accuracy')
    ax1.set_title('Overall Accuracy: SAE vs LLM Performance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Accuracy preservation by k-value
    k_values = sorted(sp_df['k_value'].unique())
    preservation_metrics = ['accuracy_preservation', 'top_1_preservation', 'top_2_preservation', 'top_5_preservation']
    
    x_pos = np.arange(len(preservation_metrics))
    width = 0.35
    
    for i, model_type in enumerate(['TopK', 'vSAE TopK']):
        model_data = sp_df[sp_df['model_type'] == model_type]
        if not model_data.empty:
            # Average across k-values for this model type
            avg_preservations = []
            for metric in preservation_metrics:
                if metric in model_data.columns:
                    avg_preservations.append(model_data[metric].mean())
                else:
                    avg_preservations.append(0)
            
            # Use the middle k-value color for the bar chart
            middle_k = sorted(model_data['k_value'].unique())[len(model_data['k_value'].unique())//2] if len(model_data['k_value'].unique()) > 0 else 64
            ax2.bar(x_pos + i*width, avg_preservations, width, 
                   color=get_color_for_k(model_type, middle_k), alpha=0.8, label=model_type)
    
    ax2.set_xlabel('Preservation Metric')
    ax2.set_ylabel('Preservation Ratio')
    ax2.set_title('Performance Preservation Comparison')
    ax2.set_xticks(x_pos + width/2)
    ax2.set_xticklabels(['Overall', 'Top-1', 'Top-2', 'Top-5'])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
    
    # 3. Top-k accuracy trends
    for model_type in ['TopK', 'vSAE TopK']:
        model_data = sp_df[sp_df['model_type'] == model_type]
        if not model_data.empty:
            for _, row in model_data.iterrows():
                k_accuracies = [row['sae_top_1_test_accuracy'], 
                               row['sae_top_2_test_accuracy'], 
                               row['sae_top_5_test_accuracy']]
                ax3.plot([1, 2, 5], k_accuracies, marker=get_marker_for_model(model_type), linestyle='-',
                        color=get_color_for_k(model_type, row['k_value']), alpha=0.8,
                        linewidth=2, markersize=8,
                        label=f'{model_type} k={int(row["k_value"])}')
    
    ax3.set_xlabel('Top-K')
    ax3.set_ylabel('SAE Top-K Accuracy')
    ax3.set_title('SAE Top-K Accuracy Trends')
    ax3.set_xticks([1, 2, 5])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Performance degradation analysis
    for model_type in ['TopK', 'vSAE TopK']:
        model_data = sp_df[sp_df['model_type'] == model_type]
        if not model_data.empty:
            degradations = []
            k_vals = []
            colors_list = []
            for _, row in model_data.iterrows():
                # Calculate accuracy degradation
                degradation = row['llm_test_accuracy'] - row['sae_test_accuracy']
                degradations.append(degradation)
                k_vals.append(row['k_value'])
                colors_list.append(get_color_for_k(model_type, row['k_value']))
            
            ax4.scatter(k_vals, degradations, c=colors_list, 
                       s=120, alpha=0.8, label=model_type, 
                       marker=get_marker_for_model(model_type),
                       edgecolors='black', linewidth=1)
    
    ax4.set_xlabel('K Value')
    ax4.set_ylabel('Accuracy Degradation (LLM - SAE)')
    ax4.set_title('Performance Degradation by K Value')
    ax4.set_xscale('log', base=2)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='No Degradation')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/sparse_probing_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create dataset-specific performance breakdown
    create_dataset_breakdown_plot(df, save_path)

def create_dataset_breakdown_plot(df, save_path='saebench_plots'):
    """Create detailed dataset breakdown for sparse probing"""
    sp_df = df[df['eval_type'] == 'sparse_probing']
    if sp_df.empty or 'eval_result_details' not in df.columns:
        return
    
    # This would require access to the detailed results
    # For now, create a summary plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Performance by model type
    model_types = sp_df['model_type'].unique()
    metrics = ['sae_test_accuracy', 'sae_top_1_test_accuracy', 'sae_top_2_test_accuracy', 'sae_top_5_test_accuracy']
    
    x_pos = np.arange(len(metrics))
    width = 0.35
    
    for i, model_type in enumerate(model_types):
        model_data = sp_df[sp_df['model_type'] == model_type]
        if not model_data.empty:
            avg_scores = [model_data[metric].mean() for metric in metrics]
            # Use the middle k-value color for the bar chart
            middle_k = sorted(model_data['k_value'].unique())[len(model_data['k_value'].unique())//2] if len(model_data['k_value'].unique()) > 0 else 64
            ax1.bar(x_pos + i*width, avg_scores, width, 
                   color=get_color_for_k(model_type, middle_k), alpha=0.8, label=model_type)
    
    ax1.set_xlabel('Accuracy Metric')
    ax1.set_ylabel('Average Accuracy')
    ax1.set_title('Average Performance by Model Type')
    ax1.set_xticks(x_pos + width/2)
    ax1.set_xticklabels(['Overall', 'Top-1', 'Top-2', 'Top-5'])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # K-value performance comparison
    k_values = sorted(sp_df['k_value'].unique())
    for model_type in model_types:
        model_data = sp_df[sp_df['model_type'] == model_type]
        if not model_data.empty:
            accuracies = []
            colors_list = []
            for k in k_values:
                k_data = model_data[model_data['k_value'] == k]
                if len(k_data) > 0:
                    accuracies.append(k_data['sae_test_accuracy'].iloc[0])
                    colors_list.append(get_color_for_k(model_type, k))
                else:
                    accuracies.append(0)
                    colors_list.append('#999999')
            
            # Create a line plot with individual colored points
            for i, (k, acc, color) in enumerate(zip(k_values, accuracies, colors_list)):
                if i == 0:  # Add label only for first point
                    ax2.scatter(k, acc, color=color, s=120, alpha=0.8, 
                              marker=get_marker_for_model(model_type),
                              edgecolors='black', linewidth=1, label=model_type)
                else:
                    ax2.scatter(k, acc, color=color, s=120, alpha=0.8, 
                              marker=get_marker_for_model(model_type),
                              edgecolors='black', linewidth=1)
            
            # Connect points with a line using the model type's base color
            base_color = get_color_for_k(model_type, 256)  # Use middle k as base color
            ax2.plot(k_values, accuracies, '--', color=base_color, alpha=0.5, linewidth=1)
    
    ax2.set_xlabel('K Value')
    ax2.set_ylabel('SAE Test Accuracy')
    ax2.set_title('Performance vs K Value')
    ax2.set_xscale('log', base=2)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/sparse_probing_breakdown.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_tpp_plots(df, save_path='saebench_plots'):
    """Create TPP-specific visualization plots"""
    tpp_df = df[df['eval_type'] == 'tpp']
    if tpp_df.empty:
        print("No TPP data found for visualization")
        return
    
    Path(save_path).mkdir(exist_ok=True)
    
    thresholds = [2, 5, 10, 20, 50, 100, 500]
    
    # TPP Performance across thresholds
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Total TPP metric performance
    for model_type in ['TopK', 'vSAE TopK']:
        model_data = tpp_df[tpp_df['model_type'] == model_type]
        if not model_data.empty:
            for _, row in model_data.iterrows():
                tpp_values = [row[f'tpp_threshold_{t}_total_metric'] for t in thresholds]
                ax1.plot(thresholds, tpp_values, marker=get_marker_for_model(model_type), linestyle='-',
                        color=get_color_for_k(model_type, row['k_value']), alpha=0.8,
                        linewidth=2, markersize=8,
                        label=f'{model_type} k={int(row["k_value"])}')
    
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('TPP Total Metric')
    ax1.set_title('TPP Total Performance Across Thresholds')
    ax1.set_xscale('log', base=2)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Intended vs Unintended effects comparison
    for model_type in ['TopK', 'vSAE TopK']:
        model_data = tpp_df[tpp_df['model_type'] == model_type]
        if not model_data.empty:
            for _, row in model_data.iterrows():
                intended_values = [row[f'tpp_threshold_{t}_intended_diff_only'] for t in thresholds]
                unintended_values = [row[f'tpp_threshold_{t}_unintended_diff_only'] for t in thresholds]
                
                color = get_color_for_k(model_type, row['k_value'])
                marker = get_marker_for_model(model_type)
                ax2.plot(thresholds, intended_values, marker=marker, linestyle='-',
                        color=color, alpha=0.9, linewidth=2, markersize=8,
                        label=f'{model_type} k={int(row["k_value"])} (Intended)')
                ax2.plot(thresholds, unintended_values, marker='s', linestyle='--',
                        color=color, alpha=0.6, linewidth=1, markersize=5,
                        label=f'{model_type} k={int(row["k_value"])} (Unintended)')
    
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('TPP Effect Size')
    ax2.set_title('Intended vs Unintended Effects')
    ax2.set_xscale('log', base=2)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Selectivity analysis (intended vs unintended ratio)
    for model_type in ['TopK', 'vSAE TopK']:
        model_data = tpp_df[tpp_df['model_type'] == model_type]
        if not model_data.empty:
            for _, row in model_data.iterrows():
                selectivity_ratios = []
                for t in thresholds:
                    intended = row[f'tpp_threshold_{t}_intended_diff_only']
                    unintended = row[f'tpp_threshold_{t}_unintended_diff_only']
                    # Calculate selectivity as intended/unintended ratio (higher = more selective)
                    if unintended != 0:
                        ratio = intended / abs(unintended)
                    else:
                        ratio = intended if intended > 0 else 0
                    selectivity_ratios.append(ratio)
                
                ax3.plot(thresholds, selectivity_ratios, marker=get_marker_for_model(model_type), linestyle='-',
                        color=get_color_for_k(model_type, row['k_value']), alpha=0.8,
                        linewidth=2, markersize=8,
                        label=f'{model_type} k={int(row["k_value"])}')
    
    ax3.set_xlabel('Threshold')
    ax3.set_ylabel('Selectivity Ratio (Intended/|Unintended|)')
    ax3.set_title('TPP Selectivity Analysis')
    ax3.set_xscale('log', base=2)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Heatmap of TPP total performance by k-value and threshold - FIXED VERSION
    if len(tpp_df) > 1:
        pivot_data = []
        for _, row in tpp_df.iterrows():
            for threshold in thresholds:
                pivot_data.append({
                    'model_k': f'{row["model_type"]} k={int(row["k_value"])}',
                    'threshold': threshold,
                    'tpp_total': row[f'tpp_threshold_{threshold}_total_metric']
                })
        
        pivot_df = pd.DataFrame(pivot_data)
        
        # Check for and handle duplicates before pivoting
        if pivot_df.duplicated(subset=['model_k', 'threshold']).any():
            print("Warning: Found duplicate entries in TPP data. Taking mean values.")
            # Group by model_k and threshold, take mean of tpp_total
            pivot_df = pivot_df.groupby(['model_k', 'threshold'], as_index=False)['tpp_total'].mean()
        
        heatmap_data = pivot_df.pivot(index='model_k', columns='threshold', values='tpp_total')
        
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', 
                   ax=ax4, cbar_kws={'label': 'TPP Total Metric'})
        ax4.set_title('TPP Performance Heatmap')
        ax4.set_xlabel('Threshold')
        ax4.set_ylabel('Model Configuration')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/tpp_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_detailed_table(df, save_path='saebench_plots'):
    """Create detailed comparison tables for core, SCR, and sparse probing metrics"""
    Path(save_path).mkdir(exist_ok=True)
    
    # Core metrics table
    core_df = df[df['eval_type'] == 'core']
    if not core_df.empty:
        table_metrics = ['kl_div_score', 'ce_loss_score', 'explained_variance', 
                        'cossim', 'l0', 'frac_alive']
        
        table_data = []
        for metric in table_metrics:
            row = {'Metric': metric.replace('_', ' ').title()}
            for model_type in ['TopK', 'vSAE TopK']:
                for k in [64, 128, 256, 512]:
                    data_point = core_df[(core_df['model_type'] == model_type) & (core_df['k_value'] == k)]
                    if not data_point.empty:
                        value = data_point[metric].iloc[0]
                        row[f'{model_type} k={k}'] = f'{value:.4f}'
                    else:
                        row[f'{model_type} k={k}'] = 'N/A'
            table_data.append(row)
        
        core_table_df = pd.DataFrame(table_data)
        core_table_df.to_csv(f'{save_path}/core_comparison.csv', index=False)
        print("Core Metrics Comparison Table:")
        print(core_table_df.to_string(index=False))
    
    # SCR metrics table  
    scr_df = df[df['eval_type'] == 'scr']
    if not scr_df.empty:
        thresholds = [2, 5, 10, 20, 50, 100, 500]
        scr_table_data = []
        
        for threshold in thresholds:
            row = {'Threshold': threshold}
            for model_type in ['TopK', 'vSAE TopK']:
                for k in [64, 128, 256, 512]:
                    data_point = scr_df[(scr_df['model_type'] == model_type) & (scr_df['k_value'] == k)]
                    if not data_point.empty:
                        value = data_point[f'scr_metric_threshold_{threshold}'].iloc[0]
                        row[f'{model_type} k={k}'] = f'{value:.4f}'
                    else:
                        row[f'{model_type} k={k}'] = 'N/A'
            scr_table_data.append(row)
        
        scr_table_df = pd.DataFrame(scr_table_data)
        scr_table_df.to_csv(f'{save_path}/scr_comparison.csv', index=False)
        print("\nSCR Metrics Comparison Table:")
        print(scr_table_df.to_string(index=False))
    
    # Sparse probing metrics table
    sp_df = df[df['eval_type'] == 'sparse_probing']
    if not sp_df.empty:
        sp_metrics = ['sae_test_accuracy', 'accuracy_preservation', 
                     'sae_top_1_test_accuracy', 'top_1_preservation',
                     'sae_top_2_test_accuracy', 'top_2_preservation',
                     'sae_top_5_test_accuracy', 'top_5_preservation']
        
        sp_table_data = []
        for metric in sp_metrics:
            if metric in sp_df.columns:
                row = {'Metric': metric.replace('_', ' ').title()}
                for model_type in ['TopK', 'vSAE TopK']:
                    for k in [64, 128, 256, 512]:
                        data_point = sp_df[(sp_df['model_type'] == model_type) & (sp_df['k_value'] == k)]
                        if not data_point.empty:
                            value = data_point[metric].iloc[0]
                            row[f'{model_type} k={k}'] = f'{value:.4f}'
                        else:
                            row[f'{model_type} k={k}'] = 'N/A'
                sp_table_data.append(row)
        
        if sp_table_data:
            sp_table_df = pd.DataFrame(sp_table_data)
            sp_table_df.to_csv(f'{save_path}/sparse_probing_comparison.csv', index=False)
            print("\nSparse Probing Metrics Comparison Table:")
            print(sp_table_df.to_string(index=False))
    
    # TPP metrics table
    tpp_df = df[df['eval_type'] == 'tpp']
    if not tpp_df.empty:
        thresholds = [2, 5, 10, 20, 50, 100, 500]
        tpp_table_data = []
        
        for threshold in thresholds:
            row = {'Threshold': threshold}
            for model_type in ['TopK', 'vSAE TopK']:
                for k in [64, 128, 256, 512]:
                    data_point = tpp_df[(tpp_df['model_type'] == model_type) & (tpp_df['k_value'] == k)]
                    if not data_point.empty:
                        total_value = data_point[f'tpp_threshold_{threshold}_total_metric'].iloc[0]
                        intended_value = data_point[f'tpp_threshold_{threshold}_intended_diff_only'].iloc[0]
                        unintended_value = data_point[f'tpp_threshold_{threshold}_unintended_diff_only'].iloc[0]
                        row[f'{model_type} k={k} Total'] = f'{total_value:.4f}'
                        row[f'{model_type} k={k} Intended'] = f'{intended_value:.4f}'
                        row[f'{model_type} k={k} Unintended'] = f'{unintended_value:.4f}'
                    else:
                        row[f'{model_type} k={k} Total'] = 'N/A'
                        row[f'{model_type} k={k} Intended'] = 'N/A'
                        row[f'{model_type} k={k} Unintended'] = 'N/A'
            tpp_table_data.append(row)
        
        if tpp_table_data:
            tpp_table_df = pd.DataFrame(tpp_table_data)
            tpp_table_df.to_csv(f'{save_path}/tpp_comparison.csv', index=False)
            print("\nTPP Metrics Comparison Table:")
            print(tpp_table_df.to_string(index=False))
        
        return (core_table_df if not core_df.empty else None, 
                scr_table_df if not scr_df.empty else None,
                sp_table_df if sp_table_data else None,
                tpp_table_df if tpp_table_data else None)
    
    return (core_table_df if not core_df.empty else None, 
            scr_table_df if not scr_df.empty else None, 
            sp_table_df if sp_table_data else None,
            None)

# Example usage:
if __name__ == "__main__":
    # Example file paths - replace with your actual file paths
    json_files = [
        # Core evaluation files
        r'eval_results\core\TopK_SAE_gelu-1l_d2048_k64_auxk0.03125_lr_auto_custom_sae_eval_results.json',
        r'eval_results\core\TopK_SAE_gelu-1l_d2048_k128_auxk0.03125_lr_auto_custom_sae_eval_results.json', 
        r'eval_results\core\TopK_SAE_gelu-1l_d2048_k256_auxk0.03125_lr_auto_custom_sae_eval_results.json',
        r'eval_results\core\TopK_SAE_gelu-1l_d2048_k512_auxk0.03125_lr_auto_custom_sae_eval_results.json',
        r'eval_results\core\VSAETopK_gelu-1l_d2048_k64_lr0.0008_kl1.0_aux0.03125_fixed_var_custom_sae_eval_results.json',
        r'eval_results\core\VSAETopK_gelu-1l_d2048_k128_lr0.0008_kl1.0_aux0.03125_fixed_var_custom_sae_eval_results.json',
        r'eval_results\core\VSAETopK_gelu-1l_d2048_k256_lr0.0008_kl1.0_aux0.03125_fixed_var_custom_sae_eval_results.json',
        r'eval_results\core\VSAETopK_gelu-1l_d2048_k512_lr0.0008_kl1.0_aux0.03125_fixed_var_custom_sae_eval_results.json',
        
        # SCR evaluation files
        r'eval_results\scr\scr\TopK_SAE_gelu-1l_d2048_k64_auxk0.03125_lr_auto_custom_sae_eval_results.json',
        r'eval_results\scr\scr\TopK_SAE_gelu-1l_d2048_k128_auxk0.03125_lr_auto_custom_sae_eval_results.json', 
        r'eval_results\scr\scr\TopK_SAE_gelu-1l_d2048_k256_auxk0.03125_lr_auto_custom_sae_eval_results.json',
        r'eval_results\scr\scr\TopK_SAE_gelu-1l_d2048_k512_auxk0.03125_lr_auto_custom_sae_eval_results.json',
        r'eval_results\scr\scr\VSAETopK_gelu-1l_d2048_k64_lr0.0008_kl1.0_aux0.03125_fixed_var_custom_sae_eval_results.json',
        r'eval_results\scr\scr\VSAETopK_gelu-1l_d2048_k128_lr0.0008_kl1.0_aux0.03125_fixed_var_custom_sae_eval_results.json',
        r'eval_results\scr\scr\VSAETopK_gelu-1l_d2048_k256_lr0.0008_kl1.0_aux0.03125_fixed_var_custom_sae_eval_results.json',
        r'eval_results\scr\scr\VSAETopK_gelu-1l_d2048_k512_lr0.0008_kl1.0_aux0.03125_fixed_var_custom_sae_eval_results.json',
        
        # Sparse probing evaluation files
        r'eval_results\sparse_probing\TopK_SAE_gelu-1l_d2048_k64_auxk0.03125_lr_auto_custom_sae_eval_results.json',
        r'eval_results\sparse_probing\TopK_SAE_gelu-1l_d2048_k128_auxk0.03125_lr_auto_custom_sae_eval_results.json', 
        r'eval_results\sparse_probing\TopK_SAE_gelu-1l_d2048_k256_auxk0.03125_lr_auto_custom_sae_eval_results.json',
        r'eval_results\sparse_probing\TopK_SAE_gelu-1l_d2048_k512_auxk0.03125_lr_auto_custom_sae_eval_results.json',
        r'eval_results\sparse_probing\VSAETopK_gelu-1l_d2048_k64_lr0.0008_kl1.0_aux0.03125_fixed_var_custom_sae_eval_results.json',
        r'eval_results\sparse_probing\VSAETopK_gelu-1l_d2048_k128_lr0.0008_kl1.0_aux0.03125_fixed_var_custom_sae_eval_results.json',
        r'eval_results\sparse_probing\VSAETopK_gelu-1l_d2048_k256_lr0.0008_kl1.0_aux0.03125_fixed_var_custom_sae_eval_results.json',
        r'eval_results\sparse_probing\VSAETopK_gelu-1l_d2048_k512_lr0.0008_kl1.0_aux0.03125_fixed_var_custom_sae_eval_results.json',
        
        # TPP evaluation files
        r'eval_results\tpp\tpp\TopK_SAE_gelu-1l_d2048_k64_auxk0.03125_lr_auto_custom_sae_eval_results.json',
        r'eval_results\tpp\tpp\TopK_SAE_gelu-1l_d2048_k128_auxk0.03125_lr_auto_custom_sae_eval_results.json', 
        r'eval_results\tpp\tpp\TopK_SAE_gelu-1l_d2048_k256_auxk0.03125_lr_auto_custom_sae_eval_results.json',
        r'eval_results\tpp\tpp\TopK_SAE_gelu-1l_d2048_k512_auxk0.03125_lr_auto_custom_sae_eval_results.json',
        r'eval_results\tpp\tpp\VSAETopK_gelu-1l_d2048_k64_lr0.0008_kl1.0_aux0.03125_fixed_var_custom_sae_eval_results.json',
        r'eval_results\tpp\tpp\VSAETopK_gelu-1l_d2048_k128_lr0.0008_kl1.0_aux0.03125_fixed_var_custom_sae_eval_results.json',
        r'eval_results\tpp\tpp\VSAETopK_gelu-1l_d2048_k256_lr0.0008_kl1.0_aux0.03125_fixed_var_custom_sae_eval_results.json',
        r'eval_results\tpp\tpp\VSAETopK_gelu-1l_d2048_k512_lr0.0008_kl1.0_aux0.03125_fixed_var_custom_sae_eval_results.json',
    ]
    
    # Load and process results
    results = load_saebench_results(json_files)
    df = extract_metrics(results)
    
    # Create visualizations based on available data
    core_df = df[df['eval_type'] == 'core']
    scr_df = df[df['eval_type'] == 'scr']
    sp_df = df[df['eval_type'] == 'sparse_probing']
    tpp_df = df[df['eval_type'] == 'tpp']
    
    if not core_df.empty:
        print("Creating core evaluation plots...")
        create_comparison_plots(core_df)
        create_pareto_plot(core_df)
    
    if not scr_df.empty:
        print("Creating SCR evaluation plots...")
        create_scr_plots(df)  # Pass full df so function can filter
    
    if not sp_df.empty:
        print("Creating sparse probing evaluation plots...")
        create_sparse_probing_plots(df)  # Pass full df so function can filter
    
    if not tpp_df.empty:
        print("Creating TPP evaluation plots...")
        create_tpp_plots(df)  # Pass full df so function can filter
    
    # Create detailed tables
    create_detailed_table(df)
    
    print(f"Visualization complete! Check the 'saebench_plots' directory for outputs.")
    print("Supported evaluation types: core, SCR, sparse_probing, TPP")