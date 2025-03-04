import numpy as np
import matplotlib.pyplot as plt
import os

def create_comparison_plot(metrics_list, metric_name, title, ylabel, use_log=True, filename=None):
    """Create a comparison plot for a specific metric."""
    plt.figure(figsize=(10, 6))
    colors = ['b', 'r', 'g']
    labels = ['Lipschitz', 'L2 Norm', 'Test Loss']
    
    for metrics, color, label in zip(metrics_list, colors, labels):
        plt.plot(range(1, len(metrics[metric_name]) + 1), metrics[metric_name], 
                color=color, label=label, linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    if use_log:
        plt.yscale('log')
    plt.legend()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Load metrics data for each auxiliary loss type
    metrics_lipschitz = np.load('metrics_lipschitz.npy', allow_pickle=True).item()
    metrics_l2_norm = np.load('metrics_l2_norm.npy', allow_pickle=True).item()
    metrics_test_loss = np.load('metrics_test_loss.npy', allow_pickle=True).item()
    
    metrics_list = [metrics_lipschitz, metrics_l2_norm, metrics_test_loss]
    
    # Create directory for comparative plots
    os.makedirs('comparative_plots', exist_ok=True)
    
    # Plot settings
    plt.style.use('seaborn')
    
    # Define plots to create
    plots = [
        {
            'metric': 'main_losses',
            'title': 'Main Loss Comparison Across Auxiliary Losses',
            'ylabel': 'Main Loss',
            'use_log': True,
            'filename': 'comparative_plots/main_loss_comparison.png'
        },
        {
            'metric': 'aux_losses',
            'title': 'Auxiliary Loss Comparison',
            'ylabel': 'Auxiliary Loss',
            'use_log': True,
            'filename': 'comparative_plots/aux_loss_comparison.png'
        },
        {
            'metric': 'g_norms',
            'title': 'Gradient g Norm Comparison',
            'ylabel': '||grad_g||²',
            'use_log': True,
            'filename': 'comparative_plots/grad_g_norm_comparison.png'
        },
        {
            'metric': 'weight_values',
            'title': 'DBGD Weight Comparison',
            'ylabel': 'Weight',
            'use_log': False,
            'filename': 'comparative_plots/weight_comparison.png'
        },
        {
            'metric': 'param_diff_norms',
            'title': 'Parameter Difference Norm Comparison',
            'ylabel': '||x_{k+1} - x_k||²',
            'use_log': True,
            'filename': 'comparative_plots/param_diff_norm_comparison.png'
        }
    ]
    
    # Create all plots
    for plot in plots:
        create_comparison_plot(
            metrics_list,
            plot['metric'],
            plot['title'],
            plot['ylabel'],
            plot['use_log'],
            plot['filename']
        )
    
    print("Comparative plots have been created in the 'comparative_plots' directory.")

if __name__ == "__main__":
    main() 