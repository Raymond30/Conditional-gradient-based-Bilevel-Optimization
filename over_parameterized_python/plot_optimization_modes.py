import numpy as np
import matplotlib.pyplot as plt
import os

def get_lower_gradient_norm(metrics, mode_label):
    """Get the appropriate gradient norm based on the optimization mode."""
    if mode_label == 'DBGD' and 'grad_f_norm_values' in metrics:
        return metrics['grad_f_norm_values']
    return metrics['g_norms']

def create_comparison_plot(metrics_list, metric_name, title, ylabel, labels, use_log=True, filename=None):
    """Create a comparison plot for a specific metric across different optimization modes."""
    plt.figure(figsize=(10, 6))
    colors = ['b', 'r', 'g']
    
    for metrics, color, label in zip(metrics_list, colors, labels):
        if metric_name == 'lower_gradient_norm':
            data = get_lower_gradient_norm(metrics, label)
        else:
            data = metrics[metric_name]
        plt.plot(range(1, len(data) + 1), data, 
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
    # Create directory for comparative plots
    os.makedirs('optimization_mode_plots', exist_ok=True)
    
    try:
        # Load metrics data for each optimization mode
        metrics_dbgd = np.load('plots_dbgd_lipschitz/metrics_lipschitz.npy', allow_pickle=True).item()
        metrics_standard = np.load('plots_standard_lipschitz/metrics_lipschitz.npy', allow_pickle=True).item()
        metrics_baseline = np.load('plots_baseline_lipschitz/metrics_lipschitz.npy', allow_pickle=True).item()
        
        metrics_list = [metrics_dbgd, metrics_standard, metrics_baseline]
        labels = ['DBGD', 'Standard', 'Baseline']
        
        # Plot settings
        plt.style.use('seaborn')
        
        # Define plots to create
        plots = [
            {
                'metric': 'main_losses',
                'title': 'Main Loss Comparison Across Optimization Modes',
                'ylabel': 'Main Loss',
                'use_log': True,
                'filename': 'optimization_mode_plots/main_loss_comparison.png'
            },
            {
                'metric': 'aux_losses',
                'title': 'Auxiliary Loss (Lipschitz) Comparison',
                'ylabel': 'Auxiliary Loss',
                'use_log': True,
                'filename': 'optimization_mode_plots/aux_loss_comparison.png'
            },
            {
                'metric': 'lower_gradient_norm',
                'title': 'Lower Gradient Norm Comparison',
                'ylabel': '||grad||²',
                'use_log': True,
                'filename': 'optimization_mode_plots/lower_gradient_norm_comparison.png'
            },
            {
                'metric': 'param_diff_norms',
                'title': 'Parameter Difference Norm Comparison',
                'ylabel': '||x_{k+1} - x_k||²',
                'use_log': True,
                'filename': 'optimization_mode_plots/param_diff_norm_comparison.png'
            }
        ]
        
        # Create all plots
        for plot in plots:
            create_comparison_plot(
                metrics_list,
                plot['metric'],
                plot['title'],
                plot['ylabel'],
                labels,
                plot['use_log'],
                plot['filename']
            )
        
        print("Comparative plots across optimization modes have been created in the 'optimization_mode_plots' directory.")
    
    except FileNotFoundError as e:
        print(f"Error: Could not find metrics file: {e}")
        print("Make sure all three experiments (DBGD, Standard, and Baseline) have completed successfully.")
        print("The metrics files should be in their respective directories:")
        print("  - plots_dbgd_lipschitz/metrics_lipschitz.npy")
        print("  - plots_standard_lipschitz/metrics_lipschitz.npy")
        print("  - plots_baseline_lipschitz/metrics_lipschitz.npy")

if __name__ == "__main__":
    main() 