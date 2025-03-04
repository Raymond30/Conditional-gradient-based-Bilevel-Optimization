import numpy as np
import matplotlib.pyplot as plt
import os

def load_metrics_files():
    """Load all metrics files from the experiments."""
    metrics_files = {
        'DBGD': 'plots_dbgd_lipschitz/metrics_lipschitz.npy',
        'Baseline (w=0.01)': 'plots_baseline_lipschitz_weight0.01/metrics_lipschitz.npy',
        'Baseline (w=0.1)': 'plots_baseline_lipschitz_weight0.1/metrics_lipschitz.npy',
        'Baseline (w=1)': 'plots_baseline_lipschitz_weight1/metrics_lipschitz.npy',
        'Baseline (w=10)': 'plots_baseline_lipschitz_weight10/metrics_lipschitz.npy'
    }
    
    metrics_data = {}
    for label, file_path in metrics_files.items():
        try:
            metrics_data[label] = np.load(file_path, allow_pickle=True).item()
            print(f"Loaded metrics for {label}")
        except FileNotFoundError:
            print(f"Warning: Could not find metrics file for {label} at {file_path}")
    
    return metrics_data

def create_comparison_plot(metrics_data, metric_name, title, ylabel, use_log=True, filename=None):
    """Create a comparison plot for a specific metric across different experiments."""
    plt.figure(figsize=(12, 7))
    colors = ['b', 'r', 'g', 'm', 'c']
    
    for i, (label, metrics) in enumerate(metrics_data.items()):
        if metric_name in metrics:
            data = metrics[metric_name]
            plt.plot(range(1, len(data) + 1), data, 
                    color=colors[i % len(colors)], label=label, linewidth=2)
    
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

def create_loss_trajectory_plot(metrics_data, filename=None):
    """Create a loss trajectory plot comparing all experiments."""
    plt.figure(figsize=(12, 10))
    colors = ['b', 'r', 'g', 'm', 'c']
    markers = ['o', 's', '^', 'D', 'x']
    
    for i, (label, metrics) in enumerate(metrics_data.items()):
        if 'main_losses' in metrics and 'aux_losses' in metrics:
            main_losses = metrics['main_losses']
            aux_losses = metrics['aux_losses']
            
            # Plot with both lines and markers
            plt.plot(main_losses, aux_losses, 
                    color=colors[i % len(colors)], 
                    linestyle='-', 
                    alpha=0.5,
                    linewidth=1)
            
            # Add markers at regular intervals (every 5 epochs)
            interval = max(1, len(main_losses) // 10)
            plt.scatter(main_losses[::interval], aux_losses[::interval], 
                       color=colors[i % len(colors)], 
                       marker=markers[i % len(markers)],
                       s=100, 
                       label=label,
                       alpha=0.8)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Cross Entropy Loss (log scale)')
    plt.ylabel('Lipschitz Loss (log scale)')
    plt.title('Training Trajectory: Cross Entropy Loss vs Lipschitz Loss')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(loc='best')
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def create_accuracy_comparison_plot(metrics_data, filename=None):
    """Create a test accuracy comparison plot."""
    plt.figure(figsize=(12, 7))
    colors = ['b', 'r', 'g', 'm', 'c']
    
    for i, (label, metrics) in enumerate(metrics_data.items()):
        if 'test_accuracies' in metrics:
            data = metrics['test_accuracies']
            plt.plot(range(1, len(data) + 1), data, 
                    color=colors[i % len(colors)], label=label, linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Test Accuracy Comparison')
    plt.grid(True)
    plt.legend()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Create directory for comparative plots
    output_dir = 'baseline_comparison_plots'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load metrics data
    metrics_data = load_metrics_files()
    
    if not metrics_data:
        print("No metrics data found. Please run the experiments first.")
        return
    
    # Plot settings
    plt.style.use('seaborn-v0_8')
    
    # Define plots to create
    plots = [
        {
            'metric': 'main_losses',
            'title': 'Cross Entropy Loss Comparison',
            'ylabel': 'Cross Entropy Loss',
            'use_log': True,
            'filename': f'{output_dir}/cross_entropy_loss_comparison.png'
        },
        {
            'metric': 'aux_losses',
            'title': 'Lipschitz Loss Comparison',
            'ylabel': 'Lipschitz Loss',
            'use_log': True,
            'filename': f'{output_dir}/lipschitz_loss_comparison.png'
        },
        {
            'metric': 'g_norms',
            'title': 'Gradient Norm Comparison',
            'ylabel': '||grad||²',
            'use_log': True,
            'filename': f'{output_dir}/gradient_norm_comparison.png'
        },
        {
            'metric': 'param_diff_norms',
            'title': 'Parameter Update Size Comparison',
            'ylabel': '||x_{k+1} - x_k||²',
            'use_log': True,
            'filename': f'{output_dir}/param_diff_norm_comparison.png'
        },
        {
            'metric': 'train_accuracies',
            'title': 'Training Accuracy Comparison',
            'ylabel': 'Train Accuracy (%)',
            'use_log': False,
            'filename': f'{output_dir}/train_accuracy_comparison.png'
        },
        {
            'metric': 'test_accuracies',
            'title': 'Test Accuracy Comparison',
            'ylabel': 'Test Accuracy (%)',
            'use_log': False,
            'filename': f'{output_dir}/test_accuracy_comparison.png'
        }
    ]
    
    # Create all plots
    for plot in plots:
        create_comparison_plot(
            metrics_data,
            plot['metric'],
            plot['title'],
            plot['ylabel'],
            plot['use_log'],
            plot['filename']
        )
    
    # Create loss trajectory plot
    create_loss_trajectory_plot(metrics_data, f'{output_dir}/loss_trajectory_comparison.png')
    
    print(f"Comparative plots have been created in the '{output_dir}' directory.")

if __name__ == "__main__":
    main()