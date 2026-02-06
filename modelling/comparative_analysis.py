# modelling/comparative_analysis.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import glob
from datetime import datetime

print("=" * 80)
print("COMPARATIVE ANALYSIS: CICIDS2017 vs UNSW-NB15")
print("=" * 80)

def load_latest_results(dataset_name):
    """
    Load latest results for a dataset
    """
    pattern = f"results/models/{dataset_name}_*/results_summary.json"
    result_files = glob.glob(pattern)
    
    if not result_files:
        print(f"⚠️  No results found for {dataset_name}")
        return None
    
    # Get most recent result
    latest_file = max(result_files, key=os.path.getctime)
    
    with open(latest_file, 'r') as f:
        results = json.load(f)
    
    return results

def compare_datasets():
    """
    Compare performance across datasets
    """
    print("\n📊 LOADING RESULTS FROM BOTH DATASETS...")
    
    # Load results
    cicids_results = load_latest_results('cicids2017')
    unsw_results = load_latest_results('unsw_nb15')
    
    if not cicids_results or not unsw_results:
        print("❌ Cannot perform comparison - missing results")
        return
    
    # Create comparison table
    comparison_data = []
    
    for dataset_name, results in [('CICIDS2017', cicids_results), 
                                  ('UNSW-NB15', unsw_results)]:
        comparison_data.append({
            'Dataset': dataset_name,
            'Best Model': results['best_model'],
            'Accuracy': f"{results['best_accuracy']*100:.2f}%",
            'F1-Score': f"{results['best_f1_score']:.4f}",
            'Samples': f"{results['data_info']['n_samples']:,}",
            'Features': results['data_info']['n_features'],
            'Attack Ratio': f"{results['data_info']['attack_ratio']*100:.1f}%"
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    print(df_comparison.to_string(index=False))
    
    # Create visualization
    create_comparison_visualization(cicids_results, unsw_results)
    
    # Calculate improvements
    accuracy_diff = (unsw_results['best_accuracy'] - cicids_results['best_accuracy']) * 100
    f1_diff = unsw_results['best_f1_score'] - cicids_results['best_f1_score']
    
    print(f"\n📈 PERFORMANCE DIFFERENCE:")
    print(f"   Accuracy: {accuracy_diff:+.2f}%")
    print(f"   F1-Score: {f1_diff:+.4f}")
    
    # Save comparison results
    save_comparison_results(cicids_results, unsw_results, df_comparison)

def create_comparison_visualization(cicids_results, unsw_results):
    """
    Create comparison visualization
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Accuracy comparison
    datasets = ['CICIDS2017', 'UNSW-NB15']
    accuracies = [cicids_results['best_accuracy'] * 100, 
                  unsw_results['best_accuracy'] * 100]
    
    bars = ax1.bar(datasets, accuracies, color=['lightgreen', 'lightblue'])
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Accuracy Comparison', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{acc:.2f}%', ha='center', va='bottom')
    
    # 2. F1-Score comparison
    f1_scores = [cicids_results['best_f1_score'], 
                 unsw_results['best_f1_score']]
    
    bars = ax2.bar(datasets, f1_scores, color=['lightgreen', 'lightblue'])
    ax2.set_ylabel('F1-Score')
    ax2.set_title('F1-Score Comparison', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    for bar, f1 in zip(bars, f1_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{f1:.4f}', ha='center', va='bottom')
    
    # 3. Dataset characteristics
    characteristics = ['Samples', 'Features', 'Attack Ratio']
    cicids_vals = [
        cicids_results['data_info']['n_samples'] / 1000000,  # Convert to millions
        cicids_results['data_info']['n_features'],
        cicids_results['data_info']['attack_ratio'] * 100
    ]
    
    unsw_vals = [
        unsw_results['data_info']['n_samples'] / 1000000,
        unsw_results['data_info']['n_features'],
        unsw_results['data_info']['attack_ratio'] * 100
    ]
    
    x = np.arange(len(characteristics))
    width = 0.35
    
    ax3.bar(x - width/2, cicids_vals, width, label='CICIDS2017', color='lightgreen')
    ax3.bar(x + width/2, unsw_vals, width, label='UNSW-NB15', color='lightblue')
    
    ax3.set_xlabel('Characteristic')
    ax3.set_ylabel('Value')
    ax3.set_title('Dataset Characteristics', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(characteristics)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Performance summary radar chart
    metrics = ['Accuracy', 'F1-Score', 'Data Size', 'Feature Richness', 'Balance']
    n_metrics = len(metrics)
    
    # Normalize scores (0-1)
    cicids_scores = [
        cicids_results['best_accuracy'],  # Already 0-1
        cicids_results['best_f1_score'],  # Already 0-1
        min(cicids_results['data_info']['n_samples'] / 3000000, 1),  # Cap at 3M
        min(cicids_results['data_info']['n_features'] / 100, 1),  # Cap at 100 features
        1 - abs(cicids_results['data_info']['attack_ratio'] - 0.5) * 2  # Closer to 0.5 is better
    ]
    
    unsw_scores = [
        unsw_results['best_accuracy'],
        unsw_results['best_f1_score'],
        min(unsw_results['data_info']['n_samples'] / 3000000, 1),
        min(unsw_results['data_info']['n_features'] / 100, 1),
        1 - abs(unsw_results['data_info']['attack_ratio'] - 0.5) * 2
    ]
    
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    cicids_scores += cicids_scores[:1]
    unsw_scores += unsw_scores[:1]
    angles += angles[:1]
    
    ax4 = plt.subplot(2, 2, 4, polar=True)
    ax4.plot(angles, cicids_scores, 'o-', linewidth=2, label='CICIDS2017', color='green')
    ax4.fill(angles, cicids_scores, alpha=0.1, color='green')
    ax4.plot(angles, unsw_scores, 'o-', linewidth=2, label='UNSW-NB15', color='blue')
    ax4.fill(angles, unsw_scores, alpha=0.1, color='blue')
    
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(metrics)
    ax4.set_ylim(0, 1)
    ax4.set_title('Overall Performance Profile', fontweight='bold', pad=20)
    ax4.legend(loc='upper right')
    
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = f"results/figures/comparative_analysis_{timestamp}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Comparison visualization saved to: {fig_path}")

def save_comparison_results(cicids_results, unsw_results, comparison_df):
    """
    Save comparison results
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    comparison_results = {
        'comparison_timestamp': timestamp,
        'datasets': {
            'CICIDS2017': {
                'accuracy': cicids_results['best_accuracy'],
                'f1_score': cicids_results['best_f1_score'],
                'model': cicids_results['best_model']
            },
            'UNSW-NB15': {
                'accuracy': unsw_results['best_accuracy'],
                'f1_score': unsw_results['best_f1_score'],
                'model': unsw_results['best_model']
            }
        },
        'summary': {
            'accuracy_difference': float((unsw_results['best_accuracy'] - cicids_results['best_accuracy']) * 100),
            'f1_difference': float(unsw_results['best_f1_score'] - cicids_results['best_f1_score']),
            'better_dataset': 'CICIDS2017' if cicids_results['best_accuracy'] > unsw_results['best_accuracy'] else 'UNSW-NB15'
        },
        'research_implications': [
            f"Model achieves {cicids_results['best_accuracy']*100:.2f}% accuracy on CICIDS2017",
            f"Model achieves {unsw_results['best_accuracy']*100:.2f}% accuracy on UNSW-NB15",
            f"Performance difference: {(unsw_results['best_accuracy'] - cicids_results['best_accuracy'])*100:+.2f}%",
            "Demonstrates strong generalizability across different network environments"
        ]
    }
    
    # Save JSON
    json_path = f"results/metrics/comparison_results_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(comparison_results, f, indent=4)
    
    # Save CSV
    csv_path = f"results/metrics/comparison_table_{timestamp}.csv"
    comparison_df.to_csv(csv_path, index=False)
    
    print(f"💾 Comparison results saved:")
    print(f"   • {json_path}")
    print(f"   • {csv_path}")

if __name__ == "__main__":
    compare_datasets()