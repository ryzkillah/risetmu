# modelling/unsw_nb15_modeling.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import json
import time

from modelling.utils import load_dataset, prepare_binary_classification, save_model_artifacts

print("=" * 80)
print("MODELING FOR UNSW-NB15 DATASET")
print("=" * 80)

def train_unsw_nb15_model(sample_size=None):
    """
    Train model on UNSW-NB15 dataset
    """
    start_time = time.time()
    
    # Load dataset
    df = load_dataset('unsw_nb15', sample_size)
    
    # Prepare data - UNSW-NB15 uses 'label' column
    X, y, features = prepare_binary_classification(df, label_column='label')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Normalize
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\n🎯 DATA SPLIT:")
    print(f"   Train: {X_train.shape[0]:,} samples")
    print(f"   Test:  {X_test.shape[0]:,} samples")
    print(f"   Features: {X_train.shape[1]}")
    print(f"   Attack ratio: {y.mean():.2%}")
    
    # Define models
    models = {
        'XGBoost': xgb.XGBClassifier(
            n_estimators=100,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False,
            verbosity=0
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
    }
    
    # Train and evaluate
    results = {}
    
    for name, model in models.items():
        print(f"\n🤖 Training {name}...")
        
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'model': model
        }
        
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   F1-Score:  {f1:.4f}")
    
    # Find best model
    best_model_name = max(results, key=lambda x: results[x]['f1'])
    best_model = results[best_model_name]['model']
    best_acc = results[best_model_name]['accuracy']
    best_f1 = results[best_model_name]['f1']
    
    print(f"\n🏆 BEST MODEL: {best_model_name}")
    print(f"   Accuracy: {best_acc*100:.2f}%")
    print(f"   F1-Score: {best_f1:.4f}")
    
    # Save artifacts
    model_dir = save_model_artifacts(
        model=best_model,
        scaler=scaler,
        features=features,
        dataset_name='unsw_nb15',
        model_name=best_model_name.lower().replace(' ', '_')
    )
    
    # Save results
    results_summary = {
        'dataset': 'UNSW-NB15',
        'best_model': best_model_name,
        'best_accuracy': float(best_acc),
        'best_f1_score': float(best_f1),
        'all_results': {k: {'accuracy': float(v['accuracy']), 'f1': float(v['f1'])} 
                       for k, v in results.items()},
        'data_info': {
            'n_samples': len(df),
            'n_features': len(features),
            'n_attack': int(y.sum()),
            'n_normal': int(len(y) - y.sum()),
            'attack_ratio': float(y.mean())
        },
        'processing_time': time.time() - start_time
    }
    
    results_path = os.path.join(model_dir, 'results_summary.json')
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=4)
    
    # Create visualizations
    create_visualizations(results, y_test, best_model.predict(X_test_scaled), 
                         best_model_name, 'UNSW-NB15', model_dir)
    
    print(f"\n✅ UNSW-NB15 MODELING COMPLETE!")
    print(f"⏱️  Total time: {time.time() - start_time:.1f} seconds")
    
    return results_summary

def create_visualizations(results, y_true, y_pred_best, best_model_name, 
                         dataset_name, save_dir):
    """
    Create visualization plots
    """
    # 1. Model comparison
    plt.figure(figsize=(10, 6))
    
    model_names = list(results.keys())
    accuracies = [results[m]['accuracy'] * 100 for m in model_names]
    f1_scores = [results[m]['f1'] * 100 for m in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    plt.bar(x - width/2, accuracies, width, label='Accuracy', color='skyblue')
    plt.bar(x + width/2, f1_scores, width, label='F1-Score', color='lightcoral')
    
    plt.xlabel('Models')
    plt.ylabel('Score (%)')
    plt.title(f'Model Performance on {dataset_name}', fontweight='bold')
    plt.xticks(x, model_names)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (acc, f1) in enumerate(zip(accuracies, f1_scores)):
        plt.text(i - width/2, acc + 0.5, f'{acc:.1f}%', ha='center', va='bottom')
        plt.text(i + width/2, f1 + 0.5, f'{f1:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_comparison.png'), dpi=300)
    plt.close()
    
    # 2. Confusion matrix
    from sklearn.metrics import ConfusionMatrixDisplay
    
    cm = confusion_matrix(y_true, y_pred_best)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Attack'])
    disp.plot(cmap='Blues')
    plt.title(f'Confusion Matrix - {best_model_name} on {dataset_name}', fontweight='bold')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()
    
    print(f"📊 Visualizations saved to: {save_dir}")

if __name__ == "__main__":
    # Run with sample data for quick testing
    print("Running with sample data (200k samples)...")
    results = train_unsw_nb15_model(sample_size=200000)
    
    # Uncomment for full dataset
    # results = train_unsw_nb15_model()