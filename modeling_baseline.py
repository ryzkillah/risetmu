# File: modeling_baseline_fixed.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import pickle
import json
import time

print("=" * 70)
print("TAHAP 2: BUILDING ML MODELS FOR NIDS (FIXED VERSION)")
print("=" * 70)

start_time = time.time()

# ==================== 1. LOAD SAMPLE DATA ====================
print("\n[1/6] 📂 Loading sample data...")
df = pd.read_csv('CICIDS2017_Sample_50k.csv')
print(f"Data shape: {df.shape}")

# ==================== 2. HANDLE INFINITE VALUES ====================
print("\n[2/6] 🧹 Handling infinite values...")

# Cari kolom label
label_col = None
for col in df.columns:
    if 'label' in col.lower():
        label_col = col
        break

if label_col:
    print(f"Label column: '{label_col}'")
    
    # Buat target variables
    df['is_attack'] = df[label_col].apply(
        lambda x: 0 if 'benign' in str(x).lower() else 1
    )
    
    def simplify_label(label):
        label_str = str(label).lower()
        if 'benign' in label_str:
            return 'Normal'
        elif 'dos' in label_str or 'ddos' in label_str:
            return 'DoS'
        elif 'portscan' in label_str:
            return 'PortScan'
        elif 'brute' in label_str or 'ftp' in label_str or 'ssh' in label_str:
            return 'BruteForce'
        elif 'bot' in label_str:
            return 'Botnet'
        elif 'web' in label_str or 'sql' in label_str or 'xss' in label_str:
            return 'WebAttack'
        else:
            return 'Other'
    
    df['attack_type'] = df[label_col].apply(simplify_label)
    
    print(f"Binary target - Normal: {(df['is_attack']==0).sum()}, Attack: {(df['is_attack']==1).sum()}")
    print(f"Multi-class distribution:\n{df['attack_type'].value_counts()}")

# ==================== 3. CLEAN DATA ====================
print("\n[3/6] 🔧 Cleaning and preparing features...")

# Pilih hanya fitur numerik
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Hapus kolom yang bukan fitur
exclude_cols = ['is_attack', 'attack_type']
if 'source_file' in df.columns:
    exclude_cols.append('source_file')
if label_col:
    exclude_cols.append(label_col)

features = [col for col in numeric_cols if col not in exclude_cols]

print(f"Selected {len(features)} numeric features")

# Pisahkan features
X = df[features]

# HANDLE INFINITE VALUES - SOLUSI UTAMA
print("\nCleaning infinite values...")

# Ganti infinite dengan NaN
X.replace([np.inf, -np.inf], np.nan, inplace=True)

# Cek missing values setelah replace
missing_before = X.isna().sum().sum()
print(f"Missing values after inf replacement: {missing_before}")

# OPTION 1: Hapus baris dengan missing values
if missing_before > 0:
    print(f"Removing rows with missing values...")
    rows_before = len(X)
    X = X.dropna()
    rows_after = len(X)
    print(f"Removed {rows_before - rows_after} rows")
    
    # Update target variables juga
    df_clean = df.loc[X.index].copy()
    y_binary = df_clean['is_attack']
    y_multi = df_clean['attack_type']
else:
    df_clean = df.copy()
    y_binary = df['is_attack']
    y_multi = df['attack_type']

print(f"Clean data shape: X={X.shape}, y_binary={y_binary.shape}")

# Normalisasi fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Feature matrix scaled shape: {X_scaled.shape}")

# ==================== 4. TRAIN-TEST SPLIT ====================
print("\n[4/6] 🎲 Splitting data...")

# Untuk binary classification
X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
    X_scaled, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

print(f"Binary - Train: {X_train_bin.shape[0]}, Test: {X_test_bin.shape[0]}")
print(f"Class distribution (train):")
print(pd.Series(y_train_bin).value_counts(normalize=True).round(3))

# ==================== 5. TRAIN MODELS ====================
print("\n[5/6] 🤖 Training models...")

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Gunakan model yang lebih cepat untuk prototyping
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(
    n_estimators=100, 
    random_state=42, 
    eval_metric='logloss',
    # Hapus use_label_encoder jika tidak perlu
    verbosity=0  # Supaya tidak print warning
)
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train model
    model.fit(X_train_bin, y_train_bin)
    
    # Predict
    y_pred = model.predict(X_test_bin)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_bin, y_pred)
    precision = precision_score(y_test_bin, y_pred, zero_division=0)
    recall = recall_score(y_test_bin, y_pred, zero_division=0)
    f1 = f1_score(y_test_bin, y_pred, zero_division=0)
    
    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'model': model
    }
    
    print(f"  Accuracy : {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall   : {recall:.4f}")
    print(f"  F1-Score : {f1:.4f}")

# ==================== 6. ANALYZE RESULTS ====================
print("\n[6/6] 📊 Analyzing results...")

# Find best model
best_model_name = max(results, key=lambda x: results[x]['f1'])
best_model = results[best_model_name]['model']
best_f1 = results[best_model_name]['f1']
best_acc = results[best_model_name]['accuracy']

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   F1-Score: {best_f1:.4f}")
print(f"   Accuracy: {best_acc:.4f}")

# Check if target is achieved
print(f"\n🎯 TARGET (SINTA 3):")
print(f"   Target Accuracy ≥ 93%: {'✅' if best_acc >= 0.93 else '❌'} {best_acc*100:.2f}%")
print(f"   Target F1-Score ≥ 0.90: {'✅' if best_f1 >= 0.90 else '❌'} {best_f1:.4f}")

# ==================== 7. SAVE EVERYTHING ====================
print("\n💾 Saving models and results...")

# Save models
with open('binary_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('all_models.pkl', 'wb') as f:
    pickle.dump(models, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('feature_names.pkl', 'wb') as f:
    pickle.dump(features, f)

# Save results summary
results_summary = {
    'best_model': best_model_name,
    'best_accuracy': float(best_acc),
    'best_f1_score': float(best_f1),
    'all_results': {k: {'accuracy': float(v['accuracy']), 
                       'f1': float(v['f1'])} for k, v in results.items()},
    'data_info': {
        'n_samples': len(X),
        'n_features': len(features),
        'n_attack': int(y_binary.sum()),
        'n_normal': int(len(y_binary) - y_binary.sum()),
        'attack_ratio': float(y_binary.mean())
    },
    'processing_time': time.time() - start_time
}

with open('results_summary.json', 'w') as f:
    json.dump(results_summary, f, indent=4)

# ==================== 8. VISUALIZE ====================
print("\n📈 Creating visualizations...")

# 1. Model comparison chart
plt.figure(figsize=(10, 6))
model_names = list(results.keys())
accuracies = [results[m]['accuracy'] for m in model_names]
f1_scores = [results[m]['f1'] for m in model_names]

x = np.arange(len(model_names))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', color='skyblue')
rects2 = ax.bar(x + width/2, f1_scores, width, label='F1-Score', color='lightcoral')

ax.set_xlabel('Models')
ax.set_ylabel('Score')
ax.set_title('Model Performance Comparison (Binary Classification)')
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3)

# Add value labels
for rects, values in [(rects1, accuracies), (rects2, f1_scores)]:
    for rect, value in zip(rects, values):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Confusion matrix for best model
from sklearn.metrics import ConfusionMatrixDisplay

y_pred_best = best_model.predict(X_test_bin)
cm = confusion_matrix(y_test_bin, y_pred_best)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Attack'])
disp.plot(cmap='Blues')
plt.title(f'Confusion Matrix - {best_model_name}')
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ SAVED FILES:")
print("   1. binary_model.pkl      - Best trained model")
print("   2. all_models.pkl        - All trained models")
print("   3. scaler.pkl            - Feature scaler")
print("   4. feature_names.pkl     - Feature names")
print("   5. results_summary.json  - Results summary")
print("   6. model_comparison.png  - Performance chart")
print("   7. confusion_matrix.png  - Confusion matrix")

print(f"\n⏱️  Total processing time: {time.time() - start_time:.1f} seconds")

print("\n" + "=" * 70)
print("NEXT STEP: RUN XAI INTEGRATION")
print("=" * 70)
print("Command: python xai_integration.py")
print("=" * 70)