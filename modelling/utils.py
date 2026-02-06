# modelling/utils.py
import pandas as pd
import numpy as np
import pickle
import json
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_dataset(dataset_name='cicids2017', sample_size=None):
    """
    Load dataset from processed folder
    """
    dataset_paths = {
        'cicids2017': 'data/processed/CICIDS2017_Full.csv',
        'cicids_sample': 'data/processed/CICIDS2017_Sample_50k.csv',
        'unsw_nb15': 'data/processed/UNSW-NB15_Full.csv',
        'unsw_sample': 'data/processed/UNSW-NB15_Sample_200k.csv'
    }
    
    if dataset_name not in dataset_paths:
        raise ValueError(f"Dataset {dataset_name} not available. Choose from {list(dataset_paths.keys())}")
    
    print(f"📂 Loading {dataset_name}...")
    
    if sample_size:
        df = pd.read_csv(dataset_paths[dataset_name], nrows=sample_size)
        print(f"   Loaded {len(df):,} samples (sampled)")
    else:
        df = pd.read_csv(dataset_paths[dataset_name])
        print(f"   Loaded {len(df):,} samples")
    
    return df

def prepare_binary_classification(df, label_column=None):
    """
    Prepare data for binary classification
    """
    # Find label column if not specified
    if label_column is None:
        label_candidates = [col for col in df.columns if 'label' in col.lower() or 'attack' in col.lower()]
        label_column = label_candidates[0] if label_candidates else None
    
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in dataframe")
    
    # Prepare features (numeric columns only)
    exclude_cols = [label_column, 'attack_cat', 'source_file', 'is_attack']
    exclude_cols = [col for col in exclude_cols if col in df.columns]
    
    features = [col for col in df.columns 
                if col not in exclude_cols 
                and df[col].dtype in [np.int64, np.float64, np.int32, np.float32]]
    
    X = df[features].copy()
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Prepare binary target
    y = df[label_column]
    
    # Ensure binary (0/1)
    if set(y.unique()) != {0, 1}:
        print(f"   Converting to binary (0=Normal, 1=Attack)")
        y = y.apply(lambda x: 0 if x == 0 or str(x).lower() in ['normal', 'benign', '0'] else 1)
    
    print(f"   Features: {len(features)}, Samples: {len(X)}")
    print(f"   Attack ratio: {y.mean():.2%}")
    
    return X, y, features

def save_model_artifacts(model, scaler, features, dataset_name, model_name):
    """
    Save model artifacts to results folder
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f"results/models/{dataset_name}_{timestamp}"
    
    # Create directory if not exists
    os.makedirs(model_dir, exist_ok=True)
    
    # Save artifacts
    model_path = os.path.join(model_dir, f"{model_name}.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    features_path = os.path.join(model_dir, "features.pkl")
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    with open(features_path, 'wb') as f:
        pickle.dump(features, f)
    
    print(f"💾 Model artifacts saved to: {model_dir}")
    
    return model_dir

def load_model_artifacts(model_path):
    """
    Load model artifacts
    """
    model_dir = os.path.dirname(model_path)
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(os.path.join(model_dir, "scaler.pkl"), 'rb') as f:
        scaler = pickle.load(f)
    
    with open(os.path.join(model_dir, "features.pkl"), 'rb') as f:
        features = pickle.load(f)
    
    return model, scaler, features