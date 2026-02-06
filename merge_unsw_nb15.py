# ============================================
# SCRIPT: GABUNG SEMUA FILE UNSW-NB15 (FIXED VERSION)
# Perbaikan: Handle 'Label' dengan huruf besar & 140 kolom duplicate issue
# ============================================

import pandas as pd
import numpy as np
import os
import glob
import time
import json
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("GABUNG SEMUA FILE UNSW-NB15 - FIXED VERSION")
print("PERBAIKAN: Handle 'Label' (capital L) & column duplication")
print("=" * 70)

# ==================== 1. SETUP PATH ====================
RAW_DATA_PATH = "data/raw/UNSW-NB15"
PROCESSED_DATA_PATH = "data/processed"

# Buat folder jika belum ada
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
os.makedirs(RAW_DATA_PATH, exist_ok=True)

# File output
OUTPUT_FULL = os.path.join(PROCESSED_DATA_PATH, "UNSW-NB15_Full.csv")
OUTPUT_SAMPLE = os.path.join(PROCESSED_DATA_PATH, "UNSW-NB15_Sample_50k.csv")

print(f"📁 Input path: {RAW_DATA_PATH}")
print(f"📁 Output path: {PROCESSED_DATA_PATH}")

# ==================== 2. FIND AND LOAD MAIN FILES ====================
print("\n[1/6] 🔍 Finding and loading main data files...")

# Cari 4 file utama
main_files = []
for i in range(1, 5):
    expected_file = os.path.join(RAW_DATA_PATH, f"UNSW-NB15_{i}.csv")
    if os.path.exists(expected_file):
        main_files.append(expected_file)

if not main_files:
    print("❌ ERROR: Tidak ditemukan file UNSW-NB15_1.csv sampai UNSW-NB15_4.csv")
    exit()

print(f"✅ Found {len(main_files)} main files:")
for i, f in enumerate(main_files, 1):
    file_size = os.path.getsize(f) / (1024*1024)
    print(f"   {i}. {os.path.basename(f):25s} - {file_size:.1f} MB")

# ==================== 3. LOAD FILES DENGAN STRUKTUR YANG SAMA ====================
print("\n[2/6] ⚙️ Loading files with consistent structure...")
start_time = time.time()

all_dataframes = []
column_sets = []  # Untuk tracking kolom di setiap file

for i, file_path in enumerate(main_files, 1):
    file_name = os.path.basename(file_path)
    print(f"  [{i}/{len(main_files)}] {file_name}...", end=" ")
    
    try:
        # Load file
        df = pd.read_csv(file_path, low_memory=False)
        
        # Simpan struktur kolom
        column_sets.append(set(df.columns))
        
        # Standardize column names: Pertahankan case asli TAPI normalize whitespace
        df.columns = df.columns.str.strip()
        
        # Tambah identifier
        df['source_file'] = file_name
        
        all_dataframes.append(df)
        print(f"✅ {df.shape[0]:,} rows, {df.shape[1]} cols")
        
        # Debug: Tampilkan kolom yang mungkin label
        potential_labels = [col for col in df.columns 
                          if any(x in col.lower() for x in ['label', 'attack', 'class'])]
        if potential_labels:
            print(f"       Potential labels: {potential_labels}")
            
    except Exception as e:
        print(f"❌ Error: {str(e)[:80]}")

# ==================== 4. ANALYZE COLUMN STRUCTURE ====================
print("\n[3/6] 📊 Analyzing column structure across files...")

# Cari kolom yang sama di semua file
if column_sets:
    common_columns = set.intersection(*column_sets)
    print(f"✅ Columns common to all files: {len(common_columns)}")
    
    # Tampilkan beberapa kolom umum
    print("   Sample of common columns:")
    for i, col in enumerate(sorted(list(common_columns))[:15], 1):
        print(f"   {i:2d}. {col}")
    
    # Cari kolom yang berbeda
    all_columns = set.union(*column_sets)
    unique_columns = all_columns - common_columns
    if unique_columns:
        print(f"\n⚠️  Columns NOT in all files: {len(unique_columns)}")
        print("   These may cause issues when merging")

# ==================== 5. SMART MERGING WITH COLUMN ALIGNMENT ====================
print(f"\n[4/6] 🔗 Smart merging {len(all_dataframes)} files...")

# Method 1: Coba merge dengan alignment kolom
print("   Method 1: Aligning columns before merge...")

# Buat list semua kolom unik
all_unique_columns = set()
for df in all_dataframes:
    all_unique_columns.update(df.columns)

# Urutkan kolom untuk konsistensi
sorted_columns = sorted(list(all_unique_columns))

print(f"   Total unique columns across all files: {len(sorted_columns)}")

# Align setiap dataframe ke kolom yang sama
aligned_dataframes = []
for df in all_dataframes:
    # Buat dataframe baru dengan semua kolom
    aligned_df = pd.DataFrame(index=df.index)
    
    for col in sorted_columns:
        if col in df.columns:
            aligned_df[col] = df[col]
        else:
            # Kolom tidak ada di file ini, isi dengan NaN
            aligned_df[col] = np.nan
    
    aligned_dataframes.append(aligned_df)

# Gabungkan
full_data = pd.concat(aligned_dataframes, ignore_index=True, sort=False)
print(f"✅ Merged with column alignment: {full_data.shape[0]:,} rows × {full_data.shape[1]} columns")

# ==================== 6. IDENTIFY TARGET COLUMNS - FIXED ====================
print("\n[5/6] 🎯 Identifying target columns (FIXED for UNSW-NB15)...")

# PERBAIKAN PENTING: UNSW-NB15 punya 'Label' (capital L) bukan 'label'
binary_label_col = None
attack_cat_col = None

# Cari dengan case-insensitive tapi pertahankan case asli
for col in full_data.columns:
    col_lower = col.lower()
    
    # Binary label: bisa 'Label', 'label', dll
    if col_lower == 'label':
        binary_label_col = col  # Simpan nama asli (mungkin 'Label')
        print(f"✅ Found binary label column: '{col}' (original case)")
        
    # Attack category: bisa 'attack_cat', 'Attack_cat', dll  
    if col_lower == 'attack_cat':
        attack_cat_col = col
        print(f"✅ Found attack category column: '{col}'")

# Jika tidak ditemukan dengan exact match, cari pattern
if not binary_label_col:
    for col in full_data.columns:
        if 'label' in col.lower() and col.lower() != 'attack_cat':
            binary_label_col = col
            print(f"⚠️  Found potential label column (pattern match): '{col}'")
            break

if not attack_cat_col:
    for col in full_data.columns:
        if 'attack' in col.lower() and 'cat' in col.lower():
            attack_cat_col = col
            print(f"⚠️  Found potential attack category column (pattern match): '{col}'")
            break

# Debug: Tampilkan semua kolom untuk verification
print(f"\n🔍 First 40 columns in dataset:")
for i, col in enumerate(full_data.columns[:40], 1):
    marker = ""
    if col == binary_label_col:
        marker = " ← LABEL"
    elif col == attack_cat_col:
        marker = " ← ATTACK_CAT"
    print(f"   {i:2d}. {col:30s}{marker}")

# Konversi binary label jika ditemukan
if binary_label_col:
    print(f"\n📊 Analyzing binary label column '{binary_label_col}':")
    
    # Tampilkan tipe data dan nilai unik
    print(f"   Data type: {full_data[binary_label_col].dtype}")
    
    unique_vals = full_data[binary_label_col].dropna().unique()
    print(f"   Unique values: {len(unique_vals)}")
    print(f"   Sample values: {unique_vals[:10]}")
    
    # Konversi ke numeric 0/1 jika perlu
    if full_data[binary_label_col].dtype == 'object':
        print(f"   Converting to numeric (0=Normal, 1=Attack)...")
        
        # Cari nilai yang menunjukkan Normal vs Attack
        normal_keywords = ['normal', '0', 'false', 'no', 'benign']
        attack_keywords = ['attack', '1', 'true', 'yes', 'malicious']
        
        def convert_label(x):
            if pd.isna(x):
                return np.nan
            x_str = str(x).lower().strip()
            for kw in normal_keywords:
                if kw in x_str:
                    return 0
            for kw in attack_keywords:
                if kw in x_str:
                    return 1
            # Default: coba parse sebagai numeric
            try:
                val = float(x)
                return 1 if val > 0 else 0
            except:
                return 1  # Default ke attack jika tidak dikenali
        
        full_data[binary_label_col] = full_data[binary_label_col].apply(convert_label)
        
        # Verifikasi konversi
        converted_counts = full_data[binary_label_col].value_counts()
        print(f"   After conversion: {dict(converted_counts)}")

# ==================== 7. CLEAN DATA & HANDLE DUPLICATE COLUMNS ====================
print("\n[6/6] 🧹 Cleaning data and handling issues...")

# 7.1. Remove completely duplicate columns (nama berbeda tapi isi sama)
print("   1. Removing duplicate columns...")

# Cari kolom dengan data yang identik
cols_before = full_data.shape[1]
dup_cols_to_remove = []

for i, col1 in enumerate(full_data.columns):
    for j, col2 in enumerate(full_data.columns[i+1:], i+1):
        try:
            if col1 != col2 and full_data[col1].equals(full_data[col2]):
                # Simpan kolom kedua untuk dihapus
                dup_cols_to_remove.append(col2)
        except:
            continue

if dup_cols_to_remove:
    print(f"   Found {len(dup_cols_to_remove)} duplicate columns to remove")
    full_data = full_data.drop(columns=dup_cols_to_remove)

cols_after = full_data.shape[1]
if cols_before != cols_after:
    print(f"   Removed {cols_before - cols_after} duplicate columns")

# 7.2. Handle missing values
print("   2. Handling missing values...")
missing_before = full_data.isnull().sum().sum()

if missing_before > 0:
    print(f"   Found {missing_before:,} missing values")
    
    # Untuk kolom numeric
    numeric_cols = full_data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if full_data[col].isnull().any():
            median_val = full_data[col].median()
            full_data[col] = full_data[col].fillna(median_val)
    
    # Untuk kolom categorical
    categorical_cols = full_data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if full_data[col].isnull().any():
            if not full_data[col].mode().empty:
                mode_val = full_data[col].mode()[0]
                full_data[col] = full_data[col].fillna(mode_val)
            else:
                full_data[col] = full_data[col].fillna('Unknown')
    
    missing_after = full_data.isnull().sum().sum()
    print(f"   After cleaning: {missing_after:,} missing values")

# 7.3. Handle infinite values
print("   3. Handling infinite values...")
numeric_cols = full_data.select_dtypes(include=[np.number]).columns

# Cek dan handle infinite values
for col in numeric_cols:
    if np.isinf(full_data[col]).any():
        # Replace inf dengan NaN lalu fill dengan median
        full_data[col] = full_data[col].replace([np.inf, -np.inf], np.nan)
        full_data[col] = full_data[col].fillna(full_data[col].median())

print(f"   ✅ Final clean data: {full_data.shape[0]:,} rows × {full_data.shape[1]} columns")

# ==================== 8. ANALYZE AND SAVE ====================
print("\n📊 Final dataset analysis...")

# Basic stats
print(f"• Total samples: {full_data.shape[0]:,}")
print(f"• Total features: {full_data.shape[1]}")
print(f"• Memory usage: {full_data.memory_usage(deep=True).sum() / (1024**2):.1f} MB")

# Label analysis
if binary_label_col:
    label_counts = full_data[binary_label_col].value_counts()
    print(f"\n🎯 BINARY LABEL DISTRIBUTION:")
    for val, count in label_counts.items():
        pct = count / len(full_data) * 100
        label_name = "Normal" if val == 0 else "Attack"
        print(f"   {label_name}: {count:,} samples ({pct:.1f}%)")

if attack_cat_col:
    attack_counts = full_data[attack_cat_col].value_counts()
    print(f"\n🎯 ATTACK CATEGORIES (Top 10):")
    for i, (cat, count) in enumerate(attack_counts.head(10).items(), 1):
        pct = count / len(full_data) * 100
        print(f"   {i:2d}. {cat:20s}: {count:>7,} ({pct:5.1f}%)")

# ==================== 9. SAVE DATA ====================
print("\n💾 Saving processed data...")

# Save full dataset
print(f"   1. Saving full dataset -> {os.path.basename(OUTPUT_FULL)}")
full_data.to_csv(OUTPUT_FULL, index=False)
full_size = os.path.getsize(OUTPUT_FULL) / (1024*1024)
print(f"      Size: {full_size:.1f} MB")

# Create and save 50k sample
print(f"   2. Creating 50k sample -> {os.path.basename(OUTPUT_SAMPLE)}")
sample_size = 50000

if len(full_data) >= sample_size:
    if binary_label_col:
        # Stratified sampling
        print(f"      Using stratified sampling...")
        
        # Hitung samples per class
        n_normal = int(sample_size * (full_data[binary_label_col] == 0).mean())
        n_attack = sample_size - n_normal
        
        # Sample dari setiap kelas
        normal_samples = full_data[full_data[binary_label_col] == 0].sample(
            n=min(n_normal, (full_data[binary_label_col] == 0).sum()), 
            random_state=42)
        attack_samples = full_data[full_data[binary_label_col] == 1].sample(
            n=min(n_attack, (full_data[binary_label_col] == 1).sum()), 
            random_state=42)
        
        sample_data = pd.concat([normal_samples, attack_samples]).sample(
            frac=1, random_state=42)  # Shuffle
    else:
        # Random sampling
        sample_data = full_data.sample(n=sample_size, random_state=42)
    
    sample_data.to_csv(OUTPUT_SAMPLE, index=False)
    sample_size_mb = os.path.getsize(OUTPUT_SAMPLE) / (1024*1024)
    print(f"      Size: {sample_size_mb:.1f} MB")
    print(f"      Samples: {len(sample_data):,}")
    
    if binary_label_col:
        sample_dist = sample_data[binary_label_col].value_counts()
        print(f"      Distribution: Normal={sample_dist.get(0,0):,}, Attack={sample_dist.get(1,0):,}")

# Save metadata
metadata = {
    "dataset": "UNSW-NB15",
    "processed_date": time.strftime("%Y-%m-%d %H:%M:%S"),
    "original_files": [os.path.basename(f) for f in main_files],
    "dimensions": {
        "rows": int(full_data.shape[0]),
        "columns": int(full_data.shape[1])
    },
    "label_columns": {
        "binary": binary_label_col if binary_label_col else "not_found",
        "attack_category": attack_cat_col if attack_cat_col else "not_found"
    },
    "processing_time_seconds": time.time() - start_time
}

metadata_path = os.path.join(PROCESSED_DATA_PATH, "unsw_nb15_metadata.json")
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=4)

print(f"📋 Metadata saved: {os.path.basename(metadata_path)}")

# ==================== 10. FINAL SUMMARY ====================
print("\n" + "=" * 70)
print("✅ UNSW-NB15 PROCESSING COMPLETE!")
print("=" * 70)
print(f"• Files processed : {len(main_files)}")
print(f"• Total rows      : {full_data.shape[0]:,}")
print(f"• Total columns   : {full_data.shape[1]}")
print(f"• Label column    : '{binary_label_col}'" if binary_label_col else "• Label column    : Not found")
print(f"• Sample created  : 50,000 rows (comparable with CICIDS2017)")
print(f"• Processing time : {time.time() - start_time:.1f} seconds")

print("\n📁 Output files in 'data/processed/':")
print(f"   1. {os.path.basename(OUTPUT_FULL)}")
print(f"   2. {os.path.basename(OUTPUT_SAMPLE)}")
print(f"   3. {os.path.basename(metadata_path)}")

print("\n🎯 Next steps:")
print("   1. Verify the 'Label' column exists and contains 0/1 values")
print("   2. Run comparable sampling for fair comparison")
print("   3. Proceed with modeling")
print("=" * 70)