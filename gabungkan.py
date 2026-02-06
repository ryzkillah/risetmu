# ============================================
# SCRIPT: MERGE ALL CICIDS2017 FILES (FIXED VERSION)
# Improvement: Handle different column structures + alignment
# ============================================

import pandas as pd
import numpy as np
import os
import glob
import time
import json
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def setup_paths():
    """Setup input and output paths"""
    raw_data_path = "data/raw/CICIDS2017/MachineLearningCSV/MachineLearningCVE"
    processed_data_path = "data/processed"
    
    # Create directories if they don't exist
    os.makedirs(processed_data_path, exist_ok=True)
    os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
    
    # Output files
    output_full = os.path.join(processed_data_path, "CICIDS2017_Full.csv")
    output_sample = os.path.join(processed_data_path, "CICIDS2017_Sample_50k.csv")
    
    return raw_data_path, processed_data_path, output_full, output_sample

def find_csv_files(raw_data_path):
    """Find all CSV files in the specified directory"""
    print("\n[1/6] 🔍 Searching for CSV files...")
    
    csv_files = sorted(glob.glob(os.path.join(raw_data_path, "*.csv")))
    
    if not csv_files:
        print("❌ ERROR: No CSV files found in directory:", raw_data_path)
        print("\n💡 SOLUTION: Ensure the folder structure is as follows:")
        print("   data/")
        print("   └── raw/")
        print("       └── CICIDS2017/")
        print("           └── MachineLearningCSV/")
        print("               └── MachineLearningCVE/")
        print("                   ├── Monday-WorkingHours.pcap_ISCX.csv")
        print("                   ├── Tuesday-WorkingHours.pcap_ISCX.csv")
        print("                   └── ...")
        return None
    
    print(f"✅ Found {len(csv_files)} CSV files:")
    for i, f in enumerate(csv_files[:5], 1):
        file_size = os.path.getsize(f) / (1024 * 1024)
        print(f"   {i}. {os.path.basename(f):55s} - {file_size:.1f} MB")
    if len(csv_files) > 5:
        print(f"   ... and {len(csv_files) - 5} more files")
    
    return csv_files

def load_dataframes(csv_files):
    """Load all CSV files with consistent structure"""
    print("\n[2/6] ⚙️ Loading files with consistent structure...")
    
    all_dataframes = []
    column_sets = []
    success_count = 0
    
    for i, file_path in enumerate(csv_files, 1):
        file_name = os.path.basename(file_path)
        print(f"  [{i:2d}/{len(csv_files)}] {file_name[:45]:45s}", end=" ")
        
        try:
            # Try different encodings
            try:
                df = pd.read_csv(file_path, encoding='cp1252', low_memory=False)
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='ISO-8859-1', low_memory=False)
            
            # Standardize column names
            df.columns = df.columns.str.strip()
            
            # Save column structure
            column_sets.append(set(df.columns))
            
            # Add identifier
            df['source_file'] = file_name
            
            all_dataframes.append(df)
            success_count += 1
            print(f"✅ {df.shape[0]:>7,} rows, {df.shape[1]:>3} columns")
            
        except Exception as e:
            print(f"❌ Error: {str(e)[:60]}")
    
    return all_dataframes, column_sets, success_count

def analyze_column_structure(column_sets):
    """Analyze column structure across files"""
    print("\n[3/6] 📊 Analyzing column structure across files...")
    
    if not column_sets:
        print("⚠️  No column sets to analyze")
        return
    
    # Find common columns
    common_columns = set.intersection(*column_sets)
    print(f"✅ Columns common to all files: {len(common_columns)}")
    
    # Show sample of common columns
    print("   Sample of common columns:")
    for i, col in enumerate(sorted(list(common_columns))[:15], 1):
        print(f"   {i:2d}. {col}")
    
    # Find unique columns
    all_columns = set.union(*column_sets)
    unique_columns = all_columns - common_columns
    if unique_columns:
        print(f"\n⚠️  Columns NOT in all files: {len(unique_columns)}")
        print("   These may cause issues when merging")
        if len(unique_columns) <= 10:
            print("   Unique columns:", sorted(list(unique_columns)))
    
    return common_columns

def smart_merge_with_alignment(all_dataframes):
    """Merge dataframes with column alignment"""
    print(f"\n[4/6] 🔗 Smart merging {len(all_dataframes)} files...")
    print("   Method: Aligning columns before merge...")
    
    # Get all unique columns
    all_unique_columns = set()
    for df in all_dataframes:
        all_unique_columns.update(df.columns)
    
    # Sort columns for consistency
    sorted_columns = sorted(list(all_unique_columns))
    print(f"   Total unique columns across all files: {len(sorted_columns)}")
    
    # Align each dataframe to the same columns
    aligned_dataframes = []
    for df in all_dataframes:
        aligned_df = pd.DataFrame(index=df.index)
        
        for col in sorted_columns:
            if col in df.columns:
                aligned_df[col] = df[col]
            else:
                # Column doesn't exist in this file, fill with NaN
                aligned_df[col] = np.nan
        
        aligned_dataframes.append(aligned_df)
    
    # Merge all aligned dataframes
    combined_data = pd.concat(aligned_dataframes, ignore_index=True, sort=False)
    print(f"✅ Successfully merged: {combined_data.shape[0]:,} rows × {combined_data.shape[1]} columns")
    
    return combined_data

def identify_target_columns(combined_data):
    """Identify label columns in the dataset"""
    print("\n[5/6] 🎯 Identifying target columns...")
    
    # Search for label columns case-insensitively
    binary_label_col = None
    label_col_candidates = []
    
    for col in combined_data.columns:
        col_lower = col.lower()
        
        # Look for common label column names
        if col_lower == 'label':
            binary_label_col = col
            print(f"✅ Found label column: '{col}'")
            break
        elif any(x in col_lower for x in ['label', 'class', 'type', 'category', 'attack']):
            label_col_candidates.append(col)
    
    # If not found with exact match, use first candidate
    if not binary_label_col and label_col_candidates:
        binary_label_col = label_col_candidates[0]
        print(f"⚠️  Using potential label column: '{binary_label_col}'")
    
    return binary_label_col

def convert_cicids_label(label_value):
    """Convert CICIDS2017 label to binary format"""
    if pd.isna(label_value):
        return np.nan
    
    label_str = str(label_value).lower()
    
    # Check for normal/benign
    if any(kw in label_str for kw in ['benign', 'normal']):
        return 0
    # Check for attack/malicious
    elif any(kw in label_str for kw in ['malicious', 'attack']):
        return 1
    
    # Try to parse as numeric
    try:
        val = float(label_value)
        return 1 if val > 0 else 0
    except:
        return 1  # Default to attack

def clean_and_process_data(combined_data, binary_label_col):
    """Clean and process the combined data"""
    print("\n[6/6] 🧹 Cleaning and processing data...")
    
    # 1. Remove duplicate columns
    print("   1. Removing duplicate columns...")
    cols_before = combined_data.shape[1]
    
    dup_cols_to_remove = []
    for i, col1 in enumerate(combined_data.columns):
        for j, col2 in enumerate(combined_data.columns[i + 1:], i + 1):
            try:
                if col1 != col2 and combined_data[col1].equals(combined_data[col2]):
                    dup_cols_to_remove.append(col2)
            except:
                continue
    
    if dup_cols_to_remove:
        combined_data = combined_data.drop(columns=dup_cols_to_remove)
    
    cols_after = combined_data.shape[1]
    if cols_before != cols_after:
        print(f"   Removed {cols_before - cols_after} duplicate columns")
    
    # 2. Handle infinite values
    print("   2. Handling infinite values...")
    numeric_cols = combined_data.select_dtypes(include=[np.number]).columns
    
    inf_count = 0
    for col in numeric_cols:
        if np.isinf(combined_data[col]).any():
            inf_count += np.isinf(combined_data[col]).sum()
            combined_data[col] = combined_data[col].replace([np.inf, -np.inf], np.nan)
    
    if inf_count > 0:
        print(f"   Found {inf_count} infinite values, converted to NaN")
    
    # 3. Handle missing values
    print("   3. Handling missing values...")
    missing_before = combined_data.isnull().sum().sum()
    
    if missing_before > 0:
        print(f"   Found {missing_before:,} missing values")
        
        # Fill numeric columns with median
        for col in numeric_cols:
            if combined_data[col].isnull().any():
                median_val = combined_data[col].median()
                combined_data[col] = combined_data[col].fillna(median_val)
        
        # Fill categorical columns with mode
        categorical_cols = combined_data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != 'source_file' and combined_data[col].isnull().any():
                if not combined_data[col].mode().empty:
                    mode_val = combined_data[col].mode()[0]
                    combined_data[col] = combined_data[col].fillna(mode_val)
                else:
                    combined_data[col] = combined_data[col].fillna('Unknown')
        
        missing_after = combined_data.isnull().sum().sum()
        print(f"   After cleaning: {missing_after:,} missing values")
    
    # 4. Remove rows with too many missing values
    print("   4. Removing problematic rows...")
    rows_before = combined_data.shape[0]
    combined_data = combined_data.dropna(thresh=combined_data.shape[1] // 2)
    rows_after = combined_data.shape[0]
    
    rows_removed = rows_before - rows_after
    if rows_removed > 0:
        print(f"   Removed {rows_removed} rows ({rows_removed / rows_before * 100:.1f}%)")
    
    print(f"   ✅ Final clean data: {combined_data.shape[0]:,} rows × {combined_data.shape[1]} columns")
    
    return combined_data, rows_removed, inf_count, (cols_before - cols_after)

def create_stratified_sample(combined_data, binary_label_col, sample_size=50000):
    """Create stratified sample preserving class distribution"""
    if binary_label_col and combined_data[binary_label_col].dtype.kind in 'biufc':
        # Calculate samples per class
        class_counts = combined_data[binary_label_col].value_counts()
        total_samples_needed = sample_size
        
        sample_data_list = []
        
        for class_val, class_count in class_counts.items():
            # Original class proportion
            class_prop = class_count / len(combined_data)
            # Number of samples for this class
            n_samples_class = int(total_samples_needed * class_prop)
            
            # Sample (minimum between needed and available)
            n_to_sample = min(n_samples_class, class_count)
            
            if n_to_sample > 0:
                class_samples = combined_data[combined_data[binary_label_col] == class_val].sample(
                    n=n_to_sample, random_state=42)
                sample_data_list.append(class_samples)
        
        # Combine and shuffle
        sample_data = pd.concat(sample_data_list).sample(frac=1, random_state=42)
        
        # Add random samples if less than target size
        if len(sample_data) < sample_size:
            remaining = sample_size - len(sample_data)
            additional_samples = combined_data.drop(sample_data.index).sample(
                n=remaining, random_state=42)
            sample_data = pd.concat([sample_data, additional_samples]).sample(frac=1, random_state=42)
        
        return sample_data
    
    # Fallback to random sampling
    return combined_data.sample(n=min(sample_size, len(combined_data)), random_state=42)

def save_metadata(metadata, processed_data_path):
    """Save processing metadata to JSON file"""
    metadata_path = os.path.join(processed_data_path, "cicids2017_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    return metadata_path

def main():
    """Main function to process CICIDS2017 dataset"""
    print("=" * 70)
    print("MERGE ALL CICIDS2017 FILES - FIXED VERSION")
    print("IMPROVEMENT: Handle different column structures & alignment")
    print("=" * 70)
    
    start_time = time.time()
    
    # Step 1: Setup paths
    raw_data_path, processed_data_path, output_full, output_sample = setup_paths()
    print(f"📁 Input path: {raw_data_path}")
    print(f"📁 Output path: {processed_data_path}")
    
    # Step 2: Find CSV files
    csv_files = find_csv_files(raw_data_path)
    if csv_files is None:
        return
    
    # Step 3: Load dataframes
    all_dataframes, column_sets, success_count = load_dataframes(csv_files)
    if not all_dataframes:
        print("❌ ERROR: No dataframes loaded successfully")
        return
    
    # Step 4: Analyze column structure
    analyze_column_structure(column_sets)
    
    # Step 5: Merge with alignment
    combined_data = smart_merge_with_alignment(all_dataframes)
    
    # Step 6: Identify target columns
    binary_label_col = identify_target_columns(combined_data)
    
    # Convert labels if found
    if binary_label_col:
        print(f"\n📊 Analyzing label column '{binary_label_col}':")
        print(f"   Data type: {combined_data[binary_label_col].dtype}")
        
        # Clean labels
        combined_data[binary_label_col] = combined_data[binary_label_col].astype(str).str.strip()
        
        unique_vals = combined_data[binary_label_col].dropna().unique()
        print(f"   Unique values: {len(unique_vals)}")
        print(f"   Sample values: {unique_vals[:10]}")
        
        # Convert to binary if needed
        if len(unique_vals) <= 3:
            print(f"   Converting labels to binary format (0/1)...")
            combined_data[binary_label_col] = combined_data[binary_label_col].apply(convert_cicids_label)
            
            # Verify conversion
            converted_counts = combined_data[binary_label_col].value_counts()
            print(f"   After conversion: {dict(converted_counts)}")
    
    # Step 7: Clean and process data
    combined_data, rows_removed, inf_count, dup_cols_removed = clean_and_process_data(
        combined_data, binary_label_col)
    
    # Step 8: Analyze final dataset
    print("\n📊 Final dataset analysis...")
    print(f"• Total samples: {combined_data.shape[0]:,}")
    print(f"• Total features: {combined_data.shape[1]}")
    print(f"• Memory usage: {combined_data.memory_usage(deep=True).sum() / (1024 ** 2):.1f} MB")
    
    if binary_label_col:
        label_counts = combined_data[binary_label_col].value_counts()
        print(f"\n🎯 LABEL DISTRIBUTION:")
        for val, count in label_counts.items():
            pct = count / len(combined_data) * 100
            label_name = "Normal/Benign" if val == 0 else f"Attack/Class_{val}"
            print(f"   {label_name}: {count:,} samples ({pct:.1f}%)")
    
    # Step 9: Save data
    print("\n💾 Saving processed data...")
    
    # Save full dataset
    print(f"   1. Saving full dataset -> {os.path.basename(output_full)}")
    combined_data.to_csv(output_full, index=False)
    full_size = os.path.getsize(output_full) / (1024 * 1024)
    print(f"      Size: {full_size:.1f} MB")
    
    # Create and save sample
    print(f"   2. Creating 50k sample -> {os.path.basename(output_sample)}")
    sample_size = 50000
    
    if len(combined_data) >= sample_size:
        if binary_label_col:
            print(f"      Using stratified sampling...")
            sample_data = create_stratified_sample(combined_data, binary_label_col, sample_size)
        else:
            sample_data = combined_data.sample(n=sample_size, random_state=42)
        
        sample_data.to_csv(output_sample, index=False)
        sample_size_mb = os.path.getsize(output_sample) / (1024 * 1024)
        print(f"      Size: {sample_size_mb:.1f} MB")
        print(f"      Sample: {len(sample_data):,} rows")
        
        if binary_label_col:
            sample_dist = sample_data[binary_label_col].value_counts()
            print(f"      Distribution:")
            for val, count in sample_dist.items():
                label_name = "Normal/Benign" if val == 0 else f"Attack/Class_{val}"
                pct = count / len(sample_data) * 100
                print(f"        {label_name}: {count:,} ({pct:.1f}%)")
    else:
        print(f"   ⚠️  Dataset too small ({len(combined_data):,} rows) for 50k sample")
    
    # Step 10: Save metadata
    processing_time = time.time() - start_time
    
    metadata = {
        "dataset": "CICIDS2017",
        "processed_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "original_files": [os.path.basename(f) for f in csv_files],
        "successful_files": success_count,
        "dimensions": {
            "rows": int(combined_data.shape[0]),
            "columns": int(combined_data.shape[1])
        },
        "label_columns": {
            "binary": binary_label_col if binary_label_col else "not_found"
        },
        "cleaning_stats": {
            "rows_removed": int(rows_removed),
            "infinite_values_handled": int(inf_count),
            "duplicate_columns_removed": int(dup_cols_removed)
        },
        "sampling_info": {
            "sample_size": sample_size if len(combined_data) >= sample_size else len(combined_data),
            "method": "stratified" if binary_label_col else "random"
        },
        "processing_time_seconds": round(processing_time, 2)
    }
    
    metadata_path = save_metadata(metadata, processed_data_path)
    
    # Final summary
    print("\n" + "=" * 70)
    print("✅ CICIDS2017 PROCESSING COMPLETE!")
    print("=" * 70)
    print(f"• Files processed : {success_count}/{len(csv_files)}")
    print(f"• Total rows      : {combined_data.shape[0]:,}")
    print(f"• Total columns   : {combined_data.shape[1]}")
    print(f"• Label column    : '{binary_label_col}'" if binary_label_col else "• Label column    : Not found")
    print(f"• Sample created  : {min(50000, len(combined_data)):,} rows")
    print(f"• Processing time : {processing_time:.1f} seconds")
    
    print("\n📁 Output files in 'data/processed/':")
    print(f"   1. {os.path.basename(output_full)}")
    print(f"   2. {os.path.basename(output_sample)}")
    print(f"   3. {os.path.basename(metadata_path)}")
    
    print("\n🎯 Next steps:")
    print("   1. Verify the label column (if exists) contains 0/1 values")
    print("   2. Run comparable sampling for fair comparison with UNSW-NB15")
    print("   3. Proceed with modeling")
    print("=" * 70)

if __name__ == "__main__":
    main()