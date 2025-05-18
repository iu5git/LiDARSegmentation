import os
import pandas as pd
import numpy as np
from lidarsegmentation.settings.coord_settings import CS
from lidarsegmentation import predict
from tqdm import tqdm

def clear_excess_stumps(cs, merged_df=None, pcd_map=None):
    """
    Process the merged coordinates DataFrame and classify stumps.
    
    Args:
        cs: Coordinate settings object
        merged_df: DataFrame containing merged coordinates
        pcd_map: Dictionary mapping stump names to PCD objects
        
    Returns:
        DataFrame with added classification labels
    """
    if merged_df is None or merged_df.empty or pcd_map is None:
        return pd.DataFrame()
    
    model_name = 'int0000_7000-512-rlish-s4762'
    
    # Count unique intensity levels
    name_cols = [col for col in merged_df.columns if col.startswith('Name_stump')]
    n = len(name_cols)
    
    # Prepare for labels
    labels_cols = []
    for name_col in name_cols:
        intensity = name_col.split('_')[-1]
        if "." in intensity:
            intensity = intensity.split(".")[0]
        labels_cols.append(f"Labels_{intensity}")
    
    # Create DataFrame to hold labels
    labels_df = pd.DataFrame(index=merged_df.index, columns=labels_cols)
    labels_df.fillna(-1, inplace=True)
    
    # Process each intensity level
    for i, name_col in enumerate(name_cols):
        labels_col = labels_cols[i]
        print(f"Processing {name_col} -> {labels_col}")
        
        for j in tqdm(range(merged_df.shape[0])):
            stump_name = merged_df.at[j, name_col]
            
            if stump_name != "File__Not__Found":
                # Get the PCD from memory
                if stump_name in pcd_map:
                    pcd_obj = pcd_map[stump_name]
                    
                    # Predict the label directly from the PCD object
                    label = predict.predict_from_pcd(pcd_obj, model_name)
                    labels_df.at[j, labels_col] = label
                else:
                    # PCD not found in memory
                    print(f"No such PCD in memory: {stump_name}")
                    labels_df.at[j, labels_col] = -3
            else:
                labels_df.at[j, labels_col] = -2
    
    # Combine original DataFrame with labels
    result_df = pd.concat([merged_df, labels_df], axis=1)
    
    return result_df

if __name__ == "__main__":
    yml_path = "settings\settings.yaml"
    cs = CS.from_yaml(yml_path)
    
    # For testing only - this would normally be called from main.py
    # with actual DataFrames and PCD objects
    test_df = pd.DataFrame({
        'Name_stump_int7000': ['stump1', 'stump2', 'File__Not__Found'],
        'X': [10.0, 20.0, 30.0],
        'Y': [15.0, 25.0, 35.0],
        'Diameter_int7000': [0.5, 0.6, 0.0],
        'Name_stump_int5000': ['File__Not__Found', 'stump3', 'stump4'],
        'Diameter_int5000': [0.0, 0.7, 0.8]
    })
    
    # Test with empty PCD dict - would get -3 labels in real use
    result = clear_excess_stumps(cs, test_df, {})
    print(result)
