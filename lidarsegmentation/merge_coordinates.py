import os
import pandas as pd
import numpy as np
from lidarsegmentation.settings.coord_settings import CS

def makedirs_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)

def merge(file1, file2, iter, names_col, array):
    array = np.asarray(array)

    n_col_name = 0 + iter
    n_col_diam = (iter+1)*2

    eps = 0.25
    for index, row in file1.iterrows():
        array = np.vstack([array, np.asarray(row)])

    array = array[1:]
    XY = array[:,iter:iter+2]

    added_column_name = np.asarray(np.full(array.shape[0],"File__Not__Found"), dtype=str)
    added_column_diameter = np.asarray(np.full(array.shape[0],0.0), dtype=np.float32)
    for index, row in file2.iterrows():
        for point in XY:
            if np.linalg.norm(np.asarray([float(point[0]), float(point[1])]) - np.asarray([float(row[1]), float(row[2])])) < eps:
                idx = np.where(XY == point)[0][0]
                
                added_column_name[idx] = row[0]
                added_column_diameter[idx] = row[3]

    array = np.insert(array, n_col_diam, added_column_diameter, axis=1)
    array = np.insert(array, n_col_name, added_column_name, axis=1)
   

    for index, row in file2.iterrows():
        AddFlag = True
        for point in XY:
            if np.linalg.norm(np.asarray([float(point[0]), float(point[1])]) - np.asarray([float(row[1]), float(row[2])])) < eps:
                AddFlag = False
        if AddFlag:
            added_row = ["File__Not__Found",row[0],row[1],row[2],0.0,row[3]]
            if iter > 1:
                added_row.insert((iter)*2, 0.0)
                added_row.insert(iter-1, "File__Not__Found")
            added_row = np.asarray(added_row)
            array = np.vstack([array, added_row])

    df = pd.DataFrame(data = array, columns=names_col)
    df = df.dropna()
    df = df[(df.X != 'nan')]
    df = df[(df.Y != 'nan')]
    return df

def init_merge_file(cs):
    txt_path = os.path.join(cs.path_base, "coordinates_paths.txt") 
    file = open(txt_path, "r")
    i = 0
    iter = 0
    while True:
        line = file.readline()
        if not line:
            break
        file_name = line.strip()
        if file_name == '':
            continue
        file1_path = file_name
        splt_fn = file_name.split(sep="_")[-1]
        splt_fn = splt_fn.split(sep=".")[0]

        if i == 0:
            names_col = ["Name_stump_" + splt_fn, "X", "Y", "Diameter_" + splt_fn]
        if i > 0:
            names_col.insert((iter+2)*2, "Diameter_" + splt_fn)
            names_col.insert(iter+1, "Name_stump_" + splt_fn)
            if i == 1:
                array = ['n',0,0,0]
            else:
                array.insert(i,'n')
                array.insert(i*2+1,0)
       
        i+=1
        if i >= 2 :
            iter += 1
            if iter == 1:
                file1 = pd.read_csv(file2_path, delimiter=";")
                file2 = pd.read_csv(file1_path, delimiter=";")
            else:
                file1 = df
                file2 = pd.read_csv(file1_path, delimiter=";")
            df = merge(file1, file2, iter, names_col, [array])
        
        file2_path = file1_path

    file.close()

    return df

def merge_dfs(dfs, eps=0.25):
    """
    Merge multiple dataframes containing tree coordinates.
    
    Args:
        dfs: List of pandas DataFrames containing tree coordinates
        eps: Epsilon distance for matching coordinates
        
    Returns:
        Merged pandas DataFrame
    """
    if not dfs:
        return pd.DataFrame()
    
    if len(dfs) == 1:
        return dfs[0]
    
    # Initialize with the first dataframe
    result_df = dfs[0].copy()
    
    # Get column names from first DataFrame
    all_column_names = []
    for i, df in enumerate(dfs):
        if df.empty:
            continue
        columns = df.columns.tolist()
        name_col = [col for col in columns if col.startswith('Name_stump')][0]
        diam_col = [col for col in columns if col.startswith('Diameter')][0]
        all_column_names.append((name_col, diam_col))
    
    # Merge additional dataframes
    for i, df in enumerate(dfs[1:], 1):
        if df.empty:
            continue
            
        # Get current column names
        name_col_curr = all_column_names[i][0]
        diam_col_curr = all_column_names[i][1]
        
        # Add placeholder columns to result_df
        if name_col_curr not in result_df.columns:
            result_df[name_col_curr] = "File__Not__Found"
        if diam_col_curr not in result_df.columns:
            result_df[diam_col_curr] = 0.0
        
        # Update existing rows
        for idx, row in df.iterrows():
            # Find matching coordinates
            matching_rows = result_df[
                (abs(result_df['X'] - row['X']) < eps) & 
                (abs(result_df['Y'] - row['Y']) < eps)
            ]
            
            if not matching_rows.empty:
                # Update existing row
                for match_idx in matching_rows.index:
                    result_df.at[match_idx, name_col_curr] = row[name_col_curr]
                    result_df.at[match_idx, diam_col_curr] = row[diam_col_curr]
            else:
                # Add new row
                new_row = pd.Series(index=result_df.columns)
                new_row['X'] = row['X']
                new_row['Y'] = row['Y']
                new_row[name_col_curr] = row[name_col_curr]
                new_row[diam_col_curr] = row[diam_col_curr]
                
                # Set default values for other columns
                for col in result_df.columns:
                    if col not in ['X', 'Y', name_col_curr, diam_col_curr]:
                        if col.startswith('Name_stump'):
                            new_row[col] = "File__Not__Found"
                        elif col.startswith('Diameter'):
                            new_row[col] = 0.0
                
                result_df = pd.concat([result_df, pd.DataFrame([new_row])], ignore_index=True)
    
    return result_df.dropna().reset_index(drop=True)

def merge_coordinates(cs: CS, dfs=None, save_to_disk: bool = True):
    """
    Legacy function that now works with in-memory DataFrames.
    
    Args:
        cs: Coordinate settings object
        dfs: Optional list of DataFrames to merge. If None, returns empty DataFrame
        save_to_disk: whether to write merged CSV to disk
    Returns:
        Merged DataFrame
    """
    if dfs is None or not dfs:
        return pd.DataFrame()
    merged_df = merge_dfs(dfs)
    if save_to_disk:
        # Save merged coordinates CSV
        filename = cs.fname_points.partition('.')[0] + "_Coordinates_Merged.csv"
        save_path = os.path.join(cs.path_base, filename)
        merged_df.to_csv(save_path, index=False, sep=';')
        print(f"Saved merged coordinates to {save_path}")
    return merged_df
    
if __name__ == "__main__":
    yml_path = "settings\settings.yaml"
    cs = CS.from_yaml(yml_path)
    
    # Test with sample dataframes
    df1 = pd.DataFrame({
        'Name_stump_int7000': ['tree1', 'tree2'], 
        'X': [10.0, 20.0], 
        'Y': [15.0, 25.0], 
        'Diameter_int7000': [0.5, 0.6]
    })
    
    df2 = pd.DataFrame({
        'Name_stump_int5000': ['tree3', 'tree4'], 
        'X': [10.1, 30.0], 
        'Y': [15.1, 35.0], 
        'Diameter_int5000': [0.7, 0.8]
    })
    
    result = merge_coordinates(cs, [df1, df2])
    print(result)
