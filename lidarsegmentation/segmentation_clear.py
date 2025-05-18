import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import os
from scipy.spatial.distance import cdist
from lidarsegmentation.settings.seg_settings import SS
from lidarsegmentation.classes.PCD_TREE import PCD_TREE
from lidarsegmentation.classes.PCD_UTILS import PCD_UTILS
from lidarsegmentation.classes.PCD import PCD

def clustering(pc_tree):
    P = pd.DataFrame(pc_tree.points, columns = ['X','Y','Z'])
    X = np.asarray(P)
    if pc_tree.points.shape[0]<100000:
        clustering = DBSCAN(eps=1.25, min_samples=25).fit(X)
        labels=clustering.labels_
    else:
        labels = np.zeros(pc_tree.points.shape[0])
    return labels

def segmentation_clear(ss, combined_df, ram_trees):
    """
    Final segmentation stage to extract the main trunk of each tree
    
    Args:
        ss: Segmentation settings
        combined_df: Combined DataFrame with tree info from segmentation_ram
        ram_trees: Dictionary of processed PCD objects from segmentation_ram
        
    Returns:
        Dictionary of final trunk PCD objects keyed by filename
    """
    inum = 0
    clear_trees = {}
   
    for fname, pc_tree in tqdm(ram_trees.items(), desc="Processing trunks"):
        inum += 1
        if inum < ss.first_num:
            continue
        
        try:
            # Check if tree name exists in the DataFrame
            if fname not in combined_df['Name_tree'].values:
                print(f"Warning: {fname} not found in combined dataframe. Skipping.")
                continue
                
            labels = clustering(pc_tree)

            min_z_values = []
            for i in np.unique(labels):
                if i>-1:
                    idx_layer=np.where(labels==i)
                    i_data = pc_tree.points[idx_layer]
                    if i_data.shape[0] == 0:  # Skip empty clusters
                        continue
                    index = i_data[:, 2].argmin()
                    min_z_value = i_data[index]
                    min_z_values.append(min_z_value)
                    
            if len(min_z_values) == 0:
                print(f"Warning: No valid clusters found for {fname}. Skipping.")
                continue
                
            min_z_values = np.asarray(min_z_values)
            
            idx_labels=np.where(min_z_values[:,2] < pc_tree.points.min(axis=0)[2]+1)
            min_z_values = min_z_values[idx_labels]
            
            if min_z_values.shape[0] == 0:
                print(f"Warning: No valid min Z values found for {fname}. Skipping.")
                continue

            centers_labels = []
            for i in range(min_z_values.shape[0]):
                idx_layer=np.where(labels==i)
                i_data = pc_tree.points[idx_layer]
                if i_data.shape[0] == 0:  # Skip empty clusters
                    continue
                center = PCD_UTILS.center_m(i_data[:,0:2])
                centers_labels.append(center)
                
            if len(centers_labels) == 0:
                print(f"Warning: No valid centers found for {fname}. Skipping.")
                continue
                
            centers_labels = np.asarray(centers_labels)
            
            x_value = combined_df.loc[combined_df['Name_tree'] == fname, 'X'].values[0]
            y_value = combined_df.loc[combined_df['Name_tree'] == fname, 'Y'].values[0]
            main_center = [x_value, y_value]

            distances = cdist(centers_labels, [main_center])
            min_distance_index = np.argmin(distances)

            pc_result = PCD(pc_tree.points, pc_tree.intensity)
            idx_l=np.where(labels==min_distance_index)
            pc_result.index_cut(idx_l)

            if pc_result.points.shape[0]>1:
                clear_trees[fname] = pc_result
            else:
                print(f"Warning: Trunk for {fname} has less than 2 points. Skipping.")
        except Exception as e:
            print(f"Error processing {fname}: {e}")
    
    return clear_trees

if __name__ == "__main__" :
    from lidarsegmentation.segmentation_vor import segmentation_vor
    from lidarsegmentation.segmentation_ram import segmentation_ram
    yml_path = "settings\settings.yaml"
    ss = SS.from_yaml(yml_path)
    binding_df, vor_trees = segmentation_vor(ss, make_binding=True)
    combined_df, ram_trees = segmentation_ram(ss, binding_df, vor_trees)
    clear_trees = segmentation_clear(ss, combined_df, ram_trees)
    print(f"Processed {len(clear_trees)} trees in Clear stage")
