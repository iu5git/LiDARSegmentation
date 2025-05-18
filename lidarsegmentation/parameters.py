from lidarsegmentation.settings.seg_settings import SS
import os
from tqdm import tqdm
from lidarsegmentation.classes.PCD_TREE import PCD_TREE
from lidarsegmentation.classes.PCD import PCD
import pandas as pd
from typing import Dict, Optional, Union, List

def parameters(ss: SS, combined_df: pd.DataFrame, clear_trees: Dict[str, PCD], K: int = 0) -> pd.DataFrame:
    """
    Calculate tree parameters from processed PCD objects
    
    Args:
        ss: Segmentation settings
        combined_df: Combined DataFrame with tree info
        clear_trees: Dictionary of final trunk PCD objects
        K: Optional starting index
        
    Returns:
        DataFrame with calculated tree parameters
    """
    mi = 0
    
    # Define parameter lists in a dictionary to make code more DRY
    columns = [
        "Name", 
        "Diameter_LS, cm", "Diameter_HLS, cm", 
        "Height, m", "Length, m", 
        "Crown_volume, m3", "Crown_square, m2", 
        "XY_crown_square, m2", "XZ_crown_square, m2", "YZ_crown_square, m2", 
        "X_UP_0", "Y_UP_0", 
        "X_UP_1", "Y_UP_1", 
        "X_UP_2", "Y_UP_2", 
        "X_UP_3", "Y_UP_3"
    ]
    param_lists = {key: [] for key in columns}

    for fname, pcd_obj in tqdm(clear_trees.items(), desc="Calculating parameters"):
        mi += 1
        if mi < K:
            continue
        
        # Default values in case of processing error
        tree_params = {"Name": fname, **{key: 0 for key in param_lists.keys()}}
        
        try:
            # Get the coordinates from the combined_df for this tree
            x_value = combined_df.loc[combined_df['Name_tree'] == fname, 'X'].values[0]
            y_value = combined_df.loc[combined_df['Name_tree'] == fname, 'Y'].values[0]
            coordinate = [x_value, y_value, 0]  # Add a default z-coordinate
            
            # Create a PCD_TREE with the points, intensity, and coordinate
            pc_tree = PCD_TREE(points=pcd_obj.points, intensity=pcd_obj.intensity, coordinate=coordinate)
            pc_tree.RGBint = pc_tree.intensity/max(pc_tree.intensity) if max(pc_tree.intensity) > 0 else pc_tree.intensity
            
            pc_tree.estimate_height()
            pc_tree.estimate_length()
            pc_tree.search_main_coordinate(combined_df, fname)
            pc_slice = pc_tree.search_slice()
            pc_expsph = pc_tree.search_points_for_center(pc_slice)
            
            if pc_expsph.points.shape[0] > 10:
                pc_tree.estimate_diameter(pc_expsph, pc_slice)

                points_no_trunk = pc_tree.search_points_no_trunk()
                if points_no_trunk.shape[0] > 0:
                    pc_tree.estimate_crown(points_no_trunk)
                else:
                    pc_tree.crown_volume = pc_tree.crown_square = pc_tree.xy_crown_square = pc_tree.yz_crown_square = pc_tree.xz_crown_square = 0
            else:
                pc_tree.height = pc_tree.length = pc_tree.diameter_LS = pc_tree.diameter_HLS = 0
                pc_tree.crown_volume = pc_tree.crown_square = pc_tree.xy_crown_square = pc_tree.yz_crown_square = pc_tree.xz_crown_square = 0
            
            # Process upper slices
            down_point = pc_tree.points.max(axis=0)[2] - 2
            pc_upslice_2m = pc_tree.search_up_slice(down_point)
            pc_tree.search_up_coord(pc_upslice_2m)
            
            # Update parameters with calculated values
            tree_params.update({
                "Diameter_LS, cm": pc_tree.diameter_LS,
                "Diameter_HLS, cm": pc_tree.diameter_HLS,
                "Height, m": pc_tree.height,
                "Length, m": pc_tree.length,
                "Crown_volume, m3": pc_tree.crown_volume,
                "Crown_square, m2": pc_tree.crown_square,
                "XY_crown_square, m2": pc_tree.xy_crown_square,
                "XZ_crown_square, m2": pc_tree.xz_crown_square,
                "YZ_crown_square, m2": pc_tree.yz_crown_square,
                "X_UP_0": pc_tree.x_up,
                "Y_UP_0": pc_tree.y_up
            })

            # Process upper third slice
            down_point = 2*(pc_tree.points.max(axis=0)[2]-pc_tree.points.min(axis=0)[2])/3 + pc_tree.points.min(axis=0)[2]
            pc_upslice_third = pc_tree.search_up_slice(down_point)
            
            # Get different coordinate measurements
            pc_tree.search_up_coord(pc_upslice_third)
            tree_params["X_UP_1"] = pc_tree.x_up
            tree_params["Y_UP_1"] = pc_tree.y_up

            pc_tree.search_up_coord(pc_upslice_third, mode='median')
            tree_params["X_UP_2"] = pc_tree.x_up
            tree_params["Y_UP_2"] = pc_tree.y_up

            pc_tree.search_up_coord(pc_upslice_2m, mode='highest')
            tree_params["X_UP_3"] = pc_tree.x_up
            tree_params["Y_UP_3"] = pc_tree.y_up
            
        except Exception as e:
            print(f"Error processing {fname}: {e}")
            # Default values already set in tree_params
        
        # Add values to parameter lists
        for key, value in tree_params.items():
            param_lists[key].append(value)
    
    # Create DataFrame from collected parameters
    return pd.DataFrame(param_lists)

if __name__ == "__main__" :
    from lidarsegmentation.segmentation_vor import segmentation_vor
    from lidarsegmentation.segmentation_ram import segmentation_ram
    from lidarsegmentation.segmentation_clear import segmentation_clear
    
    yml_path = "settings\settings.yaml"
    ss = SS.from_yaml(yml_path)
    
    # In-memory pipeline
    binding_df, vor_trees = segmentation_vor(ss, make_binding=True)
    combined_df, ram_trees = segmentation_ram(ss, binding_df, vor_trees)
    clear_trees = segmentation_clear(ss, combined_df, ram_trees)
    params_df = parameters(ss, combined_df, clear_trees)
    
    # Save final output
    param_name = ss.fname_points.partition('.')[0] + "_Parameters.csv"
    path_save = os.path.join(ss.path_base, param_name)
    params_df.to_csv(path_save, index=False, sep=';')
    print(f"Saved parameters to {path_save}")



                
