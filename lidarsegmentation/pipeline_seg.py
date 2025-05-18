import os
from lidarsegmentation.segmentation_vor import segmentation_vor
from lidarsegmentation.segmentation_ram import segmentation_ram
from lidarsegmentation.segmentation_clear import segmentation_clear
from lidarsegmentation.seg_after import seg_after
from lidarsegmentation.orbit_gif import orbit_gif
from lidarsegmentation.predict import predict_mem
from lidarsegmentation.parameters import parameters
from lidarsegmentation.settings.seg_settings import SS

def run_pipeline(ss, model_name='cpl1-1024-rp-s1024-pn2', save_final_pcds=True):
    """
    Run the complete in-memory segmentation pipeline
    
    Args:
        ss: Segmentation settings object
        model_name: Name of prediction model to use
        save_final_pcds: Whether to save the final PCD files
        
    Returns:
        Dictionary containing all outputs:
        - binding_df: Binding DataFrame from Voronoi segmentation
        - vor_trees: Dictionary of trees from Voronoi segmentation
        - combined_df: Combined DataFrame from RAM segmentation
        - ram_trees: Dictionary of trees from RAM segmentation
        - clear_trees: Dictionary of trees from Clear segmentation
        - pred_df: Predictions DataFrame
        - params_df: Parameters DataFrame
    """
    print("Running Voronoi segmentation...")
    binding_df, vor_trees = segmentation_vor(ss, make_binding=True)
    print(f"Processed {len(vor_trees)} trees in Voronoi segmentation")
    
    print("Running RAM segmentation...")
    combined_df, ram_trees = segmentation_ram(ss, binding_df, vor_trees)
    print(f"Processed {len(ram_trees)} trees in RAM segmentation")
    
    print("Running Clear segmentation...")
    clear_trees = segmentation_clear(ss, combined_df, ram_trees)
    print(f"Processed {len(clear_trees)} trees in Clear segmentation")
    
    # Save final PCD files if requested
    if save_final_pcds:
        save_path = os.path.join(ss.path_base, ss.step1_folder_name, ss.step2_folder_name, ss.step3_folder_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for fname, pcd_obj in clear_trees.items():
            pcd_obj.save(os.path.join(save_path, fname))
        print(f"Saved final PCD files to {save_path}")
        
    print("Running predictions...")
    pred_df = predict_mem(clear_trees, model_name)
    pred_path = os.path.join(ss.path_base, 'predict_' + model_name + '.csv')
    pred_df.to_csv(pred_path, index=False, sep=';')
    print(f"Saved predictions to {pred_path}")
    
    print("Calculating parameters...")
    try:
        params_df = parameters(ss, combined_df, clear_trees)
        param_name = ss.fname_points.partition('.')[0] + "_Parameters.csv"
        param_path = os.path.join(ss.path_base, param_name)
        params_df.to_csv(param_path, index=False, sep=';')
        print(f"Saved parameters to {param_path}")
    except Exception as e:
        print(f"Error calculating parameters: {e}")
        params_df = None
    
    return {
        'binding_df': binding_df,
        'vor_trees': vor_trees,
        'combined_df': combined_df,
        'ram_trees': ram_trees,
        'clear_trees': clear_trees,
        'pred_df': pred_df,
        'params_df': params_df
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run the complete in-memory segmentation pipeline')
    parser.add_argument('--settings', type=str, default='settings/settings.yaml', help='Path to settings YAML file')
    parser.add_argument('--model', type=str, default='cpl1-1024-rp-s1024-pn2', help='Model name for prediction')
    parser.add_argument('--save-pcds', action='store_true', help='Save final PCD files')
    args = parser.parse_args()
    
    ss = SS.from_yaml(args.settings)
    run_pipeline(ss, model_name=args.model, save_final_pcds=args.save_pcds)


