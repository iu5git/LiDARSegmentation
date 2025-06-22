import os
from lidarsegmentation.slope_steepness import slope_steepness_task
from lidarsegmentation.settings.slope_settings import SlopeSettings


def slope_steepnees(ss):
    """
    Run the slope steepness construction pipeline
    
    Args:
        ss: Segmentation settings object
        
    Returns:
        Dictionary containing all outputs from slope steepness construction
    """
    print("Running slope steepness construction pipeline...")
    
    # Load input point cloud
    input_path = os.path.join(ss.path_base, ss.fname_points)
    
    # Run slope steepness task with method selection from settings
    results = slope_steepness_task(
        ss,
        ground_method=ss.slope_ground_method,
        void_method=ss.slope_void_method,
        mesh_method=ss.slope_mesh_method,
        color_method=ss.slope_color_method,
        isohypse_method=ss.slope_isohypse_method
    )
    
    print("Slope steepness construction completed successfully")
    
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run the complete in-memory segmentation pipeline')
    parser.add_argument('--settings', type=str, default='settings/settings.yaml', help='Path to settings YAML file')
    parser.add_argument('--save-pcds', action='store_true', help='Save final PCD files')
    args = parser.parse_args()
    
    ss = SlopeSettings.from_yaml(args.settings)
    slope_steepnees(ss, model_name=args.model, save_final_pcds=args.save_pcds)
