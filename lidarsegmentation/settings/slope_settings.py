import yaml

from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class SlopeSettings:
    """Settings for slope steepness analysis"""
    # Task selection
    task_type: str = "tree_segmentation"  # Options: "slope_steepness"

    save_after_stage: bool = True

    # Output folder
    slope_output_folder: str = "slope_steepness"
    
    # Ground point selection settings
    slope_ground_method: str = "progressive_morphology"  # Options: "progressive_morphology", "cloth_simulation", "ransac"
    slope_cell_size: float = 0.5  # Cell size for progressive morphology
    slope_max_window_size: int = 21  # Maximum window size for progressive morphology
    slope_threshold: float = 0.3  # Slope threshold for progressive morphology
    slope_initial_distance: float = 0.1  # Initial distance threshold for progressive morphology
    slope_max_distance: float = 3.0  # Maximum distance threshold for progressive morphology
    slope_ground_threshold: float = 0.2  # Threshold for ground point selection
    
    # Cloth simulation settings
    slope_cloth_resolution: float = 0.5  # Resolution for cloth simulation
    slope_max_iterations: int = 500  # Maximum iterations for cloth simulation
    slope_height_threshold: float = 0.5  # Height threshold for cloth simulation
    
    # RANSAC settings
    slope_ransac_threshold: float = 0.2  # Distance threshold for RANSAC
    slope_ransac_iterations: int = 1000  # Number of iterations for RANSAC

    # DBSCAN settings
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 10
    dbscan_ground_label: int = 0  # метка для точек земли в результатах DBSCAN

    # OPTICS settings
    optics_min_samples: int = 10
    optics_ground_label: int = 0  # метка для точек земли в результатах OPTICS

    statistic_percentile: float = 95

    sf_slope_threshold: float = 10.0   # порог угла, градусы
    sf_window_size: int = 5            # число соседей

    # Void filling settings
    slope_void_method: str = "linear"  # "cubic", "nearest", "linear"
    slope_grid_resolution: float = 0.5  # Resolution for grid interpolation

    # Mesh creation settings
    slope_mesh_method: str = "delaunay"  # Options: "delaunay", "poisson", "alpha_shape"
    slope_normal_radius: float = 0.1  # Radius for normal estimation
    slope_normal_max_nn: int = 30  # Maximum nearest neighbors for normal estimation
    slope_poisson_depth: int = 8  # Depth for Poisson reconstruction
    slope_alpha: float = 0.5  # Alpha value for alpha shape reconstruction
    
    # Slope coloring settings
    slope_color_method: str = "angle"  # Options: "gradient", "angle", "classified"
    slope_class_ranges: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0, 5),    # Flat
        (5, 15),   # Gentle
        (15, 30),  # Moderate
        (30, 45),  # Steep
        (45, 90)   # Very steep
    ])
    
    # Isohypse map settings
    slope_isohypse_method: str = "filled_contour"  # Options: "contour", "filled_contour", "hillshade"
    slope_contour_interval: float = 1.0  # Contour interval in meters
    slope_hillshade_exaggeration: float = 10.0  # Vertical exaggeration for hillshade

    tile_points: int = None
    overlap: int = None

    input_folder: str = "test_data"
    output_folder: str = "test_data"
    path_base: str = "."
    fname_points: str = "test_data.las"

    @staticmethod
    def from_yaml(yml_path: str) -> 'SlopeSettings':
        """Load settings from a YAML file."""
        with open(yml_path) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        return SlopeSettings(**data['slope_steepness'], **data['paths'])
