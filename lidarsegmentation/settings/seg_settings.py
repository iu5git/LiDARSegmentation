import yaml
from typing import Optional

class SS():
    def __init__(
        self, 
        path_base = None, 
        fname_points = None, 
        fname_shape = None, 
        csv_name_coord = None, 
        first_num = None, 
        STEP = None, 
        z_thresholds = None, 
        eps_steps = None, 
        min_pts = None,
        cluster_max_size = None,
        slenderness_min: Optional[int] = None,
        slenderness_max: Optional[int] = None,
        on_slenderness_fail: Optional[str] = None,
        **kwargs
    ):
        self.path_base = path_base
        self.fname_points = fname_points
        self.fname_shape = fname_shape
        self.csv_name_coord = csv_name_coord
        self.first_num = first_num
        self.STEP = STEP
        self.z_thresholds = z_thresholds
        self.eps_steps = eps_steps
        self.min_pts = min_pts
        self.cluster_max_size = cluster_max_size
        self.slenderness_min = slenderness_min
        self.slenderness_max = slenderness_max
        self.on_slenderness_fail = on_slenderness_fail
        self.step1_folder_name = 'vor'
        self.step2_folder_name = 'ram'
        self.step3_folder_name = 'clear'

    @staticmethod
    def from_yaml(yml_path: str) -> 'SS':
        with open(yml_path) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        return SS(**data['segmentation'], **data['paths'])