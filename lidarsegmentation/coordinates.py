import os
from lidarsegmentation.classes.PCD import PCD
from lidarsegmentation.classes.PCD_AREA import PCD_AREA
from lidarsegmentation.classes.PCD_UTILS import PCD_UTILS
from lidarsegmentation.classes.CELL import CELL
from lidarsegmentation.classes.VOR_TES import VOR_TES
from lidarsegmentation.settings.coord_settings import CS
import numpy as np
import pandas as pd
import circle_fit as cf
import statistics
import math
from tqdm import tqdm


def coordinates(intensity_cut_make, cs):
    # Return dict: map of stump name to PCD objects
    pcd_map = {}
    
    if cs.fname_traj is not None:
        file_name_traj = os.path.join(cs.path_base, cs.fname_traj)
    if cs.fname_points is not None:
        file_name_data = os.path.join(cs.path_base, cs.fname_points)
    if cs.fname_shape is not None:
        file_shape = os.path.join(cs.path_base, cs.fname_shape)

    # Keep track of PC data in memory instead of saving to file
    pc_area = None
    pc_traj = None
    
    if cs.cut_data_method == 'flood_fill' and (cs.FLAG_cut_data or cs.FLAG_make_cells):
        pc_traj = PCD()
        pc_traj.open(file_name_traj)
        pc_traj.points = PCD_UTILS.shift(pc_traj.points, cs.x_shift, cs.y_shift, cs.z_shift)

    if cs.FLAG_cut_data:
        pc_area = PCD_AREA()
        pc_area.open(file_name_data, verbose=True)
        pc_area.points = PCD_UTILS.shift(pc_area.points, cs.x_shift, cs.y_shift, cs.z_shift)

        if file_shape is not None and os.path.exists(file_shape):
            shp_poly = PCD_UTILS.shp_open(file_shape)
            shp_poly = PCD_UTILS.shift(shp_poly, cs.x_shift, cs.y_shift, cs.z_shift)
            # Cut by polygon shape
            pc_area = pc_area.poly_cut(shp_poly)
        else:
            print('Warning: File of area boundary not found. The boundaries of the area are selected as the entire loaded area.')
            shp_poly = PCD_UTILS.shp_create(pc_area)

        print('Starting cutting main pcd ...')
        # Cut by height
        pc_area.index_cut(np.where((pc_area.points[:, 2] > cs.LOW) & (pc_area.points[:, 2] <= cs.UP)))
        # Cut by intensity
        pc_area.index_cut(np.where(pc_area.intensity >= cs.intensity_cut))

    # Store cells in memory instead of on disk
    cells_dict = {}
    
    if cs.FLAG_make_cells:
        if cs.FLAG_cut_data:
            idx_labels = np.where(pc_area.intensity >= intensity_cut_make)
            pc_area.index_cut(idx_labels)
        else:
            # Open and process the data directly
            pc_area = PCD_AREA()
            pc_area.open(file_name_data)
            pc_area.index_cut(np.where((pc_area.points[:, 2] > cs.LOW) & (pc_area.points[:, 2] <= cs.UP)))
            pc_area.index_cut(np.where(pc_area.intensity >= intensity_cut_make))
            pc_area.points = PCD_UTILS.shift(pc_area.points, cs.x_shift, cs.y_shift, cs.z_shift)

            if file_shape is not None and os.path.exists(file_shape):
                shp_poly = PCD_UTILS.shp_open(file_shape)
                shp_poly = PCD_UTILS.shift(shp_poly, cs.x_shift, cs.y_shift, cs.z_shift)
            else:
                shp_poly = PCD_UTILS.shp_create(pc_area)

        print('Starting extracting areas (cells) traj-based ...')

        if cs.cut_data_method == 'voronoi_tessellation':
            vortes = VOR_TES(
                points=pc_area.points,
                intensity=pc_area.intensity,
                algo=cs.algo,
                n_clusters=cs.n_clusters,
                intensity_cut=cs.intensity_cut_vor_tes,
            )
            # Instead of writing to disk, store the regions in memory
            cells_dict = vortes.select_clusters(shp_poly)
        elif cs.cut_data_method == 'flood_fill':
            cell = CELL(points=pc_area.points, intensity=pc_area.intensity, points_traj=pc_traj.points, cell_size=cs.cell_size)
            cell.make_cell_list(pc_area.points.min(axis=0), pc_area.points.max(axis=0), verbose=True)
            # Instead of writing to disk, store cells in memory
            cells_dict = cell.save_all_cells()
        elif cs.cut_data_method == 'none':
            # No cells to process
            pass
        else:
            raise Exception("There is no such algorithm. Choose from existing: 'voronoi_tessellation', 'flood_fill', 'none'")

        print(f'\n {len(cells_dict)} areas (cells) have been processed in memory')

    if cs.FLAG_make_stumps:
        TN, TCX, TCY, TD, stump_pcds = extract_stumps(cs, intensity_cut_make, cells_dict)

        # Update pcd_map with stumps for this intensity
        for i, name in enumerate(TN):
            pcd_map[name] = stump_pcds[i]
            
        # Create DataFrame in memory
        df = pd.DataFrame(
            {'Name_stump' + '_int' + str(intensity_cut_make): TN, 'X': TCX, 'Y': TCY, 'Diameter' + '_int' + str(intensity_cut_make): TD}
        )
        
        return df, pcd_map
    
    # Return empty DataFrame and map if FLAG_make_stumps is False
    return pd.DataFrame(), {}


def extract_stumps(cs, intensity_cut_make, cells_dict):
    """Extract all stumps from segmented cells and return arrays of (name, x, y, diameter) and PCD objects."""
    records = []
    stump_pcds = []
    
    # Process each cell
    for cell_id, cell_pcd in cells_dict.items():
        cell_records, cell_pcds = process_cell(cell_pcd, cs, intensity_cut_make)
        records.extend(cell_records)
        stump_pcds.extend(cell_pcds)

    if not records:
        return np.array([]), np.array([]), np.array([]), np.array([]), []
    
    names, xs, ys, ds = zip(*records)
    return np.asarray(names), np.asarray(xs), np.asarray(ys), np.asarray(ds), stump_pcds


def process_cell(pc_cells, cs, intensity_cut_make):
    """Process a single cell PCD object and return a list of stump records and PCDs."""
    # Convert PCD_AREA to CELL if needed
    if not hasattr(pc_cells, 'extract_stumps_labels'):
        # Create a CELL object from the PCD_AREA points and intensity
        pc_cells = CELL(points=pc_cells.points, intensity=pc_cells.intensity)
    
    labels = pc_cells.extract_stumps_labels()
    recs = []
    pcds = []
    
    for lbl in np.unique(labels):
        if lbl < 0:
            continue
        rec, pcd = process_single_stump(pc_cells, labels, lbl, cs, intensity_cut_make)
        if rec:
            recs.append(rec)
            pcds.append(pcd)
    
    return recs, pcds


def process_single_stump(pc_cells, labels, lbl, cs, intensity_cut_make):
    """Process one stump label and return (filename, x, y, diameter) and PCD or None."""
    idx = np.where(labels == lbl)
    pts = pc_cells.points[idx]
    ints = pc_cells.intensity[idx]
    
    # Filter by height_limit_1
    if pts[:, 2].ptp() < cs.height_limit_1:
        return None, None
    
    # Denoise
    pts, ints = PCD_UTILS.SOR(pts, ints)
    
    # XY clustering
    xy_labels = CELL(pts, ints).labels_XY_dbscan(eps=cs.eps_XY, max_points=cs.max_points_to_process_XY)
    best = None
    best_pcd = None
    
    for xyl in np.unique(xy_labels):
        if xyl < 0:
            continue
        rec, pcd = process_xy_cluster(pts, ints, xy_labels, xyl, cs, intensity_cut_make)
        if rec and (best is None or rec[3] > best[3]):
            best = rec
            best_pcd = pcd
    
    return best, best_pcd


# Initialize counter for stump names
stump_counter = 0

def process_xy_cluster(pts, ints, xy_labels, xyl, cs, intensity_cut_make):
    """Process one XY sub-cluster, return stump data and PCD or None."""
    global stump_counter
    idx = np.where(xy_labels == xyl)
    sub_pts, sub_ints = pts[idx], ints[idx]
    
    if sub_pts[:, 2].ptp() < cs.height_limit_2:
        return None, None
    
    # Z clustering
    z_labels = CELL(sub_pts, sub_ints).label_Z_dbscan(eps=cs.eps_Z)
    
    # pick largest Z cluster
    counts = {z: np.sum(z_labels == z) for z in np.unique(z_labels) if z >= 0}
    if not counts:
        return None, None
    
    best_z = max(counts, key=counts.get)
    idxz = np.where(z_labels == best_z)
    pts_z, ints_z = sub_pts[idxz], sub_ints[idxz]
    
    # Fit circles in layers
    cx, cy, r = fit_circle_layers(pts_z, cs)
    if r <= 0:
        return None, None
    
    # Quality check
    cx, cy, r = quality_check(pts_z, cx, cy, r)
    
    # Create stump name and return PCD
    pname = f'int{intensity_cut_make}_{stump_counter:04d}.pcd'
    stump_counter += 1
    
    # Create PCD object in memory
    pcd = PCD(pts_z, ints_z)
    
    return (pname, cx, cy, r * 2), pcd


def fit_circle_layers(points, cs):
    """Slice points into layers, fit circle on each, return median x, y, r."""
    zmin, zmax = points[:, 2].min(), points[:, 2].max()
    if zmax - zmin <= 1:
        return 0, 0, 0
    layers = 4
    dz = (zmax - zmin) / layers
    radii, centers = [], []
    for i in range(layers):
        slab = points[(points[:, 2] >= zmin + i * dz) & (points[:, 2] < zmin + (i + 1) * dz)]
        try:
            xc, yc, rad, _ = cf.hyper_fit(slab)
        except Exception as e:
            xc, yc, rad = 0, 0, 0
        radii.append(rad)
        centers.append((xc, yc))
    xs = [c[0] for c in centers]
    ys = [c[1] for c in centers]
    return statistics.median(xs), statistics.median(ys), statistics.median(radii)


def quality_check(points, cx, cy, r):
    """Adjust r and center based on bounding box and medians."""
    xmin, ymin = points[:, :2].min(axis=0)
    xmax, ymax = points[:, :2].max(axis=0)
    check_r = ((xmax - xmin) + (ymax - ymin)) / 4
    if (r > 0.65 or r > 2.1 * check_r) or r == 0:
        r = check_r
    midx, midy = np.median(points[:, 0]), np.median(points[:, 1])
    if math.hypot(cx - midx, cy - midy) > 0.25:
        cx, cy = (midx, midy)
    return cx, cy, r


if __name__ == '__main__':
    yml_path = 'settings\settings.yaml'
    cs = CS.from_yaml(yml_path)
    intensity_cut_make = 7000
    df, pcd_map = coordinates(intensity_cut_make=intensity_cut_make, cs=cs)
    print(f"Generated {len(pcd_map)} stumps")
    print(df.head())
