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
    # Имя создаваемого файла с обрезанными данными облака по высоте и границам участка (.pcd)
    fname_data_cut = cs.fname_points.partition('.')[0] + '_cut_int' + str(cs.intensity_cut) + '.pcd'
    # Имя создаваемого файла в папке path_base/cells/stumps/ (.csv)
    csv_name_coord = cs.fname_points.partition('.')[0] + '_Coordinates_int' + str(intensity_cut_make) + '.csv'

    if cs.fname_traj is not None:
        file_name_traj = os.path.join(cs.path_base, cs.fname_traj)
    if cs.fname_points is not None:
        file_name_data = os.path.join(cs.path_base, cs.fname_points)
    if cs.fname_shape is not None:
        file_shape = os.path.join(cs.path_base, cs.fname_shape)
    if fname_data_cut is not None:
        file_name_data_cut = os.path.join(cs.path_base, fname_data_cut)
    if csv_name_coord is not None:
        file_name_csv = os.path.join(cs.path_base, csv_name_coord)

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

        # Save cut pcd to file
        pc_area.save(file_name_data_cut)

    path_int = os.path.join(cs.path_base, 'int' + str(intensity_cut_make))
    os.makedirs(path_int, exist_ok=True)

    path_file_cells = os.path.join(cs.path_base, path_int, cs.cut_data_method + '_cells')
    os.makedirs(path_file_cells, exist_ok=True)

    if cs.FLAG_make_cells:
        if cs.FLAG_cut_data:
            idx_labels = np.where(pc_area.intensity >= intensity_cut_make)
            pc_area.index_cut(idx_labels)
        else:
            pc_area = PCD_AREA()
            pc_area.open(file_name_data_cut)
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
            vortes.select_borders(path_file_cells, shp_poly, verbose=False)
            vortes.select_clusters(path_file_cells)
        elif cs.cut_data_method == 'flood_fill':
            cell = CELL(points=pc_area.points, intensity=pc_area.intensity, points_traj=pc_traj.points, cell_size=cs.cell_size)
            cell.make_cell_list(pc_area.points.min(axis=0), pc_area.points.max(axis=0), verbose=True)
            cell.save_all_cells(path_file_cells, verbose=True)
        elif cs.cut_data_method == 'none':
            path_file_stumps = os.path.join(cs.path_base, 'stumps')
            os.makedirs(path_file_stumps, exist_ok=True)
        else:
            raise Exception("There is no such algorithm. Choose from existing: 'voronoi_tessellation', 'flood_fill', 'none'")

        print(f'\n {cs.n_clusters} areas (cells) have been saved to the folder {path_file_cells}')

    if cs.FLAG_make_stumps:
        TN, TCX, TCY, TD = extract_stumps(cs, intensity_cut_make, path_file_cells)

        bd = pd.DataFrame(
            {'Name_stump' + '_int' + str(intensity_cut_make): TN, 'X': TCX, 'Y': TCY, 'Diameter' + '_int' + str(intensity_cut_make): TD}
        )
        bd.to_csv(file_name_csv, index=False, sep=';')

        file = open(os.path.join(cs.path_base, 'coordinates_paths.txt'), 'a')
        file.write('\n' + file_name_csv)
        file.close()


def extract_stumps(cs, intensity_cut_make, path_file_cells):
    """Extract all stumps from segmented cells and return arrays of (name, x, y, diameter)."""
    records = []
    path_file_stumps = os.path.join(path_file_cells, 'stumps')
    os.makedirs(path_file_stumps, exist_ok=True)

    for cell_file in os.listdir(path_file_cells):
        if not cell_file.endswith('.pcd'):
            continue
        cell_path = os.path.join(path_file_cells, cell_file)
        records.extend(process_cell(cell_path, cs, intensity_cut_make, path_file_stumps))

    if not records:
        return np.array([]), np.array([]), np.array([]), np.array([])
    names, xs, ys, ds = zip(*records)
    return np.asarray(names), np.asarray(xs), np.asarray(ys), np.asarray(ds)


def process_cell(cell_path, cs, intensity_cut_make, path_file_stumps):
    """Process a single cell PCD file and return a list of stump records."""
    pc_cells = CELL()
    pc_cells.open(cell_path)
    labels = pc_cells.extract_stumps_labels()
    recs = []
    for lbl in np.unique(labels):
        if lbl < 0:
            continue
        rec = process_single_stump(pc_cells, labels, lbl, cs, intensity_cut_make, path_file_stumps)
        if rec:
            recs.append(rec)
    return recs


def process_single_stump(pc_cells, labels, lbl, cs, intensity_cut_make, path_file_stumps):
    """Process one stump label and return (filename, x, y, diameter) or None."""
    idx = np.where(labels == lbl)
    pts = pc_cells.points[idx]
    ints = pc_cells.intensity[idx]
    # Filter by height_limit_1
    if pts[:, 2].ptp() < cs.height_limit_1:
        return None
    # Denoise
    pts, ints = PCD_UTILS.SOR(pts, ints)
    # XY clustering
    xy_labels = CELL(pts, ints).labels_XY_dbscan(eps=cs.eps_XY, max_points=cs.max_points_to_process_XY)
    best = None
    for xyl in np.unique(xy_labels):
        if xyl < 0:
            continue
        rec = process_xy_cluster(pts, ints, xy_labels, xyl, cs, intensity_cut_make, path_file_stumps)
        if rec and (best is None or rec[3] > best[3]):
            best = rec
    return best


def process_xy_cluster(pts, ints, xy_labels, xyl, cs, intensity_cut_make, path_file_stumps):
    """Process one XY sub-cluster, return stump or None."""
    idx = np.where(xy_labels == xyl)
    sub_pts, sub_ints = pts[idx], ints[idx]
    if sub_pts[:, 2].ptp() < cs.height_limit_2:
        return None
    # Z clustering
    z_labels = CELL(sub_pts, sub_ints).label_Z_dbscan(eps=cs.eps_Z)
    # pick largest Z cluster
    counts = {z: np.sum(z_labels == z) for z in np.unique(z_labels) if z >= 0}
    if not counts:
        return None
    best_z = max(counts, key=counts.get)
    idxz = np.where(z_labels == best_z)
    pts_z, ints_z = sub_pts[idxz], sub_ints[idxz]
    # Fit circles in layers
    cx, cy, r = fit_circle_layers(pts_z, cs)
    if r <= 0:
        return None
    # Quality check
    cx, cy, r = quality_check(pts_z, cx, cy, r)
    # Save stump PCD
    pname = save_stump_pcd(pts_z, ints_z, intensity_cut_make, path_file_stumps)
    return (pname, cx, cy, r * 2)


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


def save_stump_pcd(points, intensity, intensity_cut_make, path_file_stumps):
    """Save a stump PCD and return its filename."""
    save_stump_pcd.counter += 1
    fname = f'int{intensity_cut_make}_{str(save_stump_pcd.counter).rjust(4, "0")}.pcd'
    pc = PCD(points, intensity)
    pc.save(os.path.join(path_file_stumps, fname))
    return fname


save_stump_pcd.counter = 0


if __name__ == '__main__':
    yml_path = 'settings\settings.yaml'
    cs = CS.from_yaml(yml_path)
    intensity_cut_make = 7000
    coordinates(intensity_cut_make=intensity_cut_make, cs=cs)
