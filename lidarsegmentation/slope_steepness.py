import os
import numpy as np

import CSF
from scipy.interpolate import griddata
from scipy.interpolate import NearestNDInterpolator
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import grey_dilation, grey_erosion
from sklearn.cluster import DBSCAN, OPTICS
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import pyvista as pv

from lidarsegmentation.settings.slope_settings import SlopeSettings
from lidarsegmentation.classes.PCD import PCD
from lidarsegmentation.classes.PCD_UTILS import PCD_UTILS


class SlopeSteepness:
    """
    Класс для построения крутизны склонов из облаков точек LiDAR.

    Этот класс содержит статические методы для выполнения следующих операций:
    1. Выделение точек земли (удаление шума, деревьев, зданий).
    2. Заполнение пустот на поверхности земли.
    3. Создание полигональной сетки (mesh).
    4. Окрашивание точек по углу наклона.
    5. Генерация карты изогипс.
    6. Сохранение результатов.
    """

    @staticmethod
    def select_ground_points(points_obj, ss, method="progressive_morphology"):
        """
        Выделяет точки земли из облака точек, используя указанный метод.

        Args:
            points_obj (PCD): Входное облако точек.
            ss (SlopeSettings): Объект настроек.
            method (str): Метод для выделения точек земли.
                - 'progressive_morphology': Прогрессивная морфологическая фильтрация.
                - 'cloth_simulation': Фильтрация с помощью симуляции ткани.
                - 'ransac': RANSAC для поиска плоскости.
                - 'DBSCAN': Кластеризация DBSCAN.
                - 'OPTICS': Кластеризация OPTICS.
                - 'statistic_based_filter': Статистический фильтр на основе высоты.
                - 'slope_filter': Фильтрация на основе угла наклона.

        Returns:
            PCD: Объект PCD, содержащий только точки земли.
        """
        print(f"Selecting ground points using {method} method...")

        if points_obj is None:
            raise ValueError("Point cloud data not initialized")

        # Конвертация в облако точек Open3D для обработки
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_obj.points)
        points_np = points_obj.points

        if method == "DBSCAN":
            clustering = DBSCAN(eps=ss.dbscan_eps, min_samples=ss.dbscan_min_samples).fit(points_np)
            ground_label = ss.dbscan_ground_label
            ground_indices = np.where(clustering.labels_ == ground_label)[0]
            
            ground_points = PCD(
                points=points_obj.points[ground_indices],
                intensity=(points_obj.intensity[ground_indices] if points_obj.intensity is not None else None),
            )

        elif method == "OPTICS":
            clustering = OPTICS(max_eps=2, min_samples=ss.optics_min_samples).fit(points_np)
            ground_label = ss.optics_ground_label
            ground_indices = np.where(clustering.labels_ == ground_label)[0]
            ground_points = PCD(
                points=points_obj.points[ground_indices],
                intensity=(points_obj.intensity[ground_indices] if points_obj.intensity is not None else None),
            )

        elif method == "statistic_based_filter":
            z_threshold = np.percentile(points_np[:, 2], ss.statistic_percentile)
            ground_indices = np.where(points_np[:, 2] <= z_threshold)[0]
            
            ground_points = PCD(
                points=points_obj.points[ground_indices],
                intensity=(points_obj.intensity[ground_indices] if points_obj.intensity is not None else None),
            )

        elif method == "progressive_morphology":
            # Параметры прогрессивной морфологической фильтрации
            cell_size = ss.slope_cell_size
            max_window_size = ss.slope_max_window_size
            slope_threshold = ss.slope_threshold
            initial_distance = ss.slope_initial_distance
            max_distance = ss.slope_max_distance

            # Создание сетки
            x_min, y_min, _ = np.min(points_np, axis=0)
            x_max, y_max, _ = np.max(points_np, axis=0)
            x_grid = np.arange(x_min, x_max + cell_size, cell_size)
            y_grid = np.arange(y_min, y_max + cell_size, cell_size)

            height_grid = np.full((len(y_grid), len(x_grid)), np.inf)

            for i, (x, y, z) in enumerate(points_np):
                col = min(int((x - x_min) / cell_size), len(x_grid) - 1)
                row = min(int((y - y_min) / cell_size), len(y_grid) - 1)
                if z < height_grid[row, col]:
                    height_grid[row, col] = z

            valid_mask = ~np.isinf(height_grid)
            if np.any(valid_mask):
                y_indices, x_indices = np.where(valid_mask)
                valid_heights = height_grid[valid_mask]
                interpolator = NearestNDInterpolator(list(zip(y_indices, x_indices)), valid_heights)
                y_missing, x_missing = np.where(~valid_mask)
                if len(y_missing) > 0:
                    height_grid[y_missing, x_missing] = interpolator(y_missing, x_missing)

            filtered_grid = height_grid.copy()
            window_size = 3
            distance_threshold = initial_distance

            while window_size <= max_window_size:
                eroded = grey_erosion(filtered_grid, size=(window_size, window_size))
                dilated = grey_dilation(eroded, size=(window_size, window_size))
                height_diff = height_grid - dilated
                if window_size > 3:
                    distance_threshold = min(slope_threshold * (window_size - 3) * cell_size + initial_distance, max_distance)
                filtered_grid[height_diff > distance_threshold] = dilated[height_diff > distance_threshold]
                window_size = 2 * window_size - 1

            ground_indices = []
            for i, (x, y, z) in enumerate(points_np):
                col = min(int((x - x_min) / cell_size), len(x_grid) - 1)
                row = min(int((y - y_min) / cell_size), len(y_grid) - 1)
                if abs(z - filtered_grid[row, col]) <= ss.slope_ground_threshold:
                    ground_indices.append(i)
            
            ground_points = PCD(
                points=points_obj.points[ground_indices],
                intensity=(points_obj.intensity[ground_indices] if points_obj.intensity is not None else None),
            )

        elif method == "cloth_simulation":
            csf = CSF.CSF()

            # prameter settings
            # csf.params.bSloopSmooth = False
            csf.params.cloth_resolution = ss.slope_cloth_resolution
            csf.params.iterations = ss.slope_max_iterations
            csf.params.class_threshold = ss.slope_height_threshold

            csf.setPointCloud(pcd.points)
            ground = CSF.VecInt()  # a list to indicate the index of ground points after calculation
            non_ground = CSF.VecInt() # a list to indicate the index of non-ground points after calculation
            csf.do_filtering(ground, non_ground) # do actual filtering.

            ground_points = PCD(
                points=points_obj.points[np.array(ground)],
                intensity=(points_obj.intensity[np.array(ground)] if points_obj.intensity is not None else None),
            )

        elif method == "ransac":
            plane_model, inliers = pcd.segment_plane(
                distance_threshold=ss.slope_ransac_threshold,
                ransac_n=3,
                num_iterations=ss.slope_ransac_iterations,
            )
            ground_points = PCD(
                points=points_obj.points[inliers],
                intensity=(points_obj.intensity[inliers] if points_obj.intensity is not None else None),
            )

        elif method == "slope_filter":
            # Для каждой точки ищем ближайших соседей (по XY), считаем угол наклона к ним, если <= threshold - земля
            
            k = getattr(ss, "sf_window_size", 5)
            slope_thr_deg = getattr(ss, "sf_slope_threshold", 10.0)
            points_xy = points_np[:, :2]
            z = points_np[:, 2]
            nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(points_xy)
            distances, indices = nbrs.kneighbors(points_xy)

            slope_thr_rad = np.deg2rad(slope_thr_deg)
            ground_mask = np.zeros(len(points_np), dtype=bool)

            for i in range(len(points_np)):
                slopes = []
                z0 = z[i]
                x0, y0 = points_xy[i]
                for j in range(1, k+1):
                    idx = indices[i, j]
                    dz = np.abs(z0 - z[idx])
                    dxy = np.linalg.norm(points_xy[i] - points_xy[idx])
                    if dxy > 0:
                        slope_rad = np.arctan(dz / dxy)
                        slopes.append(slope_rad)
                # если максимальный угол к соседям меньше порога, точка - земля
                if slopes and np.max(slopes) <= slope_thr_rad:
                    ground_mask[i] = True

            ground_indices = np.where(ground_mask)[0]
            ground_points = PCD(
                points=points_obj.points[ground_indices],
                intensity=(points_obj.intensity[ground_indices] if points_obj.intensity is not None else None),
            )

        else:
            raise ValueError(f"Unknown ground point selection method: {method}")

        return ground_points

    @staticmethod
    def fill_voids(ground_points, ss, method="idw"):
        """
        Заполняет пустоты в поверхности земли, используя указанный метод.

        Args:
            ground_points (PCD): Облако точек земли.
            ss (SlopeSettings): Объект настроек.
            method (str): Метод для заполнения пустот ('linear', 'cubic', 'nearest').

        Returns:
            tuple: (PCD, dict)
                - PCD: Облако точек с заполненными пустотами (регулярная сетка).
                - dict: Цифровая модель рельефа (DEM).
        """
        if method in ["cubic", "nearest", "linear"]:
            interpolation_method = method
            print(f"Заполнение пустот методом: '{interpolation_method}'...")
        else:
            raise ValueError(f"Неизвестный метод заполнения пустот: {method}")

        if ground_points is None:
            raise ValueError("Точки земли еще не выделены.")

        points = ground_points.points
        if len(points) < 4:
            raise ValueError(f"Недостаточно точек земли ({len(points)}) для интерполяции.")

        x_min, y_min, _ = np.min(points, axis=0)
        x_max, y_max, _ = np.max(points, axis=0)
        grid_resolution = ss.slope_grid_resolution
        x_grid = np.arange(x_min, x_max, grid_resolution)
        y_grid = np.arange(y_min, y_max, grid_resolution)
        X, Y = np.meshgrid(x_grid, y_grid)

        Z = griddata(points[:, :2], points[:, 2], (X, Y), method=interpolation_method)

        nan_mask = np.isnan(Z)
        if np.any(nan_mask):
            print(f"Найдено {np.sum(nan_mask)} точек вне выпуклой оболочки. Экстраполяция методом ближайшего соседа...")
            z_nearest = griddata(points[:, :2], points[:, 2], (X[nan_mask], Y[nan_mask]), method="nearest")
            Z[nan_mask] = z_nearest

        dem = {"X": X, "Y": Y, "Z": Z, "resolution": grid_resolution}
        grid_points = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
        filled_points = PCD(
            points=grid_points,
            intensity=(np.zeros(len(grid_points)) if ground_points.intensity is None else None),
        )

        return filled_points, dem

    @staticmethod
    def create_mesh(dem, ground_points, ss, method="delaunay"):
        """
        Создает полигональную сетку (mesh) из заполненных точек земли.

        Args:
            dem (dict): Цифровая модель рельефа.
            ground_points (PCD): Облако точек земли (используется для некоторых методов).
            ss (SlopeSettings): Объект настроек.
            method (str): Метод создания сетки ('delaunay', 'poisson', 'alpha_shape').

        Returns:
            pyvista.PolyData: Объект сетки.
        """
        print(f"Creating mesh using {method} method...")

        if dem is None:
            raise ValueError("DEM not created yet")

        if method == "delaunay":
            X, Y, Z = dem["X"], dem["Y"], dem["Z"]
            grid = pv.StructuredGrid(X, Y, Z)
            mesh = grid.extract_surface()

        elif method == "poisson":
            if ground_points is None:
                raise ValueError("Ground points not selected yet")
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(ground_points.points)
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=ss.slope_normal_radius, max_nn=ss.slope_normal_max_nn))
            pcd.orient_normals_consistent_tangent_plane(100)
            mesh_o3d, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=ss.slope_poisson_depth, width=0, scale=1.1, linear_fit=False)
            vertices = np.asarray(mesh_o3d.vertices)
            faces = np.asarray(mesh_o3d.triangles)
            mesh = pv.PolyData(vertices, faces=np.column_stack((np.full(len(faces), 3), faces)))

        elif method == "alpha_shape":
            if ground_points is None:
                raise ValueError("Ground points not selected yet")
            points = ground_points.points
            cloud = pv.PolyData(points)
            mesh = cloud.delaunay_2d(alpha=ss.slope_alpha)

        else:
            raise ValueError(f"Unknown mesh creation method: {method}")

        return mesh

    @staticmethod
    def color_by_slope(mesh, ss, method="gradient"):
        """
        Окрашивает сетку по углу наклона.

        Args:
            mesh (pyvista.PolyData): Входная сетка.
            ss (SlopeSettings): Объект настроек.
            method (str): Метод окраски ('gradient', 'angle', 'classified').

        Returns:
            pyvista.PolyData: Окрашенная сетка.
        """
        print(f"Coloring mesh by slope using {method} method...")
        if mesh is None:
            raise ValueError("Mesh not created yet")
        
        colored_mesh = mesh.copy()
        if "Normals" not in colored_mesh.array_names:
            colored_mesh.compute_normals(inplace=True, cell_normals=False, point_normals=True)

        normals = colored_mesh.point_data["Normals"]

        if method == "gradient":
            slopes = np.sqrt(normals[:, 0] ** 2 + normals[:, 1] ** 2) / np.abs(normals[:, 2])
            colored_mesh.point_data["Slope"] = slopes
        elif method == "angle":
            slope_angles = np.degrees(np.arccos(np.abs(normals[:, 2])))
            colored_mesh.point_data["SlopeAngle"] = slope_angles
        elif method == "classified":
            slope_angles = np.degrees(np.arccos(np.abs(normals[:, 2])))
            slope_classes = np.zeros_like(slope_angles)
            class_ranges = ss.slope_class_ranges
            for i, (min_angle, max_angle) in enumerate(class_ranges):
                mask = (slope_angles >= min_angle) & (slope_angles < max_angle)
                slope_classes[mask] = i + 1
            colored_mesh.point_data["SlopeClass"] = slope_classes
        else:
            raise ValueError(f"Unknown slope coloring method: {method}")

        return colored_mesh

    @staticmethod
    def generate_isohypse_map(dem, ss, output_path, method="contour"):
        """
        Генерирует 2D карту изогипс (контуров) из DEM.

        Args:
            dem (dict): Цифровая модель рельефа.
            ss (SlopeSettings): Объект настроек.
            output_path (str): Путь для сохранения карты.
            method (str): Метод генерации ('contour', 'filled_contour', 'hillshade').

        Returns:
            str: Путь к сохраненной карте.
        """
        print(f"Generating isohypse map using {method} method...")
        if dem is None:
            raise ValueError("DEM not created yet")

        X, Y, Z = dem["X"], dem["Y"], dem["Z"]
        fig, ax = plt.subplots(figsize=(10, 8))
        min_height, max_height = np.nanmin(Z), np.nanmax(Z)
        contour_interval = ss.slope_contour_interval
        levels = np.arange(np.floor(min_height / contour_interval) * contour_interval, np.ceil(max_height / contour_interval) * contour_interval + contour_interval, contour_interval)

        if method == "contour":
            contours = ax.contour(X, Y, Z, levels=levels, colors="k", linewidths=0.5)
            ax.clabel(contours, inline=True, fontsize=8, fmt="%1.1f")
        elif method == "filled_contour":
            contourf = ax.contourf(X, Y, Z, levels=levels, cmap="terrain")
            contours = ax.contour(X, Y, Z, levels=levels, colors="k", linewidths=0.5)
            ax.clabel(contours, inline=True, fontsize=8, fmt="%1.1f")
            fig.colorbar(contourf, ax=ax, label="Elevation (m)")
        elif method == "hillshade":
            ls = LightSource(azdeg=315, altdeg=45)
            hillshade = ls.hillshade(Z, vert_exag=ss.slope_hillshade_exaggeration)
            ax.imshow(hillshade, extent=[X.min(), X.max(), Y.min(), Y.max()], cmap="gray", origin="lower", alpha=0.5)
            contours = ax.contour(X, Y, Z, levels=levels, colors="k", linewidths=0.5)
            ax.clabel(contours, inline=True, fontsize=8, fmt="%1.1f")
        else:
            raise ValueError(f"Unknown isohypse map method: {method}")

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("Isohypse Map")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

        return output_path

    @staticmethod
    def merge_all_tiles(tile_dems, grid_resolution: float):
        """
        Объединяет все DEM тайлов в одну глобальную DEM.

        Args:
            tile_dems (list): Список словарей DEM для каждого тайла.
            grid_resolution (float): Разрешение для объединенной DEM.

        Returns:
            dict: Объединенная DEM.
        """
        print("Merging all tiles into a global DEM...")
        if not tile_dems:
            raise ValueError("Нет данных для объединения.")

        X_all = np.concatenate([dem['X'].flatten() for dem in tile_dems])
        Y_all = np.concatenate([dem['Y'].flatten() for dem in tile_dems])
        Z_all = np.concatenate([dem['Z'].flatten() for dem in tile_dems])

        x_min, x_max = np.min(X_all), np.max(X_all)
        y_min, y_max = np.min(Y_all), np.max(Y_all)
        x_grid = np.arange(x_min, x_max + grid_resolution, grid_resolution)
        y_grid = np.arange(y_min, y_max + grid_resolution, grid_resolution)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

        print("Интерполяция в единую сетку...")
        Z_grid = griddata((X_all, Y_all), Z_all, (X_grid, Y_grid), method="linear")

        nan_mask = np.isnan(Z_grid)
        if np.any(nan_mask):
            print("Заполнение оставшихся пустот методом ближайшего соседа...")
            Z_grid[nan_mask] = griddata((X_all, Y_all), Z_all, (X_grid[nan_mask], Y_grid[nan_mask]), method="nearest")

        return {"X": X_grid, "Y": Y_grid, "Z": Z_grid, "resolution": grid_resolution}

    @staticmethod
    def save_results(output_dir, ss, ground_points=None, mesh=None, dem=None, prefix=''):
        """
        Сохраняет результаты обработки в указанную директорию.

        Args:
            output_dir (str): Директория для сохранения.
            ss (SlopeSettings): Объект настроек.
            ground_points (PCD, optional): Точки земли для сохранения.
            mesh (pyvista.PolyData, optional): Сетка для сохранения.
            dem (dict, optional): DEM для сохранения.
            prefix (str, optional): Префикс для имен файлов.

        Returns:
            dict: Словарь с путями к сохраненным файлам.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        results = {}

        if ground_points is not None:
            ground_points_path = os.path.join(output_dir, f'{prefix}ground_points.pcd')
            ground_points.save(ground_points_path)
            results['ground_points'] = ground_points_path

        if mesh is not None:
            mesh_path = os.path.join(output_dir, f'{prefix}slope_mesh.vtk')
            mesh.save(mesh_path)
            results['mesh'] = mesh_path

            screenshot_path = os.path.join(output_dir, f'{prefix}slope_mesh_view.png')
            plotter = pv.Plotter(off_screen=True)
            if 'SlopeAngle' in mesh.point_data:
                plotter.add_mesh(mesh, scalars='SlopeAngle', cmap='terrain', show_edges=False)
                plotter.add_scalar_bar(title='Slope Angle (°)')
            elif 'Slope' in mesh.point_data:
                plotter.add_mesh(mesh, scalars='Slope', cmap='terrain', show_edges=False)
                plotter.add_scalar_bar(title='Slope Gradient')
            else:
                plotter.add_mesh(mesh)
            plotter.screenshot(screenshot_path)
            plotter.close()
            results['screenshot'] = screenshot_path

        if dem is not None:
            dem_path = os.path.join(output_dir, f'{prefix}dem.npz')
            np.savez(dem_path, X=dem['X'], Y=dem['Y'], Z=dem['Z'], resolution=dem['resolution'])
            results['dem'] = dem_path

            isohypse_path = os.path.join(output_dir, f'{prefix}isohypse_map.png')
            SlopeSteepness.generate_isohypse_map(dem, ss, isohypse_path, method='filled_contour')
            results['isohypse_map'] = isohypse_path

        return results


def slope_steepness_task(
    ss: SlopeSettings,
    input_pcd=None,
    run_ground_selection: bool = True,
    run_void_filling: bool = True,
    run_global_merge: bool = True,
    run_meshing_and_coloring: bool = True,
    run_map_generation: bool = True,
):
    """
    Выполняет полную задачу по построению крутизны склона.
    """
    print("--- Запуск задачи по анализу крутизны склонов ---")

    output_dir = os.path.join(ss.path_base, ss.slope_output_folder)
    os.makedirs(output_dir, exist_ok=True)

    if input_pcd is None:
        input_path = os.path.join(ss.path_base, ss.fname_points)
        input_pcd = PCD()
        input_pcd.open(input_path)
        print(f"Загружено облако точек из {input_path}: {len(input_pcd.points)} точек")

    tiles = PCD_UTILS.split_points_to_tiles(
        input_pcd.points, tile_points=ss.tile_points, overlap=ss.overlap
    ) if ss.tile_points is not None else [{"points": input_pcd.points, "bounds": None, "index": "full"}]

    all_tile_dems = []

    for tile_info in tiles:
        pts, tile_idx = tile_info["points"], tile_info["index"]
        prefix = f"tile_{tile_idx[0]}_{tile_idx[1]}_" if tile_idx != "full" else ""
        print(f"\n--- Обработка тайла {tile_idx} ({pts.shape[0]} точек) ---")

        tile_pcd = PCD(points=pts)
        
        ground_points_dir = os.path.join(output_dir, "ground_points")
        gp_path = os.path.join(ground_points_dir, f"{prefix}ground_points.pcd")
        if run_ground_selection:
            print("Этап 1: Выделение точек земли...")
            ground_pts = SlopeSteepness.select_ground_points(tile_pcd, ss, method=ss.slope_ground_method)
            print(f"Найдено {len(ground_pts.points)} точек земли.")
            if ss.save_after_stage:
                os.makedirs(ground_points_dir, exist_ok=True)
                ground_pts.save(gp_path)
        else:
            print("Этап 1: Пропуск. Загрузка точек земли из файла...")
            ground_pts = PCD()
            ground_pts.open(gp_path)

        if len(ground_pts.points) == 0:
            print(f"Внимание: для тайла {tile_idx} не найдено точек земли. Пропускаем.")
            continue

        filled_points_dir = os.path.join(output_dir, "filled_points")
        dem_path = os.path.join(filled_points_dir, f"{prefix}dem.npz")
        if run_void_filling:
            print("Этап 2: Заполнение пустот...")
            _, dem = SlopeSteepness.fill_voids(ground_pts, ss, method=ss.slope_void_method)
            if ss.save_after_stage:
                os.makedirs(filled_points_dir, exist_ok=True)
                np.savez(dem_path, **dem)
        else:
            print("Этап 2: Пропуск. Загрузка DEM из файла...")
            dem_data = np.load(dem_path)
            dem = {key: dem_data[key] for key in dem_data}

        all_tile_dems.append(dem)

    global_results_dir = os.path.join(output_dir, "global_results")
    global_dem_path = os.path.join(global_results_dir, "global_dem.npz")
    
    if run_global_merge:
        print("\n--- Этап 3: Объединение DEM всех тайлов ---")
        global_dem = SlopeSteepness.merge_all_tiles(all_tile_dems, grid_resolution=ss.slope_grid_resolution)
        os.makedirs(global_results_dir, exist_ok=True)
        np.savez(global_dem_path, **global_dem)
    else:
        print("\n--- Этап 3: Пропуск. Загрузка глобальной DEM ---")
        dem_data = np.load(global_dem_path)
        global_dem = {key: dem_data[key] for key in dem_data}

    mesh_path = os.path.join(global_results_dir, "global_slope_mesh.vtk")
    if run_meshing_and_coloring:
        print("\n--- Этап 4: Создание и окраска глобальной сетки ---")
        # Для методов poisson и alpha_shape могут потребоваться исходные точки земли
        # Здесь для простоты предполагаем, что они не нужны или что можно загрузить все
        mesh = SlopeSteepness.create_mesh(global_dem, None, ss, method=ss.slope_mesh_method)
        colored_mesh = SlopeSteepness.color_by_slope(mesh, ss, method=ss.slope_color_method)
        colored_mesh.save(mesh_path)
        print(f"Окрашенная сетка сохранена в {mesh_path}")
        SlopeSteepness.save_results(global_results_dir, ss, mesh=colored_mesh, prefix="global_")
    else:
        print("\n--- Этап 4: Пропуск создания сетки ---")

    if run_map_generation:
        print("\n--- Этап 5: Генерация карты изолиний ---")
        isohypse_path = os.path.join(global_results_dir, "global_isohypse_map.png")
        SlopeSteepness.generate_isohypse_map(global_dem, ss, isohypse_path, method=ss.slope_isohypse_method)
        print(f"Карта изолиний сохранена в {isohypse_path}")
    else:
        print("\n--- Этап 5: Пропуск генерации карты изолиний ---")

    print("\n--- Обработка завершена ---")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run slope steepness construction task")
    parser.add_argument("--settings", type=str, default="./lidarsegmentation/settings/slope_settings.yaml", help="Path to settings YAML file")
    args = parser.parse_args()
    ss = SlopeSettings.from_yaml(args.settings)
    slope_steepness_task(ss)
