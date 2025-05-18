import numpy as np
from sklearn import cluster
import os
# from pycobra.visualisation import voronoi_finite_polygons_2d
from scipy.spatial import Voronoi
from shapely.geometry import Polygon
from lidarsegmentation.classes.PCD import PCD
from lidarsegmentation.classes.PCD_AREA import PCD_AREA
from lidarsegmentation.classes.PCD_UTILS import PCD_UTILS
from tqdm import tqdm
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import random
# from matplotlib.collections import PatchCollection

class VOR_TES(PCD):
    """ Clustering with Voronoi Tesselations """
    def __init__(self, points, intensity, algo = 'birch', n_clusters = 8, intensity_cut = 20000):
        super().__init__(points, intensity)
        self.n_clusters = n_clusters

        pc = PCD(points = self.points, intensity = self.intensity)
        idx_labels = np.where(pc.intensity>=intensity_cut)
        pc.index_cut(idx_labels)
        self.pts = pc.points[:,0:2]

        if algo == 'birch':
            self.algo = cluster.Birch(n_clusters=n_clusters)
        elif algo == 'spectral':
            self.algo = cluster.SpectralClustering(n_clusters=n_clusters, eigen_solver='arpack', affinity="nearest_neighbors")
        elif algo == 'kmeans':
            self.algo = cluster.KMeans(n_clusters=n_clusters)
        else:
            raise Exception("There is no such algorithm. Choose from existing: 'birch', 'spectral', 'kmeans'")

        print(f'Starting clustering via {algo} algorithm ...')
        self.algo.fit(self.pts)

    def plot_cluster_voronoi(self):
        vor = Voronoi(self.pts)
        regions, vertices = PCD_UTILS.voronoi_finite_polygons_2d(vor)
        fig, ax = plt.subplots()
        plot = ax.scatter([], [])
        indice = 0
        for region in regions:
            # ax.plot(self.pts[:,0][indice], self.pts[:,1][indice], 'ko')
            polygon = vertices[region]
            color = self.algo.labels_[indice]
            random.seed(color)
            clr = [random.random(), random.random(), random.random()]
            ax.fill(*zip(*polygon), alpha=0.4, color=clr, label="")
            indice += 1
        ax.axis('equal')
        plt.xlim(vor.min_bound[0] - 0.1, vor.max_bound[0] + 0.1)
        plt.ylim(vor.min_bound[1] - 0.1, vor.max_bound[1] + 0.1)
        plt.show()


    def select_borders(self, path_folder, shp_poly, verbose = False):
        print(f'Starting create Voronoi partitions ...')
        vor = Voronoi(self.pts)
        regions, vertices = PCD_UTILS.voronoi_finite_polygons_2d(vor)
        indice = 0
        mp = [[]]
        for i in range(self.n_clusters):
            mp.append([])
        if verbose:
            self.plot_cluster_voronoi()
        with tqdm(total=self.pts.shape[0]) as pbar:
            for region in regions:
                polygon = vertices[region]
                label = self.algo.labels_[indice]
                poly = Polygon(polygon)
                for i in range(self.n_clusters):
                    if label == i:
                        mp[i].append(poly)
                        break
                indice += 1
                pbar.update(1)
        
        print(f'Starting saving borders in .csv files ...')

        for i in tqdm(range(self.n_clusters)):
            multi_poly = unary_union(mp[i])

            shp_ply = Polygon(shp_poly)
            cut_poly = multi_poly.intersection(shp_ply)

            if verbose:
                fig, axs = plt.subplots()
                axs.set_aspect('equal', 'datalim')
                for geom in cut_poly.geoms:    
                    xs, ys = geom.exterior.xy    
                    axs.fill(xs, ys, alpha=0.5, fc='r', ec='none')
                plt.show()

                fig, axs = plt.subplots()
                axs.set_aspect('equal', 'datalim')   
                xs, ys = multi_poly.exterior.xy    
                axs.fill(xs, ys, alpha=0.5, fc='b', ec='none')
                xs, ys = shp_ply.exterior.xy    
                axs.fill(xs, ys, alpha=0.5, fc='g', ec='none')
                plt.show()
            
            if str(type(cut_poly)) == "<class 'shapely.geometry.polygon.Polygon'>":
                polygon_xy = np.asarray(cut_poly.exterior.coords.xy)       
                polygon = np.vstack((polygon_xy[0], polygon_xy[1])).T.reshape(-1, 2)
                z = np.zeros(polygon.shape[0])
                dt1 = np.c_[polygon, z]
                dt1 = np.array(dt1, dtype=np.float32)
                filename_border = str(i).rjust(4, '0') + '.csv'
                fname_brd = os.path.join(path_folder, filename_border) 
                np.savetxt(fname_brd, dt1, delimiter=',', header='x,y,z')

            elif str(type(cut_poly)) == "<class 'shapely.geometry.multipolygon.MultiPolygon'>":
                j = 0
                for poly_item in cut_poly.geoms:
                    polygon_xy = np.asarray(poly_item.exterior.coords.xy)       
                    polygon = np.vstack((polygon_xy[0], polygon_xy[1])).T.reshape(-1, 2)
                    z = np.zeros(polygon.shape[0])
                    dt1 = np.c_[polygon, z]
                    dt1 = np.array(dt1, dtype=np.float32)
                    filename_border = str(i).rjust(4, '0')
                    filename_border = filename_border + f"_{j}.csv"
                    fname_brd = os.path.join(path_folder, filename_border) 
                    np.savetxt(fname_brd, dt1, delimiter=',', header='x,y,z')
                    j += 1

    def select_clusters(self, shp_poly):
        """
        Process Voronoi cells and return dictionary of cell PCDs instead of saving to disk.
        
        Args:
            shp_poly: Shapefile polygon to use for cutting
            
        Returns:
            Dictionary mapping cell_id -> CELL object
        """
        print('Processing voronoi tessellation cells in memory...')
        
        # First get the regions and vertices
        vor = Voronoi(self.pts)
        regions, vertices = PCD_UTILS.voronoi_finite_polygons_2d(vor)
        
        # Create border polygons first
        shp_ply = Polygon(shp_poly)
        borders = []
        
        for i in range(self.n_clusters):
            polys = []
            for region_idx, region in enumerate(regions):
                if self.algo.labels_[region_idx] == i:
                    polygon = vertices[region]
                    poly = Polygon(polygon)
                    polys.append(poly)
            
            if polys:
                multi_poly = unary_union(polys)
                cut_poly = multi_poly.intersection(shp_ply)
                
                if isinstance(cut_poly, Polygon):
                    polygon_xy = np.asarray(cut_poly.exterior.coords.xy)       
                    polygon = np.vstack((polygon_xy[0], polygon_xy[1])).T.reshape(-1, 2)
                    z = np.zeros(polygon.shape[0])
                    borders.append(np.c_[polygon, z])
                    
                elif hasattr(cut_poly, 'geoms'):  # MultiPolygon
                    for j, poly_item in enumerate(cut_poly.geoms):
                        polygon_xy = np.asarray(poly_item.exterior.coords.xy)       
                        polygon = np.vstack((polygon_xy[0], polygon_xy[1])).T.reshape(-1, 2)
                        z = np.zeros(polygon.shape[0])
                        borders.append(np.c_[polygon, z])
        
        # Now process each cell
        cells_dict = {}
        pc_area = PCD_AREA(points=self.points, intensity=self.intensity)
        
        for i, border in enumerate(borders):
            # Cut by border
            pc_part = pc_area.poly_cut(border, returned='area')
            
            if pc_part.points.shape[0] > 1:
                # Convert PCD_AREA to CELL for compatibility with extract_stumps_labels
                from lidarsegmentation.classes.CELL import CELL
                cell_obj = CELL(points=pc_part.points, intensity=pc_part.intensity)
                
                cell_id = f"{i:04d}.pcd"
                cells_dict[cell_id] = cell_obj
        
        print(f'Processed {len(cells_dict)} cells in memory')
        return cells_dict


