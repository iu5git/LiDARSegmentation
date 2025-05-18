import torch
import numpy as np
from lidarsegmentation.predictmdl.models.pointnet2_cls_ssg import get_model
import lidarsegmentation.predictmdl.utils.pointcloud_utils as pcu
from pyntcloud import PyntCloud
import os
import pandas as pd
from tqdm import tqdm
from lidarsegmentation.settings.seg_settings import SS

def farthest_point_sample(xyz, npoint):
    device = xyz.device
    batchsize, ndataset, dimension = xyz.shape
    centroids = torch.zeros(batchsize, npoint, dtype=torch.long).to(device)
    distance = torch.ones(batchsize, ndataset).to(device) * 1e10
    farthest =  torch.randint(0, ndataset, (batchsize,), dtype=torch.long).to(device)
    batch_indices = torch.arange(batchsize, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:,i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(batchsize, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def predict_from_pcd(pcd_obj, model_name):
    """
    Predict tree/not_tree label directly from a PCD object in memory.
    
    Args:
        pcd_obj: PCD object with points and intensity
        model_name: Name of the model to use for prediction
        
    Returns:
        Label (1: tree, 0: not_tree, -1: error)
    """
    model_path = os.path.join('lidarsegmentation', 'predictmdl', 'checkpoints', model_name, 'models', 'model.t7')
    
    species_names = ['Tree', 'Not_Tree']
    
    try:
        # Extract points from PCD object
        points = pcd_obj.points
        points = np.array([points])
        
        # Force CPU usage to avoid CUDA errors
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device('cpu')
        points = torch.Tensor(points).to(device)
        centroids = farthest_point_sample(points, 2048)
        pc_sampled = points[0][centroids[0]]
        pc_sampled = pc_sampled.cpu().detach().numpy()
        
        X_test = np.array([pc_sampled])
        y_test = [0]
        
        X_test = pcu.tree_normalize(X_test)
        int2name = {i: name for i, name in enumerate(species_names)}
        
        NUM_CLASSES = len(int2name)
        
        model = get_model(NUM_CLASSES, normal_channel=False).to(device)
        # Load model with map_location to ensure it loads on CPU regardless of where it was saved
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.eval()
        test_true = []
        test_pred = []
        data, label = torch.tensor(X_test, device=device), torch.tensor(y_test, device=device)
        data = data.permute(0, 2, 1)
        logits, trans_feat = model(data)
        preds = logits.max(dim=1)[1].detach()
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.cpu().numpy())
        
        if test_pred[0][0] == 1:
            ans = 0  # "Not a tree"
        else:
            ans = 1  # "Tree"
    except Exception as e:
        print(f"Error in predict_from_pcd: {e}")
        ans = -1
        
    return ans

def predict_mem(clear_trees, model_name):
    """
    Predict tree/not_tree labels for PCD objects in memory
    
    Args:
        clear_trees: Dictionary of PCD objects keyed by filename
        model_name: Name of the model to use for prediction
        
    Returns:
        DataFrame with tree names and predicted labels
    """
    names = []
    labels = []
    
    for filename, pcd_obj in tqdm(clear_trees.items(), desc=f"Predicting with {model_name}"):
        label = predict_from_pcd(pcd_obj, model_name)
        names.append(filename)
        labels.append(label)
    
    pred_df = pd.DataFrame({"Name_tree": names, "Label": labels})
    return pred_df

def predict(path_file, model_name):
    """
    Original file-based prediction function
    """
    names = []
    labels = []
    for filename in tqdm(os.listdir(path_file)):
        if filename.endswith('.pcd'):
            src = os.path.join(path_file,filename)
            label = test(src, model_name)
            names.append(filename)
            labels.append(label)     
    bd = pd.DataFrame({"Name_tree": names,"Label": labels})
    bd.to_csv(os.path.join(path_file,'predict_' + model_name + '.csv'), index = False, sep=';')

def test(src, model_name):
    """Original test function that loads a PCD file from disk"""
    model_path = 'predictmdl/checkpoints/'+ model_name +'/models/model.t7'

    species_names = ['Tree','Not_Tree']
    # species_names = ['E','C','B','R','D','OC']
    # species_names = ['E','C','B']
    try:
        pc = PyntCloud.from_file(src)
        points = pc.points.loc[:,["x","y","z"]].values
        points = np.array([points])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        points = torch.Tensor(points).to(device)
        centroids = farthest_point_sample(points, 2048)
        pc_sampled = points[0][centroids[0]]
        pc_sampled = pc_sampled.cpu().detach().numpy()

        X_test = np.array([pc_sampled])
        y_test = [0]

        X_test = pcu.tree_normalize(X_test)
        int2name = { i:name for i, name in enumerate(species_names)}

        NUM_CLASSES = len(int2name)

        model = get_model(NUM_CLASSES,normal_channel=False).to(device)
        model.load_state_dict(torch.load(model_path))
        model = model.eval()
        test_true = []
        test_pred = []
        data, label = torch.tensor(X_test, device=device), torch.tensor(y_test, device=device)
        data = data.permute(0, 2, 1)
        logits, trans_feat = model(data)
        preds = logits.max(dim=1)[1].detach()
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.cpu().numpy())
        if test_pred[0][0] == 1:
            ans = 0 #"Это не дерево"
        else:
            ans = 1 #"Это дерево"
    except:
        ans = -1
    return ans

if __name__ == '__main__':
    from lidarsegmentation.segmentation_vor import segmentation_vor
    from lidarsegmentation.segmentation_ram import segmentation_ram
    from lidarsegmentation.segmentation_clear import segmentation_clear
    
    yml_path = "settings\settings.yaml"
    ss = SS.from_yaml(yml_path)
    model_name = 'cpl1-1024-rp-s1024-pn2'
    
    # In-memory pipeline
    binding_df, vor_trees = segmentation_vor(ss, make_binding=True)
    combined_df, ram_trees = segmentation_ram(ss, binding_df, vor_trees)
    clear_trees = segmentation_clear(ss, combined_df, ram_trees)
    pred_df = predict_mem(clear_trees, model_name)
    
    # Save final output
    output_path = os.path.join(ss.path_base, 'predict_' + model_name + '.csv')
    pred_df.to_csv(output_path, index=False, sep=';')
    print(f"Saved predictions to {output_path}")     
    