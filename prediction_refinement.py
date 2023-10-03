import torch
import numpy as np
from sklearn.cluster import DBSCAN, HDBSCAN, OPTICS, SpectralClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances_argmin_min

def prediction_refinement(features, predicts, clustering_method, eps=0.5, min_samples=5, min_cluster_size=5, noise_handling='merge_most'):
    
    '''
    Arguments:
        features_list: feature vectors to be used for clustering
        clustering_method: 'DBSCAN' or 'HDBSCAN'
        eps: eps for DBSCAN
        min_samples: min_samples for DBSCAN
        noise_handling: 'merge_most' or 'merge_all' or 'retain_all'
    '''

    features = features.cpu().numpy()
    predicts = np.array(predicts)

    # clustering
    clustering_method = clustering_method.upper()
    if clustering_method == 'DBSCAN':
        pred_cluster = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(features)
    elif clustering_method == 'HDBSCAN':
        pred_cluster = HDBSCAN(allow_single_cluster=True, min_samples=min_samples, min_cluster_size=min_cluster_size).fit_predict(features)
    elif clustering_method == 'OPTICS':
        pred_cluster = OPTICS(max_eps=eps, cluster_method="xi").fit_predict(features)
    elif clustering_method == 'SPECTRAL_CLUSTERING':
        pred_cluster = SpectralClustering(n_clusters=3, assign_labels="cluster_qr", random_state=0).fit_predict(features)
    elif clustering_method == 'KMEANS':
        pred_cluster = KMeans(n_clusters=3, random_state=0).fit_predict(features)
    elif clustering_method == 'GAUSSIAN_MIXTURE':
        pred_cluster = GaussianMixture(n_components=3, random_state=0).fit_predict(features)
    else:
        raise ValueError('clustering_method not recognized')
    
    # # keep only largest cluster, set others to -1
    # largest_cluster = np.argmax(np.bincount(pred_cluster[pred_cluster != -1]))
    # pred_cluster[pred_cluster != largest_cluster] = -1
    # pred_cluster[pred_cluster == largest_cluster] = 0

    # noise handling
    if noise_handling == 'retain_all':
        pass
    else:
        if -1 in pred_cluster and len(np.unique(pred_cluster)) > 1:
            argmin, min = pairwise_distances_argmin_min(features[pred_cluster == -1], features[pred_cluster != -1])
            replace = pred_cluster[pred_cluster != -1][argmin]
            if noise_handling == 'merge_most':
                # 10th largest value of min
                min_thres = np.sort(min)[np.max([-10, -len(min)])]
                replace[min > min_thres] = -1
                pred_cluster[pred_cluster == -1] = replace
            elif noise_handling == 'merge_all':
                pred_cluster[pred_cluster == -1] = replace

    # samples in same cluster should share same prediction, decide by majority voting
    for i in range(np.max(pred_cluster.astype(int))+1):
        cluster_predictions = predicts[pred_cluster == i]

        pred_dict = {}
        for pred in cluster_predictions:
            if pred in pred_dict:
                pred_dict[pred] += 1
            else:
                pred_dict[pred] = 1
        
        # majority voting
        cluster_prediction = max(pred_dict, key=pred_dict.get)
        predicts[pred_cluster == i] = cluster_prediction
    
    return predicts, pred_cluster