
# Phase 3 aggregation and clustering for EU Parliament discourse analysis.
# Main functions: aggregate_average, aggregate_kmeans, find_optimal_k, process_group.
# Result: Aggregates raw embeddings using either averaging or k-means clustering methods,
# producing final processed embeddings ready for polarization analysis and visualization.


import pickle
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
import argparse

warnings.filterwarnings('ignore')

def load_raw_embeddings(run_folder, target_keyword="security", theme_name=None, model_name="roberta", keyword_group=None):
    if keyword_group:
        keyword_part = "_".join(keyword_group)
    else:
        keyword_part = target_keyword if isinstance(target_keyword, str) else "_".join(target_keyword)
    
    if theme_name:
        filename = f'poc_{keyword_part}_{theme_name}_{model_name}_raw_embeddings.pkl'
    else:
        filename = f'poc_{keyword_part}_{model_name}_raw_embeddings.pkl'
    
    filepath = os.path.join(run_folder, "data", filename)
    
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw embeddings file not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        raw_embeddings = pickle.load(f)

    return raw_embeddings

def aggregate_average(raw_embeddings):
    averaged_embeddings = {}
    
    for group_key, embeddings_list in raw_embeddings.items():
        if embeddings_list:
            embeddings_array = np.array(embeddings_list)
            avg_embedding = np.mean(embeddings_array, axis=0)
            averaged_embeddings[group_key] = avg_embedding
    
    return averaged_embeddings

def find_optimal_k(embeddings_array, k_min=2, k_max=4, silhouette_threshold=0.25):

    n_samples = len(embeddings_array)

    if n_samples < 2:
        return 1, 0.0, "Too few samples for clustering", {}

    k_max_effective = min(k_max, n_samples - 1)
    k_min_effective = min(k_min, n_samples - 1)

    if k_min_effective < 2 or k_max_effective < 2:
        return 1, 0.0, f"Sample size ({n_samples}) too small for meaningful k range", {}

    scores = {}

    for k in range(max(2, k_min_effective), k_max_effective + 1):
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings_array)

            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                scores[k] = 0.0
            else:
                scores[k] = silhouette_score(embeddings_array, labels)
        except Exception as e:
            scores[k] = 0.0

    if not scores:
        return 1, 0.0, "No valid k values could be tested", {}

    best_k = max(scores, key=scores.get)
    best_score = scores[best_k]

    if best_score < silhouette_threshold:
        return 1, best_score, f"Best score {best_score:.3f} below threshold {silhouette_threshold:.3f}, forcing k=1", scores

    return best_k, best_score, f"Optimal k={best_k} with score {best_score:.3f}", scores

from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def process_group(args):
    group_key, embeddings_list, k_min, k_max, silhouette_threshold = args

    if not embeddings_list:
        return group_key, None

    embeddings_array = np.array(embeddings_list)

    optimal_k, best_score, reason, k_scores = find_optimal_k(
        embeddings_array, k_min, k_max, silhouette_threshold
    )

    if optimal_k == 1:
        avg_centroid = np.mean(embeddings_array, axis=0)
        result = {
            'centroids': np.array([avg_centroid]),
            'labels': np.zeros(len(embeddings_list)),
            'embeddings': embeddings_array,
            'n_clusters_actual': 1,
            'silhouette_score': best_score,
            'k_selection_reason': reason,
            'k_scores': k_scores
        }
    else:
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings_array)
        result = {
            'centroids': kmeans.cluster_centers_,
            'labels': labels,
            'embeddings': embeddings_array,
            'n_clusters_actual': optimal_k,
            'silhouette_score': best_score,
            'k_selection_reason': reason,
            'k_scores': k_scores
        }

    return group_key, result


def aggregate_kmeans(raw_embeddings, k_min=2, k_max=8, silhouette_threshold=0.25, n_jobs=None):

    if n_jobs is None:
        n_jobs = cpu_count()

    tasks = [
        (group_key, embeddings_list, k_min, k_max, silhouette_threshold)
        for group_key, embeddings_list in raw_embeddings.items()
    ]

    clustered_data = {}
    with Pool(processes=n_jobs) as pool:
        for group_key, result in tqdm(pool.imap_unordered(process_group, tasks),
                                      total=len(tasks),
                                      desc="Clustering groups"):
            if result is not None:
                clustered_data[group_key] = result

    total_groups = len(clustered_data)
    k_distribution = {}
    for data in clustered_data.values():
        k = data['n_clusters_actual']
        k_distribution[k] = k_distribution.get(k, 0) + 1

    k_dist_str = ", ".join([f"k={k}: {count} groups" for k, count in sorted(k_distribution.items())])

    return clustered_data


def save_aggregated_embeddings(aggregated_data, run_folder, mode, target_keyword="security", theme_name=None, model_name="roberta", keyword_group=None):

    if keyword_group:
        keyword_part = "_".join(keyword_group)
    else:
        keyword_part = target_keyword if isinstance(target_keyword, str) else "_".join(target_keyword)
    
    if mode == "average":
        suffix = "avg_embeddings"
    elif mode == "kmeans":
        suffix = "clustered_data"
    else:
        suffix = f"{mode}_embeddings"
    
    if theme_name:
        filename = f'poc_{keyword_part}_{theme_name}_{model_name}_{suffix}.pkl'
    else:
        filename = f'poc_{keyword_part}_{model_name}_{suffix}.pkl'
    
    filepath = os.path.join(run_folder, "data", filename)
    
    
    with open(filepath, 'wb') as f:
        pickle.dump(aggregated_data, f)
    
    if mode == "average":
        pass
    elif mode == "kmeans":
        total_clusters = sum(data['n_clusters_actual'] for data in aggregated_data.values())
    
    return filepath

def main(run_folder=None, mode="average", target_keyword='security', theme_name=None, model_name='roberta', keyword_group=None, n_clusters=3, k_min=2, k_max=8, silhouette_threshold=0.25):

    if run_folder is None:
        run_folder = "../results"


    if theme_name:
        pass
    else:
        pass

    try:
        raw_embeddings = load_raw_embeddings(run_folder, target_keyword, theme_name, model_name, keyword_group)

        if mode == "average":
            aggregated_data = aggregate_average(raw_embeddings)
        elif mode == "kmeans":
            aggregated_data = aggregate_kmeans(raw_embeddings, k_min, k_max, silhouette_threshold)
        else:
            raise ValueError(f"Unknown aggregation mode: {mode}. Supported modes: 'average', 'kmeans'")

        output_file = save_aggregated_embeddings(aggregated_data, run_folder, mode, target_keyword, theme_name, model_name, keyword_group)
        
        
        return output_file
        
    except Exception as e:
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Phase 3: Aggregation & Clustering for Keyword Analysis')
    parser.add_argument('run_folder', help='Run folder path')
    parser.add_argument('--mode', '-m', choices=['average', 'kmeans'], default='average',
                       help='Aggregation mode: average (simple averaging) or kmeans (clustering)')
    parser.add_argument('--target-keyword', '-k', default='security',
                       help='Target keyword to analyze')
    parser.add_argument('--theme', '-t', help='Theme name for thematic analysis')
    parser.add_argument('--model', choices=['roberta', 'distilroberta'], default='roberta',
                       help='Embedding model name')
    parser.add_argument('--keyword-group', nargs='+', help='List of keywords for grouped analysis')
    parser.add_argument('--clusters', '-c', type=int, default=3,
                       help='Number of clusters for K-means mode (deprecated, kept for backward compatibility)')
    parser.add_argument('--k-min', type=int, default=2,
                       help='Minimum number of clusters to test for Dynamic K-means (default: 2)')
    parser.add_argument('--k-max', type=int, default=8,
                       help='Maximum number of clusters to test for Dynamic K-means (default: 8)')
    parser.add_argument('--silhouette-threshold', type=float, default=0.25,
                       help='Minimum silhouette score threshold for meaningful clusters (default: 0.25)')

    args = parser.parse_args()

    main(
        run_folder=args.run_folder,
        mode=args.mode,
        target_keyword=args.target_keyword,
        theme_name=args.theme,
        model_name=args.model,
        keyword_group=args.keyword_group,
        n_clusters=args.clusters,
        k_min=args.k_min,
        k_max=args.k_max,
        silhouette_threshold=args.silhouette_threshold
    )