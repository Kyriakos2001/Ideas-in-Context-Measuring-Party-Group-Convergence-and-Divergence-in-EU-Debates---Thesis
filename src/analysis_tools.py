# Analysis tools for EU Parliament discourse analysis project.
# Key functions: calculate_absolute_polarization, calculate_polarization_metrics,
# perform_meta_clustering_analysis, analyze_cluster_diversity, calculate_advanced_polarization_metrics

import numpy as np
import pandas as pd
import pickle
import os
import copy
import nltk.stem
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist
from itertools import combinations
from tqdm import tqdm
from utils.plotting_utils import filter_temporal_data
from poc_phase1_data_prep import load_stem_lemma_map, get_morphological_variations
from clustering_themes import CLUSTERING_THEMES


def calculate_absolute_polarization(embeddings, output_dir="../results", temporal_unit='term',
                                   sentence_data=None, use_weighted_centroid=False):
    embeddings = filter_temporal_data(embeddings, temporal_unit)

    polarization_data = []
    terms_processed = []

    terms = sorted(set(term for term, group in embeddings.keys()))

    for term in terms:
        term_embeddings = {group: embeddings[(term, group)]
                          for t, group in embeddings.keys()
                          if t == term}

        if len(term_embeddings) < 2:
            continue

        terms_processed.append(term)

        all_embeddings = list(term_embeddings.values())

        if use_weighted_centroid and sentence_data is not None:
            sentence_counts = {group: len(sentence_data.get((term, group), []))
                             for group in term_embeddings.keys()}

            weights = [1.0 / count if count > 0 else 0
                      for group, count in sentence_counts.items()]

            if sum(weights) > 0:
                centroid = np.average(all_embeddings, axis=0, weights=weights)
                centroid_method = 'weighted_cosine_distance'
            else:
                centroid = np.mean(all_embeddings, axis=0)
                centroid_method = 'cosine_distance'
        else:
            centroid = np.mean(all_embeddings, axis=0)
            centroid_method = 'weighted_cosine_distance' if use_weighted_centroid else 'cosine_distance'
            if use_weighted_centroid:
                pass

        for group, embedding in term_embeddings.items():
            distance_from_center = cosine_distances([embedding], [centroid])[0, 0]

            polarization_data.append({
                'term': term,
                'group': group,
                'distance_from_center': distance_from_center,
                'centroid_method': centroid_method
            })

    df = pd.DataFrame(polarization_data)

    return df


def load_embeddings(run_folder="../results", filename=None, theme_name=None, target_keyword="security",
                   model_name="roberta", keyword_group=None, aggregation_mode="average"):
    if filename is None:
        if keyword_group:
            keyword_part = "_".join(keyword_group)
        else:
            keyword_part = target_keyword if isinstance(target_keyword, str) else "_".join(target_keyword)

        if aggregation_mode == "average":
            suffix = "avg_embeddings"
        elif aggregation_mode == "kmeans":
            suffix = "clustered_data"
        else:
            suffix = f"{aggregation_mode}_embeddings"

        if theme_name:
            filename = f'poc_{keyword_part}_{theme_name}_{model_name}_{suffix}.pkl'
        else:
            filename = f'poc_{keyword_part}_{model_name}_{suffix}.pkl'

    filepath = os.path.join(run_folder, "data", filename)

    with open(filepath, 'rb') as f:
        raw_data = pickle.load(f)

    if aggregation_mode == "kmeans":
        clustered_embeddings = raw_data

        embeddings = {}
        for group_key, cluster_data in raw_data.items():
            centroids = cluster_data['centroids']
            labels = cluster_data['labels']

            if len(centroids) > 0:
                if len(centroids) == 1:
                    embeddings[group_key] = centroids[0]
                else:
                    unique_labels, counts = np.unique(labels, return_counts=True)
                    weights = counts / len(labels)
                    weighted_centroid = np.average(centroids, axis=0, weights=weights)
                    embeddings[group_key] = weighted_centroid

        return embeddings, clustered_embeddings
    else:
        embeddings = raw_data
        return embeddings, None


def calculate_polarization_metrics(embeddings, run_folder="../results", theme_name=None, temporal_unit='term'):
    print("\nCalculating polarization metrics...")

    embeddings = filter_temporal_data(embeddings, temporal_unit)

    all_groups = sorted(set(group for term, group in embeddings.keys()))
    print(f"Found {len(all_groups)} political groups in embeddings: {', '.join(all_groups[:10])}{'...' if len(all_groups) > 10 else ''}")

    ideological_pairs = list(combinations(all_groups, 2))
    print(f"Generated {len(ideological_pairs)} ideological pairs for comparison")

    polarization_data = []
    missing_data_log = []

    terms = sorted(set(term for term, group in embeddings.keys()))

    total_operations = len(terms) * len(ideological_pairs)
    completed_operations = 0

    for term in tqdm(terms, desc="Processing parliamentary terms"):
        term_embeddings = {group: embeddings[(t, group)]
                          for t, group in embeddings.keys()
                          if t == term}

        for group1, group2 in ideological_pairs:
            completed_operations += 1
            progress_pct = completed_operations / total_operations * 100

            if group1 in term_embeddings and group2 in term_embeddings:
                emb1 = term_embeddings[group1].reshape(1, -1)
                emb2 = term_embeddings[group2].reshape(1, -1)

                distance = cosine_distances(emb1, emb2)[0, 0]

                polarization_data.append({
                    'term': term,
                    'group1': group1,
                    'group2': group2,
                    'pair': f"{group1} vs {group2}",
                    'distance': distance
                })

            else:
                missing_data_log.append(f"Term {term}: {group1} vs {group2} - Missing data")

    polarization_df = pd.DataFrame(polarization_data)
    print(f"\nGenerated {len(polarization_df)} polarization measurements")

    return polarization_df


def extract_keyword_embedding(text, keyword, tokenizer, model, device):
    import torch

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.squeeze(0)

    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze(0))

    keyword_token_indices = []
    keyword_lower = keyword.lower()

    for i, token in enumerate(tokens):
        clean_token = token.replace('Ġ', '').lower()

        if (clean_token == keyword_lower or
            keyword_lower.startswith(clean_token) or
            clean_token in keyword_lower):
            keyword_token_indices.append(i)

    if not keyword_token_indices:
        for i, token in enumerate(tokens):
            clean_token = token.replace('Ġ', '').lower()
            if len(clean_token) > 2 and clean_token in keyword_lower:
                keyword_token_indices.append(i)

    if not keyword_token_indices:
        non_special_indices = []
        for i, token in enumerate(tokens):
            if token not in ['<s>', '</s>', '[CLS]', '[SEP]', '<pad>']:
                non_special_indices.append(i)
        keyword_token_indices = non_special_indices

    if keyword_token_indices:
        keyword_embeddings = embeddings[keyword_token_indices]
        keyword_embedding = torch.mean(keyword_embeddings, dim=0)
    else:
        keyword_embedding = torch.mean(embeddings[1:-1], dim=0)

    return keyword_embedding.cpu().numpy()


def get_dynamic_theme_vectors(target_keyword, model_name, run_folder):

    cache_filename = f"clustering_vectors_{model_name}_sans_{target_keyword}.pkl"
    cache_path = os.path.join(run_folder, "data", cache_filename)

    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            vectors = pickle.load(f)
        return vectors

    stem_map = load_stem_lemma_map()
    stemmer = nltk.stem.PorterStemmer()
    target_variants = get_morphological_variations(target_keyword, stem_map, stemmer)

    filtered_themes = copy.deepcopy(CLUSTERING_THEMES)
    for theme_name, keywords in filtered_themes.items():
        original_count = len(keywords)
        filtered_keywords = [kw for kw in keywords if kw.lower() not in target_variants]
        filtered_themes[theme_name] = filtered_keywords
        removed_count = original_count - len(filtered_keywords)
        if removed_count > 0:
            pass

    from poc_phase2_embeddings import setup_model
    tokenizer, model, device, _, _, _ = setup_model(model_name)

    theme_vectors = {}

    for theme_name, keywords in filtered_themes.items():
        if not keywords:
            continue

        embeddings = []

        for keyword in keywords:
            try:
                context_template = f"A discussion about {keyword}."
                embedding = extract_keyword_embedding(context_template, keyword, tokenizer, model, device)
                embeddings.append(embedding)
            except Exception as e:
                continue

        if embeddings:
            theme_vectors[theme_name] = np.mean(embeddings, axis=0)
        else:
            pass

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(theme_vectors, f)

    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return theme_vectors


def perform_meta_clustering_analysis(clustered_embeddings, sentence_data_map, theme_concept_vectors,
                                    target_keyword=None, model_name='roberta', run_folder="../results",
                                    k_min=2, k_max=10, silhouette_threshold=0.25):

    all_centroids = []
    centroid_lookup = []

    for (term, group), cluster_data in clustered_embeddings.items():
        centroids = cluster_data['centroids']
        for cluster_idx, centroid in enumerate(centroids):
            all_centroids.append(centroid)
            centroid_lookup.append(((term, group), cluster_idx))

    if len(all_centroids) == 0:
        return {}, {}, {}


    centroids_array = np.array(all_centroids)

    n_samples = len(all_centroids)

    if n_samples < 4:
        return {}, {}, {}

    perplexity = min(30, max(5, n_samples // 3))

    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    embeddings_2d = tsne.fit_transform(centroids_array)

    max_possible_k = min(k_max, n_samples - 1)
    if max_possible_k < k_min:
        n_clusters = 2
    else:
        best_k = k_min
        best_score = -1
        best_labels = None

        for k in range(k_min, max_possible_k + 1):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(centroids_array)

                if k < n_samples:
                    score = silhouette_score(centroids_array, labels)

                    if score > best_score:
                        best_k = k
                        best_score = score
                        best_labels = labels
            except Exception as e:
                continue

        if best_score >= silhouette_threshold:
            n_clusters = best_k
        else:
            n_clusters = min(5, max(2, n_samples // 6))

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    macro_cluster_labels = kmeans.fit_predict(centroids_array)


    unique_labels = set(macro_cluster_labels)
    n_clusters_found = len(unique_labels)

    macro_cluster_map = {}
    for i, macro_cluster_id in enumerate(macro_cluster_labels):
        macro_cluster_map[i] = macro_cluster_id

    macro_cluster_sentences = {label: [] for label in unique_labels}

    for lookup_idx, ((term, group), local_cluster_id) in enumerate(centroid_lookup):
        macro_cluster_id = macro_cluster_map[lookup_idx]

        if (term, group) in sentence_data_map:
            original_sentences = sentence_data_map[(term, group)]
            cluster_data = clustered_embeddings[(term, group)]
            labels = cluster_data['labels']

            cluster_sentence_indices = [i for i, label in enumerate(labels) if label == local_cluster_id]
            cluster_sentences = [original_sentences[i] for i in cluster_sentence_indices
                               if i < len(original_sentences)]

            macro_cluster_sentences[macro_cluster_id].extend(cluster_sentences)

    for cluster_id, sentences in macro_cluster_sentences.items():
        pass

    themes_list = list(theme_concept_vectors.keys())
    concept_vectors_matrix = np.array([theme_concept_vectors[theme] for theme in themes_list])
    unique_cluster_ids = sorted(set(macro_cluster_map.values()))

    all_centroids_array = np.array(all_centroids)
    sim_matrix = cosine_similarity(all_centroids_array, concept_vectors_matrix)

    z_matrix = np.zeros_like(sim_matrix)

    for i in range(sim_matrix.shape[0]):
        row = sim_matrix[i]
        row_mean = np.mean(row)
        row_std = np.std(row)
        z_matrix[i] = (row - row_mean) / (row_std + 1e-9)

    winning_theme_indices = np.argmax(z_matrix, axis=1)

    cluster_votes = {cluster_id: {} for cluster_id in unique_cluster_ids}

    for i in range(len(all_centroids)):
        macro_id = macro_cluster_map[i]
        theme_idx = winning_theme_indices[i]
        theme_name = themes_list[theme_idx]

        if theme_name not in cluster_votes[macro_id]:
            cluster_votes[macro_id][theme_name] = 0
        cluster_votes[macro_id][theme_name] += 1

    macro_theme_labels = {}

    print("\n" + "="*80)
    print("BOTTOM-UP Z-SCORE SEMANTIC LABELING RESULTS")
    print("="*80)
    print(f"Micro-centroids analyzed: {len(all_centroids)}")
    print(f"Macro-clusters: {len(unique_cluster_ids)}")
    print(f"Themes available: {len(themes_list)}")
    print()

    for macro_id in unique_cluster_ids:
        votes = cluster_votes[macro_id]
        if votes:

            winning_theme = max(votes, key=votes.get)
            max_votes = votes[winning_theme]
            total_votes = sum(votes.values())
            purity_percent = (max_votes / total_votes) * 100


            macro_theme_labels[macro_id] = f"{winning_theme.title()} ({purity_percent:.0f}%)"

            print(f"Cluster {macro_id}: {winning_theme.title()} "
                  f"({max_votes}/{total_votes} votes, {purity_percent:.1f}% purity)")


            sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)
            for theme, count in sorted_votes:
                percentage = (count / total_votes) * 100
                if theme == winning_theme:
                    print(f"    Winner: {theme} - {count} votes ({percentage:.1f}%)")
                else:
                    print(f"           {theme} - {count} votes ({percentage:.1f}%)")
        else:
            macro_theme_labels[macro_id] = f"Cluster_{macro_id}"
            print(f"Cluster {macro_id}: No votes - using generic label")

    print("\n" + "="*80)

    report_filename = f"bottom_up_zscore_analysis_{target_keyword}_{model_name}.txt"
    report_path = os.path.join(run_folder, report_filename)

    with open(report_path, 'w') as f:
        f.write(f"BOTTOM-UP Z-SCORE SEMANTIC LABELING ANALYSIS\n")
        f.write(f"Keyword: {target_keyword}\n")
        f.write(f"Model: {model_name}\n")
        f.write("="*70 + "\n\n")

        f.write(f"Analysis Overview:\n")
        f.write(f"- Micro-centroids processed: {len(all_centroids)}\n")
        f.write(f"- Macro-clusters: {len(unique_cluster_ids)}\n")
        f.write(f"- Themes analyzed: {len(themes_list)}\n")
        f.write(f"- Target keyword excluded: {target_keyword}\n\n")

        f.write("FINAL CLUSTER ASSIGNMENTS:\n")
        f.write("="*50 + "\n")

        for macro_id in unique_cluster_ids:
            votes = cluster_votes[macro_id]
            f.write(f"\nCluster {macro_id}: {macro_theme_labels.get(macro_id, 'N/A')}\n")
            f.write("-" * 40 + "\n")

            if votes:
                total_votes = sum(votes.values())
                sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)

                for rank, (theme, count) in enumerate(sorted_votes, 1):
                    percentage = (count / total_votes) * 100
                    marker = "*** WINNER ***" if rank == 1 else ""
                    f.write(f"  {rank:2d}. {theme:<20} {count:3d} votes ({percentage:5.1f}%) {marker}\n")
            else:
                f.write("  No votes recorded\n")


    tsne_data = {
        'embeddings_2d': embeddings_2d,
        'centroid_lookup': centroid_lookup,
        'all_centroids': all_centroids
    }

    return macro_cluster_map, macro_theme_labels, tsne_data


def analyze_cluster_diversity(clustered_embeddings, sentence_data_map=None, run_folder="../results",
                             theme_name=None, target_keyword="security"):

    group_diversity = {}
    temporal_evolution = {}

    for (term, group), cluster_data in clustered_embeddings.items():
        centroids = cluster_data['centroids']
        labels = cluster_data['labels']
        embeddings = cluster_data['embeddings']

        if group not in group_diversity:
            group_diversity[group] = []
        if term not in temporal_evolution:
            temporal_evolution[term] = []

        n_clusters = len(centroids)
        n_embeddings = len(embeddings)

        if n_clusters > 1 and n_embeddings > 1:
            silhouette = silhouette_score(embeddings, labels)

            if n_clusters > 1:
                centroid_distances = pdist(centroids)
                max_centroid_distance = np.max(centroid_distances)
                avg_centroid_distance = np.mean(centroid_distances)
            else:
                max_centroid_distance = 0
                avg_centroid_distance = 0

        else:
            silhouette = 0
            max_centroid_distance = 0
            avg_centroid_distance = 0

        diversity_metrics = {
            'term': term,
            'n_clusters': n_clusters,
            'n_embeddings': n_embeddings,
            'silhouette_score': silhouette,
            'max_centroid_distance': max_centroid_distance,
            'avg_centroid_distance': avg_centroid_distance,
            'diversity_ratio': max_centroid_distance / (avg_centroid_distance + 1e-6)
        }

        group_diversity[group].append(diversity_metrics)
        temporal_evolution[term].append((group, diversity_metrics))

    report_filename = f"cluster_diversity_analysis{('_' + theme_name) if theme_name else ''}.txt"
    report_path = os.path.join(run_folder, report_filename)

    with open(report_path, 'w') as f:
        f.write(f"CLUSTER DIVERSITY ANALYSIS REPORT\n")
        f.write(f"Keyword: {target_keyword.title()}\n")
        if theme_name:
            f.write(f"Context: {theme_name.title()}\n")
        f.write(f"=" * 50 + "\n\n")

        f.write("DISCOURSE FRAGMENTATION SUMMARY:\n")
        f.write("-" * 35 + "\n")

        all_diversities = []
        all_silhouettes = []

        for group, metrics_list in group_diversity.items():
            avg_diversity = np.mean([m['avg_centroid_distance'] for m in metrics_list])
            avg_silhouette = np.mean([m['silhouette_score'] for m in metrics_list if m['silhouette_score'] > 0])

            all_diversities.append((group, avg_diversity))
            if avg_silhouette > 0:
                all_silhouettes.append((group, avg_silhouette))

            f.write(f"\n{group}:\n")
            f.write(f"  Average Diversity Score: {avg_diversity:.4f}\n")
            f.write(f"  Average Clustering Quality: {avg_silhouette:.4f}\n")
            f.write(f"  Terms Analyzed: {len(metrics_list)}\n")

        if all_diversities:
            most_diverse = max(all_diversities, key=lambda x: x[1])
            least_diverse = min(all_diversities, key=lambda x: x[1])

            f.write(f"\nKEY INSIGHTS:\n")
            f.write(f"• Most Fragmented: {most_diverse[0]} (diversity: {most_diverse[1]:.4f})\n")
            f.write(f"• Most Coherent: {least_diverse[0]} (diversity: {least_diverse[1]:.4f})\n")

        if all_silhouettes:
            best_clustering = max(all_silhouettes, key=lambda x: x[1])
            f.write(f"• Best Clustering Quality: {best_clustering[0]} (silhouette: {best_clustering[1]:.4f})\n")


    return {
        'group_diversity': group_diversity,
        'temporal_evolution': temporal_evolution,
        'report_path': report_path
    }


def generate_summary_report(polarization_df, embeddings, run_folder="../results", theme_name=None,
                           target_keyword='security', advanced_polarization_df=None):
    print("\n" + "=" * 60)
    print("PROOF OF CONCEPT SUMMARY REPORT")
    if theme_name:
        print(f"{target_keyword.title()} Keyword Analysis - {theme_name.title()} Context")
    else:
        print(f"{target_keyword.title()} Keyword Analysis - Baseline")
    print("=" * 60)

    print(f"Analysis Coverage:")
    print(f"  Total embedding vectors: {len(embeddings)}")
    print(f"  Parliamentary terms covered: {len(set(term for term, group in embeddings.keys()))}")
    print(f"  Political groups covered: {len(set(group for term, group in embeddings.keys()))}")
    print(f"  Polarization measurements: {len(polarization_df)}")

    print(f"\nPolarization Analysis Findings:")

    avg_distances = polarization_df.groupby('pair')['distance'].mean().sort_values(ascending=False)
    print(f"  Most polarized pair: {avg_distances.index[0]} (avg distance: {avg_distances.iloc[0]:.4f})")
    print(f"  Least polarized pair: {avg_distances.index[-1]} (avg distance: {avg_distances.iloc[-1]:.4f})")

    if len(set(polarization_df['term'])) > 1:
        temporal_trend = polarization_df.groupby('term')['distance'].mean()
        print(f"  Average distance over time:")
        for term, avg_dist in temporal_trend.items():
            print(f"    Term {term}: {avg_dist:.4f}")

    if advanced_polarization_df is not None and not advanced_polarization_df.empty:
        print(f"\nAdvanced Polarization Analysis (K-Means Mode):")
        print(f"  Advanced measurements: {len(advanced_polarization_df)}")

        for metric in ['refined_polarization', 'discourse_extremity', 'semantic_overlap', 'average_set_polarization']:
            overall_avg = advanced_polarization_df[metric].mean()
            metric_name = metric.replace('_', ' ').title()
            print(f"  Overall {metric_name}: {overall_avg:.4f}")

        print(f"\n  Most Polarized (Discourse Extremity):")
        extremity_avg = advanced_polarization_df.groupby('pair')['discourse_extremity'].mean().sort_values(ascending=False)
        for pair, value in extremity_avg.head(3).items():
            print(f"    {pair}: {value:.4f}")

        print(f"\n  Closest Overlap (Semantic Bridge):")
        overlap_avg = advanced_polarization_df.groupby('pair')['semantic_overlap'].mean().sort_values(ascending=True)
        for pair, value in overlap_avg.head(3).items():
            print(f"    {pair}: {value:.4f}")

    report_filename = f"poc_summary_report{('_' + theme_name) if theme_name else ''}.txt"
    report_path = os.path.join(run_folder, report_filename)
    with open(report_path, 'w') as f:
        if theme_name:
            f.write(f"{target_keyword.title()} Keyword PoC - Summary Report ({theme_name} context)\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Thematic Context: {theme_name}\n\n")
        else:
            f.write(f"{target_keyword.title()} Keyword PoC - Summary Report (baseline)\n")
            f.write("=" * 60 + "\n\n")
        f.write(f"Total embeddings analyzed: {len(embeddings)}\n")
        f.write(f"Polarization measurements: {len(polarization_df)}\n\n")
        f.write("Average distances by political pair:\n")
        for pair, distance in avg_distances.items():
            f.write(f"  {pair}: {distance:.4f}\n")

        if advanced_polarization_df is not None and not advanced_polarization_df.empty:
            f.write(f"\n\nADVANCED POLARIZATION ANALYSIS (K-MEANS MODE):\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Advanced measurements: {len(advanced_polarization_df)}\n\n")

            f.write("Overall Averages Across All Terms:\n")
            for metric in ['refined_polarization', 'discourse_extremity', 'semantic_overlap', 'average_set_polarization']:
                overall_avg = advanced_polarization_df[metric].mean()
                metric_name = metric.replace('_', ' ').title()
                f.write(f"  {metric_name}: {overall_avg:.4f}\n")

            f.write(f"\nMost Polarized Pairs (Discourse Extremity):\n")
            extremity_avg = advanced_polarization_df.groupby('pair')['discourse_extremity'].mean().sort_values(ascending=False)
            for pair, value in extremity_avg.items():
                f.write(f"  {pair}: {value:.4f}\n")

            f.write(f"\nClosest Semantic Overlap (Bridge Points):\n")
            overlap_avg = advanced_polarization_df.groupby('pair')['semantic_overlap'].mean().sort_values(ascending=True)
            for pair, value in overlap_avg.items():
                f.write(f"  {pair}: {value:.4f}\n")

            f.write(f"\nPolarization Ranges (Extremity - Overlap):\n")
            range_data = advanced_polarization_df.groupby('pair').agg({
                'discourse_extremity': 'mean',
                'semantic_overlap': 'mean'
            })
            range_data['polarization_range'] = range_data['discourse_extremity'] - range_data['semantic_overlap']
            range_sorted = range_data.sort_values('polarization_range', ascending=False)
            for pair, row in range_sorted.iterrows():
                f.write(f"  {pair}: {row['polarization_range']:.4f} (max: {row['discourse_extremity']:.4f}, min: {row['semantic_overlap']:.4f})\n")


    return report_path


def calculate_advanced_polarization_metrics(clustered_embeddings, temporal_unit='term'):

    clustered_embeddings = filter_temporal_data(clustered_embeddings, temporal_unit)

    all_groups = sorted(set(group for term, group in clustered_embeddings.keys()))
    all_terms = sorted(set(term for term, group in clustered_embeddings.keys()))

    ideological_pairs = list(combinations(all_groups, 2))

    advanced_polarization_data = []

    for term in tqdm(all_terms, desc="Processing terms for advanced metrics"):
        term_clustered_embeddings = {group: clustered_embeddings[(term, group)]
                                    for t, group in clustered_embeddings.keys()
                                    if t == term}

        for group1, group2 in ideological_pairs:
            if group1 in term_clustered_embeddings and group2 in term_clustered_embeddings:
                cluster_data1 = term_clustered_embeddings[group1]
                cluster_data2 = term_clustered_embeddings[group2]

                centroids1 = cluster_data1['centroids']
                centroids2 = cluster_data2['centroids']
                labels1 = cluster_data1['labels']
                labels2 = cluster_data2['labels']

                if len(centroids1) == 1:
                    weighted_center1 = centroids1[0]
                else:
                    unique_labels1, counts1 = np.unique(labels1, return_counts=True)
                    weights1 = counts1 / len(labels1)
                    weighted_center1 = np.average(centroids1, axis=0, weights=weights1)

                if len(centroids2) == 1:
                    weighted_center2 = centroids2[0]
                else:
                    unique_labels2, counts2 = np.unique(labels2, return_counts=True)
                    weights2 = counts2 / len(labels2)
                    weighted_center2 = np.average(centroids2, axis=0, weights=weights2)

                refined_polarization = cosine_distances([weighted_center1], [weighted_center2])[0, 0]

                all_distances = []
                for c1 in centroids1:
                    for c2 in centroids2:
                        distance = cosine_distances([c1], [c2])[0, 0]
                        all_distances.append(distance)

                discourse_extremity = max(all_distances) if all_distances else 0

                semantic_overlap = min(all_distances) if all_distances else 0

                average_set_polarization = np.mean(all_distances) if all_distances else 0

                advanced_polarization_data.append({
                    'term': term,
                    'group1': group1,
                    'group2': group2,
                    'pair': f"{group1} vs {group2}",
                    'refined_polarization': refined_polarization,
                    'discourse_extremity': discourse_extremity,
                    'semantic_overlap': semantic_overlap,
                    'average_set_polarization': average_set_polarization,
                    'num_comparisons': len(all_distances)
                })

    advanced_df = pd.DataFrame(advanced_polarization_data)

    return advanced_df


def calculate_absolute_advanced_metrics(clustered_embeddings, temporal_unit='term'):

    clustered_embeddings = filter_temporal_data(clustered_embeddings, temporal_unit)

    all_terms = sorted(set(term for term, group in clustered_embeddings.keys()))

    absolute_advanced_data = []

    for term in tqdm(all_terms, desc="Processing terms for absolute advanced metrics"):
        term_clustered_embeddings = {group: clustered_embeddings[(term, group)]
                                    for t, group in clustered_embeddings.keys()
                                    if t == term}

        if len(term_clustered_embeddings) < 2:
            continue

        all_centroids = []
        all_weights = []

        for group, cluster_data in term_clustered_embeddings.items():
            centroids = cluster_data['centroids']
            labels = cluster_data['labels']

            unique_labels, counts = np.unique(labels, return_counts=True)
            total_embeddings = len(labels)

            for cluster_idx, centroid in enumerate(centroids):
                all_centroids.append(centroid)
                cluster_weight = counts[cluster_idx] / total_embeddings if cluster_idx < len(counts) else 1.0
                all_weights.append(cluster_weight)

        if not all_centroids:
            continue

        all_centroids_array = np.array(all_centroids)
        all_weights_array = np.array(all_weights)
        overall_center = np.average(all_centroids_array, axis=0, weights=all_weights_array)

        for group, cluster_data in term_clustered_embeddings.items():
            centroids = cluster_data['centroids']
            labels = cluster_data['labels']

            distances_to_center = []
            for centroid in centroids:
                distance = cosine_distances([centroid], [overall_center])[0, 0]
                distances_to_center.append(distance)

            if not distances_to_center:
                continue

            absolute_extremity = max(distances_to_center)
            absolute_overlap = min(distances_to_center)
            absolute_range = absolute_extremity - absolute_overlap

            absolute_advanced_data.append({
                'term': term,
                'party': group,
                'absolute_extremity': absolute_extremity,
                'absolute_overlap': absolute_overlap,
                'absolute_range': absolute_range,
                'num_clusters': len(centroids)
            })

    absolute_df = pd.DataFrame(absolute_advanced_data)

    return absolute_df