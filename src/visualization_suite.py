
# Visualization suite for EU Parliament discourse analysis.
# Creates temporal line plots (polarization over parliamentary terms), polarization heatmaps (group distances),
# t-SNE scatter plots (embedding space visualization), radar/spider charts (semantic theme distribution),
# word clouds (discourse content), constellation plots (cluster relationships), and comparative analysis charts
# for measuring political discourse semantic polarization across EU political groups and time periods.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_distances
from wordcloud import WordCloud
import os
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from utils.plotting_utils import (
    PARTY_COLORS, DEFAULT_COLOR,
    filter_temporal_data,
    filter_temporal_dataframe,
    generate_colors_for_groups,
    generate_colormap_names_for_groups,
    setup_plot_aesthetics,
    save_figure
)


def plot_absolute_polarization(polarization_df, output_dir="../results", temporal_unit='term',
                              target_keyword=None, theme_name=None):
    plt.figure(figsize=(14, 10))

    all_groups = sorted(polarization_df['group'].unique())
    colors = generate_colors_for_groups(all_groups)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

    polarization_df = polarization_df.sort_values('term')

    polarization_df = filter_temporal_dataframe(polarization_df, temporal_unit)

    for group in polarization_df['group'].unique():
        group_data = polarization_df[polarization_df['group'] == group]

        if len(group_data) > 1:
            color = colors.get(group, '#333333')
            ax1.plot(group_data['term'], group_data['distance_from_center'],
                    marker='o', linewidth=2.5, markersize=8, label=group, color=color)

    keyword_part = target_keyword.title() if target_keyword else "Analysis"
    context_part = f" ({theme_name.title()} Context)" if theme_name else " (Baseline Context)"
    title = f'Absolute Polarization: {keyword_part} Analysis{context_part}\nDistance from Semantic Center Over Time'
    ax1.set_title(title, fontsize=16, fontweight='bold')
    xlabel = 'Year' if temporal_unit == 'year' else 'Parliamentary Term'
    ax1.set_xlabel(xlabel, fontsize=12)
    ax1.set_ylabel('Distance from Semantic Center (Cosine Distance)', fontsize=12)
    ax1.legend(title='Political Groups', title_fontsize=12, fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    term_averages = polarization_df.groupby('term')['distance_from_center'].agg(['mean', 'std'])

    ax2.fill_between(term_averages.index,
                     term_averages['mean'] - term_averages['std'],
                     term_averages['mean'] + term_averages['std'],
                     alpha=0.3, color='gray', label='±1 Standard Deviation')
    ax2.plot(term_averages.index, term_averages['mean'],
             marker='s', linewidth=3, markersize=10, color='black', label='Average Polarization')

    temporal_desc = 'Year' if temporal_unit == 'year' else 'Parliamentary Term'
    subtitle = f'Overall {keyword_part} Polarization by {temporal_desc}{context_part}'
    ax2.set_title(subtitle, fontsize=16, fontweight='bold')
    ax2.set_xlabel(temporal_desc, fontsize=12)
    ax2.set_ylabel('Average Distance from Center', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    plt.tight_layout()

    keyword_file = target_keyword.lower() if target_keyword else "analysis"
    theme_file = theme_name.lower() if theme_name else "baseline"
    filename = f'{keyword_file}_{theme_file}_absolute_polarization_trends.png'
    output_path = os.path.join(output_dir, filename)
    save_figure(fig, output_path)

    return output_path


def plot_individual_party_analysis(polarization_df, output_dir="../results", temporal_unit='term',
                                  target_keyword=None, theme_name=None):

    polarization_df = filter_temporal_dataframe(polarization_df, temporal_unit)

    parties = sorted(polarization_df['group'].unique())
    n_parties = len(parties)

    if n_parties <= 4:
        rows, cols = 2, 2
    elif n_parties <= 6:
        rows, cols = 2, 3
    elif n_parties <= 9:
        rows, cols = 3, 3
    elif n_parties <= 12:
        rows, cols = 3, 4
    else:
        rows, cols = 3, 4
        parties = parties[:12]

    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))
    if rows * cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    colors = generate_colors_for_groups(parties)

    for i, party in enumerate(parties[:rows*cols]):
        party_data = polarization_df[polarization_df['group'] == party].sort_values('term')

        if len(party_data) > 1:
            color = colors.get(party, '#333333')

            axes[i].plot(party_data['term'], party_data['distance_from_center'],
                        marker='o', linewidth=3, markersize=10, color=color)
            axes[i].fill_between(party_data['term'],
                               party_data['distance_from_center'] * 0.95,
                               party_data['distance_from_center'] * 1.05,
                               alpha=0.2, color=color)

            z = np.polyfit(party_data['term'], party_data['distance_from_center'], 1)
            p = np.poly1d(z)
            axes[i].plot(party_data['term'], p(party_data['term']),
                        linestyle='--', alpha=0.7, color=color)

            keyword_part = target_keyword.title() if target_keyword else "Analysis"
            context_part = f" ({theme_name.title()})" if theme_name else " (Baseline)"
            axes[i].set_title(f'{party} - Absolute Polarization {keyword_part} Trend{context_part}', fontsize=14, fontweight='bold')
            xlabel = 'Year' if temporal_unit == 'year' else 'Parliamentary Term'
            axes[i].set_xlabel(xlabel, fontsize=11)
            axes[i].set_ylabel('Distance from Center', fontsize=11)
            axes[i].grid(True, alpha=0.3)
            axes[i].set_ylim(bottom=0)

            slope = z[0]
            trend_text = "Increasing" if slope > 0.001 else "Decreasing" if slope < -0.001 else "Stable"
            axes[i].text(0.05, 0.95, f'Trend: {trend_text}',
                        transform=axes[i].transAxes, fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3))

    for j in range(len(parties), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()

    keyword_file = target_keyword.lower() if target_keyword else "analysis"
    theme_file = theme_name.lower() if theme_name else "baseline"
    filename = f'{keyword_file}_{theme_file}_absolute_polarization_individual_party_trends.png'
    output_path = os.path.join(output_dir, filename)
    save_figure(fig, output_path)

    return output_path


def generate_word_clouds(sentence_data, output_dir="../results", target_keyword="security",
                        keyword_group=None, theme_name=None):

    group_texts = {}
    sentence_counts = {}

    for (term, group), sentences in sentence_data.items():
        if group not in group_texts:
            group_texts[group] = []
            sentence_counts[group] = 0
        group_texts[group].extend(sentences)
        sentence_counts[group] += len(sentences)

    sorted_groups = sorted(group_texts.keys(), key=lambda g: len(group_texts[g]), reverse=True)


    n_groups = len(sorted_groups)
    if n_groups <= 4:
        rows, cols = 2, 2
    elif n_groups <= 6:
        rows, cols = 2, 3
    elif n_groups <= 9:
        rows, cols = 3, 3
    elif n_groups <= 12:
        rows, cols = 3, 4
    else:
        rows, cols = 3, 4
        sorted_groups = sorted_groups[:12]

    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
    if rows * cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    colors = generate_colormap_names_for_groups(sorted_groups)

    output_paths = []
    word_stats = {}

    for i, group in enumerate(sorted_groups[:rows*cols]):
        sentences = group_texts[group]

        combined_text = ' '.join(sentences)

        colormap = colors.get(group, 'viridis')

        keyword_stopwords = set()
        if keyword_group:
            keyword_stopwords.update([kw.lower() for kw in keyword_group])
        else:
            keyword_stopwords.add(target_keyword.lower())

        all_stopwords = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'} | keyword_stopwords

        wordcloud = WordCloud(
            width=400, height=300,
            background_color='white',
            colormap=colormap,
            max_words=100,
            relative_scaling=0.5,
            stopwords=all_stopwords
        ).generate(combined_text)

        word_stats[group] = {
            'total_words': len(combined_text.split()),
            'unique_words': len(set(combined_text.lower().split())),
            'top_words': list(wordcloud.words_.keys())[:20],
            'sentences': len(sentences)
        }

        axes[i].imshow(wordcloud, interpolation='bilinear')

        context_name = " + ".join(keyword_group).title() if keyword_group else target_keyword.title()
        theme_context = f" ({theme_name.title()} Theme)" if theme_name else " (Baseline)"
        axes[i].set_title(f'{group} - {context_name} Key Terms{theme_context}',
                         fontsize=14, fontweight='bold')
        axes[i].axis('off')

    for j in range(len(sorted_groups), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()

    if keyword_group:
        filename_part = "_".join(keyword_group)
        display_name = " + ".join(keyword_group)
    else:
        filename_part = target_keyword
        display_name = target_keyword.title()

    theme_file = theme_name.lower() if theme_name else "baseline"
    filename = f'{filename_part}_{theme_file}_word_clouds.png'
    output_path = os.path.join(output_dir, filename)
    save_figure(fig, output_path)


    return output_path


def plot_semantic_fingerprint(polarization_df, output_dir, term=None, target_keyword=None, theme_name=None):

    if term is None:
        term = polarization_df['term'].value_counts().idxmax()

    df_term = polarization_df[polarization_df['term'] == term]

    if df_term.empty:
        return None

    fingerprint_data = df_term.pivot_table(
        index='group',
        columns='keyword',
        values='distance_from_center'
    ).fillna(0)

    keywords = fingerprint_data.columns
    num_vars = len(keywords)

    if num_vars < 3:
        return None

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    all_groups = sorted(fingerprint_data.index.unique())
    colors = generate_colors_for_groups(all_groups)

    fig, ax = plt.subplots(figsize=(14, 14), subplot_kw=dict(polar=True))

    for group, row in fingerprint_data.iterrows():
        values = row.values.flatten().tolist()
        values += values[:1]
        color = colors.get(group, '#333333')

        ax.plot(angles, values, color=color, linewidth=2.5, label=group)
        ax.fill(angles, values, color=color, alpha=0.25)

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(keywords, size=14)

    keyword_part = target_keyword.title() if target_keyword else "Multi-Keyword"
    context_part = f" ({theme_name.title()} Context)" if theme_name else " (Baseline Context)"
    title = f"Semantic Fingerprint: {keyword_part} Analysis{context_part}\nParliamentary Term {int(term)}"
    ax.set_title(title, size=20, weight='bold', y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)


    keyword_file = target_keyword.lower() if target_keyword else "multikeyword"
    theme_file = theme_name.lower() if theme_name else "baseline"
    filename = f'{keyword_file}_{theme_file}_semantic_fingerprint_term_{int(term)}.png'
    output_path = os.path.join(output_dir, filename)
    save_figure(fig, output_path)

    return output_path


def plot_metacluster_spider_chart(clustered_embeddings, macro_cluster_map, macro_theme_labels,
                                 output_dir="../results", temporal_unit='term', target_keyword=None, theme_name=None):

    #extract all political parties and metaclusters
    all_parties = sorted(set(group for term, group in clustered_embeddings.keys()))
    all_metaclusters = sorted(set(macro_theme_labels.values()))

    if len(all_metaclusters) < 3:
        return None

    party_metacluster_matrix = {}
    for party in all_parties:
        party_metacluster_matrix[party] = {mc: 0.0 for mc in all_metaclusters}


    for (term, group), cluster_data in clustered_embeddings.items():
        centroids = cluster_data['centroids']
        labels = cluster_data['labels']

        metacluster_counts = {}
        for cluster_idx in range(len(centroids)):
            if cluster_idx in macro_cluster_map:
                metacluster_idx = macro_cluster_map[cluster_idx]
                metacluster_name = macro_theme_labels.get(metacluster_idx, f"Metacluster_{metacluster_idx}")
                metacluster_counts[metacluster_name] = metacluster_counts.get(metacluster_name, 0) + 1

        total_clusters = sum(metacluster_counts.values())
        if total_clusters > 0:
            for metacluster_name, count in metacluster_counts.items():
                party_metacluster_matrix[group][metacluster_name] += count / total_clusters

    term_counts = {}
    for (term, group) in clustered_embeddings.keys():
        term_counts[group] = term_counts.get(group, 0) + 1

    for party in all_parties:
        if term_counts[party] > 0:
            for metacluster in all_metaclusters:
                party_metacluster_matrix[party][metacluster] /= term_counts[party]

    num_vars = len(all_metaclusters)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    colors = generate_colors_for_groups(all_parties)

    fig, ax = plt.subplots(figsize=(16, 16), subplot_kw=dict(polar=True))

    for party in all_parties:
        values = [party_metacluster_matrix[party][mc] for mc in all_metaclusters]
        values += values[:1]
        color = colors.get(party, '#333333')

        ax.plot(angles, values, color=color, linewidth=2.5, label=party, marker='o', markersize=6)
        ax.fill(angles, values, color=color, alpha=0.15)

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([mc.title() for mc in all_metaclusters], size=14, weight='bold')


    keyword_part = target_keyword.title() if target_keyword else "Analysis"
    context_part = f" ({theme_name.title()} Context)" if theme_name else " (Baseline Context)"
    temporal_desc = "Terms" if temporal_unit == 'term' else "Years"
    title = f'Political Party Engagement with Semantic Themes\n{keyword_part}{context_part} - Averaged Across {temporal_desc}'

    ax.set_title(title, size=18, weight='bold', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1), fontsize=11, title="Political Parties", title_fontsize=12)


    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(max(party_metacluster_matrix[party].values()) for party in all_parties) * 1.1)

    plt.tight_layout()


    keyword_file = target_keyword.lower() if target_keyword else "analysis"
    theme_file = theme_name.lower() if theme_name else "baseline"
    filename = f'{keyword_file}_{theme_file}_metacluster_spider_chart.png'
    output_path = os.path.join(output_dir, filename)
    save_figure(fig, output_path)

    return output_path


def plot_polarization_over_time(polarization_df, run_folder="../results", theme_name=None,
                                temporal_unit='term', target_keyword='security', keyword_group=None):

    print("\nCreating polarization over time visualization...")

    plt.figure(figsize=(14, 10))


    fig, axes = plt.subplots(2, 1, figsize=(14, 12))


    polarization_df = polarization_df.sort_values('term')


    polarization_df = filter_temporal_dataframe(polarization_df, temporal_unit)


    sns.lineplot(data=polarization_df, x='term', y='distance', hue='pair',
                marker='o', linewidth=2.5, markersize=8, ax=axes[0])

    if theme_name:
        title = f'Semantic Distance Evolution: "{target_keyword.title()}" in {theme_name.title()} Context'
    else:
        title = f'Semantic Distance Evolution: "{target_keyword.title()}" Across Political Groups (Baseline)'

    axes[0].set_title(title, fontsize=16, fontweight='bold')
    xlabel = 'Year' if temporal_unit == 'year' else 'Parliamentary Term'
    axes[0].set_xlabel(xlabel, fontsize=12)
    axes[0].set_ylabel('Cosine Distance', fontsize=12)
    axes[0].legend(title='Political Group Pairs', title_fontsize=12, fontsize=10)
    axes[0].grid(True, alpha=0.3)


    axes[0].set_ylim(bottom=0)


    avg_distances = polarization_df.groupby('pair')['distance'].mean().sort_values(ascending=False)
    key_pairs = avg_distances.head(min(5, len(avg_distances))).index.tolist()
    print(f"Selected key pairs for detailed view: {', '.join(key_pairs)}")
    key_data = polarization_df[polarization_df['pair'].isin(key_pairs)]

    if not key_data.empty:
        sns.lineplot(data=key_data, x='term', y='distance', hue='pair',
                    marker='s', linewidth=3, markersize=10, ax=axes[1])

        if theme_name:
            axes[1].set_title(f'Relative Key Ideological Polarization: "{target_keyword.title()}" ({theme_name.title()})',
                             fontsize=16, fontweight='bold')
        else:
            axes[1].set_title(f'Relative Key Ideological Polarization: "{target_keyword.title()}" (Baseline)',
                             fontsize=16, fontweight='bold')
        axes[1].set_xlabel(xlabel, fontsize=12)
        axes[1].set_ylabel('Cosine Distance', fontsize=12)
        axes[1].legend(title='Key Political Divides', title_fontsize=12, fontsize=10)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(bottom=0)

    plt.tight_layout()

    keyword_part = "_".join(keyword_group).lower() if keyword_group else target_keyword.lower()
    theme_part = f"_{theme_name.lower()}" if theme_name else ""
    filename = f"poc_{keyword_part}{theme_part}_polarization_over_time.png"
    output_path = os.path.join(run_folder, "visualizations", filename)
    save_figure(fig, output_path)


    summary_stats = polarization_df.groupby('pair')['distance'].agg(['mean', 'std', 'min', 'max'])

    return output_path


def create_semantic_space_visualization(embeddings, run_folder="../results", theme_name=None,
                                       temporal_unit='term', target_keyword='security', keyword_group=None):

    embeddings = filter_temporal_data(embeddings, temporal_unit)

    embedding_data = []
    labels = []

    all_groups = sorted(set(group for term, group in embeddings.keys()))
    colors_map = generate_colors_for_groups(all_groups)
    print(f"Generated colors for {len(all_groups)} political groups")

    for (term, group), embedding in embeddings.items():
        embedding_data.append(embedding)
        prefix = "Y" if temporal_unit == 'year' else "T"
        labels.append(f"{prefix}{term}-{group}")

    embedding_matrix = np.array(embedding_data)
    print(f"Embedding matrix shape: {embedding_matrix.shape}")

    print("Applying t-SNE dimensionality reduction...")
    print("This may take a few minutes for large datasets...")

    n_samples = len(embedding_data)
    print(f"Number of data points for t-SNE: {n_samples}")

    if n_samples < 4:
        print("Warning: Too few data points for reliable t-SNE visualization")
        print("Consider running with more data or examining results with caution")
        perplexity = 1
    else:
        perplexity = min(30, max(5, (n_samples - 1) // 3))

    print(f"Using perplexity: {perplexity}")

    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, verbose=1)
    embeddings_2d = tsne.fit_transform(embedding_matrix)
    print("t-SNE reduction complete!")

    plt.figure(figsize=(16, 12))

    for i, label in enumerate(labels):
        term_str, group = label.split('-')
        term = int(term_str[1:])

        color = colors_map.get(group, 'gray')
        plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1],
                   c=color, s=100, alpha=0.7, edgecolors='black', linewidth=1)

        plt.annotate(f"T{term}", (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, fontweight='bold')

    if theme_name:
        title = f'Semantic Space: "{target_keyword.title()}" in {theme_name.title()} Context'
    else:
        title = f'Semantic Space: "{target_keyword.title()}" Across Political Groups (Baseline)'

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)

    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                             markersize=10, label=group)
                      for group, color in colors_map.items()]

    if len(colors_map) <= 8:
        plt.legend(handles=legend_elements, title='Political Groups',
                  title_fontsize=12, fontsize=10, loc='best')
    elif len(colors_map) <= 15:
        plt.legend(handles=legend_elements, title='Political Groups',
                  title_fontsize=11, fontsize=9, loc='best', ncol=2)
    else:
        plt.legend(handles=legend_elements, title='Political Groups',
                  title_fontsize=10, fontsize=8, loc='best', ncol=3)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    keyword_file = "_".join(keyword_group).lower() if keyword_group else target_keyword.lower()
    theme_file = f"_{theme_name.lower()}" if theme_name else ""
    filename = f"poc_{keyword_file}{theme_file}_semantic_space_tsne.png"
    output_path = os.path.join(run_folder, "visualizations", filename)
    save_figure(plt.gcf(), output_path)

    print("\nSemantic Space Analysis:")

    group_embeddings = {}
    for (term, group), embedding in embeddings.items():
        if group not in group_embeddings:
            group_embeddings[group] = []
        group_embeddings[group].append(embedding)

    group_avg_embeddings = {}
    for group, embeddings_list in group_embeddings.items():
        group_avg_embeddings[group] = np.mean(embeddings_list, axis=0)

    print("Average semantic distances between political groups:")
    groups = list(group_avg_embeddings.keys())
    group_distances = []
    for i, group1 in enumerate(groups):
        for j, group2 in enumerate(groups[i+1:], i+1):
            dist = cosine_distances(group_avg_embeddings[group1].reshape(1, -1),
                                   group_avg_embeddings[group2].reshape(1, -1))[0, 0]
            print(f"  {group1} - {group2}: {dist:.4f}")
            group_distances.append((group1, group2, dist))

    return output_path


def create_constellation_visualization(clustered_embeddings, run_folder="../results", theme_name=None,
                                      temporal_unit='term', target_keyword="security", aggregation_mode="kmeans",
                                      macro_cluster_map=None, macro_theme_labels=None):

    all_centroids = []
    centroid_info = []


    for (term, group), cluster_data in clustered_embeddings.items():
        centroids = cluster_data['centroids']
        labels = cluster_data['labels']


        unique_labels, counts = np.unique(labels, return_counts=True)
        cluster_sizes = dict(zip(unique_labels, counts))

        for cluster_idx, centroid in enumerate(centroids):
            all_centroids.append(centroid)
            cluster_size = cluster_sizes.get(cluster_idx, 1)
            centroid_info.append((term, group, cluster_idx, cluster_size))

    if len(all_centroids) == 0:
        return None


    embeddings_array = np.array(all_centroids)


    n_samples = len(all_centroids)
    perplexity = min(30, max(5, n_samples // 3))


    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    embeddings_2d = tsne.fit_transform(embeddings_array)


    fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))

    all_groups = sorted(set(info[1] for info in centroid_info))
    all_terms = sorted(set(info[0] for info in centroid_info))

    if macro_cluster_map is not None and macro_theme_labels is not None:

        unique_cluster_ids = sorted(set(macro_theme_labels.keys()))


        colormap = cm.get_cmap('tab10')
        cluster_colors = {}
        for i, cluster_id in enumerate(unique_cluster_ids):
            cluster_key = int(cluster_id)
            color_rgb = colormap(i / max(len(unique_cluster_ids), 2))
            color_hex = '#{:02x}{:02x}{:02x}'.format(
                int(color_rgb[0] * 255),
                int(color_rgb[1] * 255),
                int(color_rgb[2] * 255)
            )
            cluster_colors[cluster_key] = color_hex

        coloring_mode = "cluster"
        colors = cluster_colors
        legend_title = "Meta Clusters"

        legend_items = [f"Cluster {cid}: {macro_theme_labels[cid]}" for cid in unique_cluster_ids]

    else:

        colors = generate_colors_for_groups(all_groups)
        coloring_mode = "group"
        legend_title = "Political Groups"
        legend_items = all_groups


    group_positions = {}
    theme_positions = {}

    for i, (term, group, cluster_idx, cluster_size) in enumerate(centroid_info):
        x, y = embeddings_2d[i]


        if coloring_mode == "cluster" and i in macro_cluster_map:
            macro_cluster_id = macro_cluster_map[i]
            theme_name = macro_theme_labels.get(macro_cluster_id, f"Cluster_{macro_cluster_id}")

            cluster_key = int(macro_cluster_id)
            color = colors.get(cluster_key, '#333333')


            legend_label = f"Cluster {macro_cluster_id}: {theme_name}"
            already_labeled_clusters = set()
            if hasattr(ax1, '_labeled_clusters'):
                already_labeled_clusters = ax1._labeled_clusters
            else:
                ax1._labeled_clusters = set()

            label_for_legend = legend_label if macro_cluster_id not in already_labeled_clusters else ""
            if label_for_legend:
                ax1._labeled_clusters.add(macro_cluster_id)
        else:
            color = colors.get(group, '#333333')
            label_for_legend = group if cluster_idx == 0 else ""


        point_size = max(50, min(500, cluster_size * 20))

        ax1.scatter(x, y, c=color, s=point_size, alpha=0.7,
                   edgecolors='black', linewidth=0.5, label=label_for_legend)

        if group not in group_positions:
            group_positions[group] = []
        group_positions[group].append((x, y, term))


    for group, positions in group_positions.items():
        if len(positions) > 1:
            color = colors.get(group, '#333333')

            positions_sorted = sorted(positions, key=lambda p: p[2])
            for i in range(len(positions_sorted) - 1):
                x1, y1, _ = positions_sorted[i]
                x2, y2, _ = positions_sorted[i + 1]
                ax1.plot([x1, x2], [y1, y2], color=color, alpha=0.3, linewidth=1, linestyle='--')

    ax1.set_title(f'Discourse Constellation: Multi-Centroid View\n"{target_keyword.title()}" - {aggregation_mode.title()} Clustering',
                 fontsize=16, fontweight='bold')
    ax1.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax1.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax1.grid(True, alpha=0.3)

    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), title=legend_title,
              bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    if theme_name:
        filename = f"poc_{target_keyword}_{theme_name}_constellation_{aggregation_mode}.png"
    else:
        filename = f"poc_{target_keyword}_constellation_{aggregation_mode}.png"

    output_path = os.path.join(run_folder, "visualizations", filename)
    save_figure(fig, output_path)

    return output_path


def plot_cluster_diversity(group_diversity, temporal_evolution, output_dir="../results",
                          theme_name=None, target_keyword="security"):

    term_silhouettes = {}

    for group, metrics_list in group_diversity.items():
        for metric in metrics_list:
            term = metric['term']
            silhouette_score = metric['silhouette_score']

            if term not in term_silhouettes:
                term_silhouettes[term] = []

            if silhouette_score > 0:
                term_silhouettes[term].append(silhouette_score)

    if not term_silhouettes:
        return None

    fig, ax = plt.subplots(figsize=(12, 8))

    terms = sorted(term_silhouettes.keys())
    means = []
    stds = []
    counts = []

    for term in terms:
        scores = term_silhouettes[term]
        if scores:
            means.append(np.mean(scores))
            stds.append(np.std(scores) if len(scores) > 1 else 0)
            counts.append(len(scores))
        else:
            means.append(0)
            stds.append(0)
            counts.append(0)


    ax.plot(terms, means, 'b-', linewidth=4, marker='o', markersize=10,
            label='Average Silhouette Score', zorder=3)


    upper_bound = [m + s for m, s in zip(means, stds)]
    lower_bound = [max(0, m - s) for m, s in zip(means, stds)]

    ax.fill_between(terms, lower_bound, upper_bound,
                   alpha=0.3, color='blue', label='±1 Standard Deviation', zorder=1)


    ax.axhline(y=0.25, color='red', linestyle='--', linewidth=2, alpha=0.7,
               label='Good Clustering Threshold', zorder=2)


    for i, (term, mean_score, count) in enumerate(zip(terms, means, counts)):
        ax.annotate(f'{count} parties', (term, mean_score),
                   xytext=(0, 10), textcoords='offset points',
                   ha='center', fontsize=9, alpha=0.7)


    context_part = f" ({theme_name.title()} Context)" if theme_name else " (Baseline Context)"
    ax.set_title(f'Overall Clustering Quality Over Time: {target_keyword.title()}{context_part}\n'
                f'Average Silhouette Score Across All Political Groups',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Parliamentary Term', fontsize=12)
    ax.set_ylabel('Average Silhouette Score', fontsize=12)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0, top=max(1.0, max(upper_bound) * 1.1) if upper_bound else 1.0)

    plt.tight_layout()


    if theme_name:
        filename = f"poc_{target_keyword}_{theme_name}_cluster_analysis.png"
    else:
        filename = f"poc_{target_keyword}_cluster_analysis.png"

    output_path = os.path.join(output_dir, filename)
    save_figure(fig, output_path)

    return output_path


def plot_meta_cluster_centroids(tsne_data, macro_cluster_map, macro_theme_labels,
                               output_dir="../results", target_keyword="security", model_name="roberta"):

    embeddings_2d = tsne_data['embeddings_2d']
    centroid_lookup = tsne_data['centroid_lookup']

    fig, ax = plt.subplots(figsize=(12, 8))

    unique_cluster_ids = sorted(set(macro_theme_labels.keys()))
    colormap = cm.get_cmap('tab10')
    cluster_colors = {}

    for i, cluster_id in enumerate(unique_cluster_ids):
        cluster_key = int(cluster_id)
        color_rgb = colormap(i / max(len(unique_cluster_ids), 2))
        color_hex = '#{:02x}{:02x}{:02x}'.format(
            int(color_rgb[0] * 255),
            int(color_rgb[1] * 255),
            int(color_rgb[2] * 255)
        )
        cluster_colors[cluster_key] = color_hex

    for i, (x, y) in enumerate(embeddings_2d):
        if i in macro_cluster_map:
            cluster_id = macro_cluster_map[i]
            cluster_key = int(cluster_id)
            color = cluster_colors.get(cluster_key, '#333333')
            theme_label = macro_theme_labels.get(cluster_id, f"Cluster_{cluster_id}")

            ax.scatter(x, y, c=color, s=100, alpha=0.7,
                      edgecolors='black', linewidth=0.5, label=theme_label)

    ax.set_title(f'Meta-Cluster Analysis: "{target_keyword.title()}" Centroids\nClustered by Semantic Themes',
                fontsize=16, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.grid(True, alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title='Semantic Themes',
             bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    filename = f"meta_cluster_analysis_{target_keyword}_{model_name}.png"
    output_path = os.path.join(output_dir, filename)
    save_figure(fig, output_path)

    return output_path


def plot_polarization_profile(absolute_polarization_df, output_dir="../results", temporal_unit='term',
                             target_keyword='security', theme_name=None):

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 20))
    plt.subplots_adjust(hspace=0.4)

    absolute_polarization_df = absolute_polarization_df.sort_values('term')

    absolute_polarization_df = filter_temporal_dataframe(absolute_polarization_df, temporal_unit)

    system_metrics = absolute_polarization_df.groupby('term').agg({
        'absolute_extremity': 'sum',
        'absolute_overlap': 'sum',
        'absolute_range': 'sum'
    }).reset_index()

    xlabel = 'Year' if temporal_unit == 'year' else 'Parliamentary Term'


    keyword_part = target_keyword.title() if target_keyword else "Analysis"
    context_part = f" ({theme_name.title()} Context)" if theme_name else " (Baseline Context)"


    ax1.plot(system_metrics['term'], system_metrics['absolute_extremity'],
            marker='o', linewidth=4, markersize=8, color='#1f77b4', markerfacecolor='white',
            markeredgewidth=2, markeredgecolor='#1f77b4')

    ax1.set_title(f'Total System Extremity: {keyword_part}{context_part}\n(Collective Ideological Stretch)',
                 fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel('Sum of Maximum Distances', fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)


    ax1.text(0.5, -0.15, 'Higher values indicate the parliamentary system contains more radical outlier positions across all parties',
            transform=ax1.transAxes, ha='center', va='top', fontsize=11, style='italic',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3))


    ax2.plot(system_metrics['term'], system_metrics['absolute_overlap'],
            marker='^', linewidth=4, markersize=8, color='#2ca02c', markerfacecolor='white',
            markeredgewidth=2, markeredgecolor='#2ca02c')

    ax2.set_title(f'Total Mainstream Adherence: {keyword_part}{context_part}\n(System-Wide Consensus Distance)',
                 fontsize=16, fontweight='bold', pad=20)
    ax2.set_ylabel('Sum of Minimum Distances', fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)


    ax2.text(0.5, -0.15, 'Lower values indicate stronger system-wide consensus, with all parties adhering closely to the mainstream average',
            transform=ax2.transAxes, ha='center', va='top', fontsize=11, style='italic',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.3))


    ax3.plot(system_metrics['term'], system_metrics['absolute_range'],
            marker='D', linewidth=4, markersize=8, color='#d62728', markerfacecolor='white',
            markeredgewidth=2, markeredgecolor='#d62728')

    ax3.set_title(f'Total System Fragmentation: {keyword_part}{context_part}\n(Collective Internal Incoherence)',
                 fontsize=16, fontweight='bold', pad=20)
    ax3.set_xlabel(xlabel, fontsize=13)
    ax3.set_ylabel('Sum of Fragmentation Scores', fontsize=13)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(bottom=0)


    ax3.text(0.5, -0.15, "Higher values indicate all parties are internally divided ('big tent' mode); lower values mean unified, disciplined messaging",
            transform=ax3.transAxes, ha='center', va='top', fontsize=11, style='italic',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.3))


    keyword_file = target_keyword.lower() if target_keyword else "analysis"
    theme_file = theme_name.lower() if theme_name else "baseline"
    filename = f'{keyword_file}_{theme_file}_system_polarization_profile.png'
    output_path = os.path.join(output_dir, filename)
    save_figure(fig, output_path)

    return output_path


def plot_polarization_spectrum(advanced_polarization_df, party1, party2, output_dir="../results",
                               temporal_unit='term', target_keyword='security', theme_name=None):

    pair_key = f"{party1} vs {party2}"
    reverse_pair_key = f"{party2} vs {party1}"

    pair_data = advanced_polarization_df[
        (advanced_polarization_df['pair'] == pair_key) |
        (advanced_polarization_df['pair'] == reverse_pair_key)
    ].copy()

    if pair_data.empty:
        return None

    pair_data = pair_data.sort_values('term')

    pair_data = filter_temporal_dataframe(pair_data, temporal_unit)

    fig, ax = plt.subplots(figsize=(16, 8))

    terms = pair_data['term']


    extremity_line = ax.plot(terms, pair_data['discourse_extremity'],
                            color='red', linewidth=3, marker='v', markersize=8,
                            label='Discourse Extremity (Max Disagreement)', zorder=3)


    overlap_line = ax.plot(terms, pair_data['semantic_overlap'],
                          color='green', linewidth=3, marker='^', markersize=8,
                          label='Semantic Overlap (Bridge Points)', zorder=3)


    ax.fill_between(terms, pair_data['semantic_overlap'], pair_data['discourse_extremity'],
                   alpha=0.3, color='lightblue',
                   label='Polarization Range (Full Spectrum)', zorder=1)


    avg_line = ax.plot(terms, pair_data['average_set_polarization'],
                      color='darkblue', linewidth=4, marker='o', markersize=6,
                      label='Average Set Polarization (Center of Gravity)', zorder=2)


    keyword_part = target_keyword.title() if target_keyword else "Analysis"
    context_part = f" ({theme_name.title()} Context)" if theme_name else " (Baseline Context)"
    xlabel = 'Year' if temporal_unit == 'year' else 'Parliamentary Term'

    ax.set_title(f'Relative Polarization Spectrum: {party1} vs {party2}\n{keyword_part}{context_part}',
                fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel('Cosine Distance', fontsize=14)


    ax.legend(loc='upper left', fontsize=12, framealpha=0.9)


    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    range_width = pair_data['discourse_extremity'] - pair_data['semantic_overlap']
    max_range_idx = range_width.idxmax()
    min_range_idx = range_width.idxmin()

    if len(pair_data) > 1:
        max_range_term = pair_data.loc[max_range_idx, 'term']
        max_range_value = range_width.loc[max_range_idx]

        ax.annotate(f'Widest Range\n(Term {max_range_term}: {max_range_value:.3f})',
                   xy=(max_range_term, pair_data.loc[max_range_idx, 'average_set_polarization']),
                   xytext=(10, 30), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))


    plt.tight_layout()


    keyword_file = target_keyword.lower() if target_keyword else "analysis"
    theme_file = theme_name.lower() if theme_name else "baseline"
    safe_party1 = party1.replace('/', '_').replace('&', 'and')
    safe_party2 = party2.replace('/', '_').replace('&', 'and')
    filename = f'{keyword_file}_{theme_file}_{safe_party1}_vs_{safe_party2}_spectrum.png'
    output_path = os.path.join(output_dir, filename)
    save_figure(fig, output_path)

    return output_path


def plot_comparative_constellation(clustered_embeddings, party1, party2, output_dir="../results",
                                  temporal_unit='term', target_keyword='security', theme_name=None):


    filtered_embeddings = {}
    for (term, group), cluster_data in clustered_embeddings.items():
        if group in [party1, party2]:
            filtered_embeddings[(term, group)] = cluster_data

    if not filtered_embeddings:
        return None

    all_centroids = []
    centroid_info = []

    for (term, group), cluster_data in filtered_embeddings.items():
        centroids = cluster_data['centroids']
        for cluster_idx, centroid in enumerate(centroids):
            all_centroids.append(centroid)
            centroid_info.append((term, group, cluster_idx))

    if len(all_centroids) < 4:
        return None

    embeddings_array = np.array(all_centroids)
    n_samples = len(all_centroids)
    perplexity = min(30, max(5, n_samples // 3))

    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    embeddings_2d = tsne.fit_transform(embeddings_array)

    fig, ax = plt.subplots(figsize=(12, 10))

    all_terms = sorted(set(info[0] for info in centroid_info))

    party1_colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(all_terms)))
    party2_colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(all_terms)))

    party1_color_map = {term: party1_colors[i] for i, term in enumerate(all_terms)}
    party2_color_map = {term: party2_colors[i] for i, term in enumerate(all_terms)}

    plotted_terms = set()

    for i, (term, group, cluster_idx) in enumerate(centroid_info):
        x, y = embeddings_2d[i]

        if group == party1:
            color = party1_color_map[term]
            marker = 'o'
        else:
            color = party2_color_map[term]
            marker = 's'

        ax.scatter(x, y, c=[color], s=120, marker=marker,
                  edgecolors='black', linewidth=1, alpha=0.8)


    min_distance = float('inf')
    max_distance = 0
    min_pair = None
    max_pair = None


    for i, (term1, group1, cluster1) in enumerate(centroid_info):
        for j, (term2, group2, cluster2) in enumerate(centroid_info):
            if i < j and group1 != group2:

                orig_distance = cosine_distances([all_centroids[i]], [all_centroids[j]])[0, 0]

                if orig_distance < min_distance:
                    min_distance = orig_distance
                    min_pair = (i, j, term1, term2)
                if orig_distance > max_distance:
                    max_distance = orig_distance
                    max_pair = (i, j, term1, term2)


    if min_pair:
        i, j, term1, term2 = min_pair
        ax.plot([embeddings_2d[i, 0], embeddings_2d[j, 0]],
               [embeddings_2d[i, 1], embeddings_2d[j, 1]],
               'g-', linewidth=3, alpha=0.7, label=f'Semantic Bridge (T{term1}-T{term2})')

    if max_pair and max_pair != min_pair:
        i, j, term1, term2 = max_pair
        ax.plot([embeddings_2d[i, 0], embeddings_2d[j, 0]],
               [embeddings_2d[i, 1], embeddings_2d[j, 1]],
               'r-', linewidth=3, alpha=0.7, label=f'Max Extremity (T{term1}-T{term2})')


    keyword_part = target_keyword.title() if target_keyword else "Analysis"
    context_part = f" ({theme_name.title()} Context)" if theme_name else " (Baseline Context)"

    ax.set_title(f'Comparative Constellation: {party1} vs {party2}\n{keyword_part}{context_part}',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)


    legend_elements = []


    for i, term in enumerate(all_terms):
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                        markerfacecolor=party1_color_map[term],
                                        markersize=10, label=f'{party1} T{term}'))

    for i, term in enumerate(all_terms):
        legend_elements.append(plt.Line2D([0], [0], marker='s', color='w',
                                        markerfacecolor=party2_color_map[term],
                                        markersize=10, label=f'{party2} T{term}'))

    ax.legend(handles=legend_elements, title='Party-Term Combinations',
             loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=10)


    if min_pair and max_pair:
        ax.text(0.02, 0.98, f'Semantic Bridge Distance: {min_distance:.4f}',
               transform=ax.transAxes, fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))

        ax.text(0.02, 0.92, f'Maximum Extremity Distance: {max_distance:.4f}',
               transform=ax.transAxes, fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))

    ax.grid(True, alpha=0.3)
    plt.tight_layout()


    keyword_file = target_keyword.lower() if target_keyword else "analysis"
    theme_file = theme_name.lower() if theme_name else "baseline"
    safe_party1 = party1.replace('/', '_').replace('&', 'and')
    safe_party2 = party2.replace('/', '_').replace('&', 'and')
    filename = f'{keyword_file}_{theme_file}_{safe_party1}_vs_{safe_party2}_constellation.png'
    output_path = os.path.join(output_dir, filename)
    save_figure(fig, output_path)

    return output_path