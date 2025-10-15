
import sys
import os
import pickle
import argparse
import pandas as pd


sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:

    from analysis_tools import (
        calculate_absolute_polarization,
        calculate_polarization_metrics,
        get_dynamic_theme_vectors,
        perform_meta_clustering_analysis,
        analyze_cluster_diversity,
        calculate_absolute_advanced_metrics
    )

    from visualization_suite import (
        plot_absolute_polarization,
        plot_individual_party_analysis,
        plot_semantic_fingerprint,
        plot_metacluster_spider_chart,
        create_constellation_visualization,
        plot_cluster_diversity,
        create_semantic_space_visualization,
        plot_polarization_over_time,
        plot_polarization_profile,
        plot_polarization_spectrum,
        plot_comparative_constellation
    )
    import numpy as np

    print("[OK] Modules loaded successfully")
except ImportError as e:
    print(f"[ERROR] Error importing modules: {e}")
    sys.exit(1)


def get_latest_run_folder():

    runs_dir = os.path.join("results", "runs")
    if not os.path.exists(runs_dir): return None
    run_folders = [d for d in os.listdir(runs_dir) if d.startswith("run_") and os.path.isdir(os.path.join(runs_dir, d))]
    if not run_folders: return None
    latest_folder = sorted(run_folders, reverse=True)[0]
    return os.path.join(runs_dir, latest_folder)


def find_available_analyses(run_folder):

    data_dir = os.path.join(run_folder, "data")
    if not os.path.exists(data_dir): 
        return {}

    analyses = {}
    for filename in os.listdir(data_dir):

        if filename.startswith("poc_") and (filename.endswith("_avg_embeddings.pkl") or filename.endswith("_clustered_data.pkl")):
            if filename.endswith("_avg_embeddings.pkl"):
                parts = filename.replace("poc_", "").replace("_avg_embeddings.pkl", "").split("_")
                aggregation_mode = "average"
            else:
                parts = filename.replace("poc_", "").replace("_clustered_data.pkl", "").split("_")
                aggregation_mode = "kmeans"
            
            model = parts[-1]
            keyword_with_context = "_".join(parts[:-1])
            

            context = "baseline"
            keyword = keyword_with_context
            

            if "combined" in keyword_with_context:
                context = "combined"
                keyword = keyword_with_context.replace("_combined", "")
            else:

                known_themes = ["security", "health", "immigration", "enlargement", "war", "economy", 
                               "environment", "energy", "education", "digitalization", "democracy", 
                               "agriculture", "transport", "social"]
                for theme in known_themes:
                    if f"_{theme}_" in f"_{keyword_with_context}_":

                        keyword_parts = keyword_with_context.split("_")
                        if theme in keyword_parts:
                            theme_idx = keyword_parts.index(theme)
                            if theme_idx > 0:
                                context = theme
                                keyword = "_".join(keyword_parts[:theme_idx] + keyword_parts[theme_idx+1:])
                                break


            sentences_file = f"poc_{keyword_with_context}_sentences.pkl"
            if os.path.exists(os.path.join(data_dir, sentences_file)):
                analysis_key = f"{keyword}_{context}" if context != "baseline" else keyword
                
                if analysis_key not in analyses:
                    analyses[analysis_key] = {
                        'keyword': keyword,
                        'context': context,
                        'modes': {}
                    }
                analyses[analysis_key]['modes'][aggregation_mode] = model
    
    return analyses


def list_available_keywords(run_folder):

    analyses = find_available_analyses(run_folder)
    if not analyses:
        print("[ERROR] No completed analyses found in this run folder.")
        return
    
    print(f"\n[INFO] Available Keywords in {run_folder}:")
    print("=" * 60)
    

    contexts = {}
    for analysis_key, analysis_data in analyses.items():
        context = analysis_data['context']
        keyword = analysis_data['keyword']
        modes = analysis_data['modes']
        
        if context not in contexts:
            contexts[context] = []
        contexts[context].append({
            'keyword': keyword,
            'modes': list(modes.keys()),
            'models': list(set(modes.values()))
        })
    

    for context, keywords in contexts.items():
        print(f"\n[CONTEXT]  {context.upper()} Context:")
        for i, item in enumerate(keywords, 1):
            modes_str = ", ".join(item['modes'])
            models_str = ", ".join(item['models'])
            print(f"   {i}. {item['keyword']} (modes: {modes_str}, models: {models_str})")
    
    print(f"\n📈 Total: {len(analyses)} analysis combinations available")


def interactive_keyword_selection(available_analyses):

    if not available_analyses:
        print("[ERROR] No analyses available for selection.")
        return [], False
    
    print("\n[INTERACTIVE] Interactive Keyword Selection")
    print("=" * 50)
    

    analysis_list = list(available_analyses.items())
    for i, (analysis_key, analysis_data) in enumerate(analysis_list, 1):
        keyword = analysis_data['keyword']
        context = analysis_data['context']
        modes = list(analysis_data['modes'].keys())
        print(f"{i:2d}. {keyword} ({context}) - {', '.join(modes)}")
    
    print("\nSelect keywords by number (e.g., '1,3,5' or '1-3' or 'all'):")
    selection = input("> ").strip().lower()
    
    selected_keys = []
    if selection == 'all':
        selected_keys = [key for key, _ in analysis_list]
    elif ',' in selection:

        try:
            indices = [int(x.strip()) for x in selection.split(',')]
            selected_keys = [analysis_list[i-1][0] for i in indices if 1 <= i <= len(analysis_list)]
        except (ValueError, IndexError):
            print("[ERROR] Invalid selection format.")
            return [], False
    elif '-' in selection:

        try:
            start, end = map(int, selection.split('-'))
            selected_keys = [analysis_list[i-1][0] for i in range(start, end+1) if 1 <= i <= len(analysis_list)]
        except (ValueError, IndexError):
            print("[ERROR] Invalid range format.")
            return [], False
    else:

        try:
            index = int(selection)
            if 1 <= index <= len(analysis_list):
                selected_keys = [analysis_list[index-1][0]]
            else:
                print("[ERROR] Number out of range.")
                return [], False
        except ValueError:
            print("[ERROR] Invalid selection format.")
            return [], False
    
    if not selected_keys:
        print("[ERROR] No valid keywords selected.")
        return [], False
    
    print(f"[SELECTED] Selected: {', '.join([available_analyses[key]['keyword'] for key in selected_keys])}")
    

    if len(selected_keys) > 1:
        combine_input = input("[COMBINE] Combine these keywords in comparative analysis? (y/n): ").strip().lower()
        combine = combine_input in ['y', 'yes', '1', 'true']
    else:
        combine = False
    
    return selected_keys, combine


def detect_temporal_unit(embeddings):

    if not embeddings: return 'term'
    sample_temporal_value = next(iter(embeddings.keys()))[0]
    return 'year' if sample_temporal_value > 100 else 'term'


def parse_party_pairs(party_pairs_args):

    if not party_pairs_args:

        return [
            ("GUENGL", "IND"),
            ("GUENGL", "ECR"),
            ("GEFA", "IND"),
            ("SOCCESPASD", "IND"),
            ("ALDERE", "IND")
        ]

    parsed_pairs = []
    for pair_str in party_pairs_args:
        if " vs " in pair_str:
            party1, party2 = pair_str.split(" vs ", 1)
            parsed_pairs.append((party1.strip(), party2.strip()))
        else:
            pass

    return parsed_pairs


def main():

    parser = argparse.ArgumentParser(
        description="Regenerate visualizations for a completed pipeline run with keyword selection.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'run_folder', nargs='?', default='latest',
        help='Path to the run folder (e.g., results/runs/run_...). Defaults to "latest".'
    )
    parser.add_argument(
        '--keywords', '-k', nargs='+',
        help='Select specific keywords to process (e.g., --keywords security migration defense). '
             'If not specified, processes all available keywords.'
    )
    parser.add_argument(
        '--combine', '-c', action='store_true',
        help='Combine selected keywords in comparative analysis (only works with multiple keywords).'
    )
    parser.add_argument(
        '--list', '-l', action='store_true',
        help='List all available keywords with their contexts and exit.'
    )
    parser.add_argument(
        '--interactive', '-i', action='store_true',
        help='Interactive keyword selection mode.'
    )
    parser.add_argument(
        '--weighted', action='store_true',
        help='Use weighted centroid calculation for absolute polarization (corrects for data imbalances).'
    )
    parser.add_argument(
        '--aggregation-mode', '-a', choices=['average', 'kmeans', 'auto'], default='auto',
        help='Aggregation mode to use: average, kmeans, or auto (detect from available files).'
    )
    parser.add_argument(
        '--party-pairs', nargs='+',
        help='Specify party pairs for focused analysis (e.g., "EPP vs S&D" "ALDE vs ECR"). '
             'Only applies to k-means mode. Uses default pairs if not specified.'
    )
    parser.add_argument(
        '--force-new-themes', action='store_true',
        help='Force regeneration of theme vectors and meta-clustering analysis using updated clustering themes. '
             'Use this when you have modified the clustering themes and want to apply them to existing runs.'
    )

    parser.add_argument(
        '--keyword', type=str,
        help='(Legacy) Single keyword selection - use --keywords instead.'
    )
    args = parser.parse_args()


    if args.run_folder.lower() == 'latest':
        run_folder = get_latest_run_folder()
        if not run_folder:
            print("[ERROR] No run folders found.")
            return
    else:
        run_folder = args.run_folder

    if not os.path.isdir(run_folder):
        print(f"[ERROR] Run folder not found: {run_folder}")
        return

    available_analyses = find_available_analyses(run_folder)
    if not available_analyses:
        print("[ERROR] No completed analyses found in the specified run folder.")
        return


    if args.list:
        list_available_keywords(run_folder)
        return


    selected_analysis_keys = []
    combine_keywords = args.combine
    
    if args.interactive:
        selected_analysis_keys, combine_keywords = interactive_keyword_selection(available_analyses)
        if not selected_analysis_keys:
            print("[ERROR] No keywords selected. Exiting.")
            return
    elif args.keywords:

        selected_keywords = set(args.keywords)
        selected_analysis_keys = []
        for analysis_key, analysis_data in available_analyses.items():
            keyword = analysis_data['keyword']
            if keyword in selected_keywords:
                selected_analysis_keys.append(analysis_key)
                selected_keywords.discard(keyword)  # Mark as found

        if selected_keywords:
            print(f"[WARNING]  Keywords not found: {', '.join(selected_keywords)}")
            print(f"Available keywords: {', '.join(set(data['keyword'] for data in available_analyses.values()))}")
        
        if not selected_analysis_keys:
            print("[ERROR] No valid keywords selected.")
            return
    elif args.keyword:

        selected_analysis_keys = [k for k in available_analyses.keys() 
                                if available_analyses[k]['keyword'] == args.keyword]
        if not selected_analysis_keys:
            print(f"[ERROR] Keyword '{args.keyword}' not found.")
            print(f"Available keywords: {', '.join(set(data['keyword'] for data in available_analyses.values()))}")
            return
    else:

        selected_analysis_keys = list(available_analyses.keys())


    target_analyses = {k: v for k, v in available_analyses.items() if k in selected_analysis_keys}
    

    log_dir = os.path.join(run_folder, "logs")
    os.makedirs(log_dir, exist_ok=True)  # Ensure logs directory exists
    log_file = os.path.join(log_dir, "visualization_regeneration.log")
    
    keywords_to_process = [available_analyses[k]['keyword'] for k in selected_analysis_keys]

    all_polarization_data = []

    for analysis_key, analysis_data in target_analyses.items():
        keyword = analysis_data['keyword']
        context = analysis_data['context']
        mode_data = analysis_data['modes']
        

        modes_to_process = []

        if args.aggregation_mode == 'auto':

            if 'average' in mode_data:
                modes_to_process.append(('average', mode_data['average']))
            if 'kmeans' in mode_data:
                modes_to_process.append(('kmeans', mode_data['kmeans']))

            if not modes_to_process:
                continue
        else:

            if args.aggregation_mode in mode_data:
                modes_to_process.append((args.aggregation_mode, mode_data[args.aggregation_mode]))
            else:
                continue


        first_mode, first_model = modes_to_process[0]


        if context == "baseline":
            keyword_with_context = keyword
        elif context == "combined":
            keyword_with_context = f"{keyword}_combined"
        else:
            keyword_with_context = f"{keyword}_{context}"


        if first_mode == "average":
            first_embeddings_path = os.path.join(run_folder, "data", f"poc_{keyword_with_context}_{first_model}_avg_embeddings.pkl")
        elif first_mode == "kmeans":
            first_embeddings_path = os.path.join(run_folder, "data", f"poc_{keyword_with_context}_{first_model}_clustered_data.pkl")

        sentences_path = os.path.join(run_folder, "data", f"poc_{keyword_with_context}_sentences.pkl")

        try:
            with open(first_embeddings_path, 'rb') as f:
                first_raw_data = pickle.load(f)
            with open(sentences_path, 'rb') as f:
                sentences = pickle.load(f)

            if first_mode == "kmeans":

                first_embeddings = {}
                for group_key, cluster_data in first_raw_data.items():
                    centroids = cluster_data['centroids']
                    labels = cluster_data['labels']

                    if len(centroids) > 0:
                        if len(centroids) == 1:
                            first_embeddings[group_key] = centroids[0]
                        else:

                            unique_labels, counts = np.unique(labels, return_counts=True)
                            weights = counts / len(labels)
                            weighted_centroid = np.average(centroids, axis=0, weights=weights)
                            first_embeddings[group_key] = weighted_centroid
            else:

                first_embeddings = first_raw_data

            viz_dir = os.path.join(run_folder, "visualizations")
            os.makedirs(viz_dir, exist_ok=True)
            temporal_unit = detect_temporal_unit(first_embeddings)



            if args.weighted:
                polarization_df = calculate_absolute_polarization(first_embeddings, run_folder, temporal_unit,
                                                                sentence_data=sentences, use_weighted_centroid=True)
            else:
                polarization_df = calculate_absolute_polarization(first_embeddings, run_folder, temporal_unit)


            theme_context = context


            plot_absolute_polarization(polarization_df, viz_dir, temporal_unit, keyword, theme_context)
            plot_individual_party_analysis(polarization_df, viz_dir, temporal_unit, keyword, theme_context)


            polarization_df['keyword'] = keyword
            all_polarization_data.append(polarization_df)


        except Exception as e:
            continue


        for aggregation_mode, model in modes_to_process:

            if context == "baseline":
                keyword_with_context = keyword
            elif context == "combined":
                keyword_with_context = f"{keyword}_combined"
            else:
                keyword_with_context = f"{keyword}_{context}"


            if aggregation_mode == "average":
                embeddings_path = os.path.join(run_folder, "data", f"poc_{keyword_with_context}_{model}_avg_embeddings.pkl")
            elif aggregation_mode == "kmeans":
                embeddings_path = os.path.join(run_folder, "data", f"poc_{keyword_with_context}_{model}_clustered_data.pkl")
            else:
                continue
            
            sentences_path = os.path.join(run_folder, "data", f"poc_{keyword_with_context}_sentences.pkl")

            try:
                with open(embeddings_path, 'rb') as f:
                    raw_data = pickle.load(f)
                with open(sentences_path, 'rb') as f:
                    sentences = pickle.load(f)
                

                if aggregation_mode == "kmeans":

                    embeddings = {}
                    clustered_embeddings = raw_data
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
                else:

                    embeddings = raw_data
                    clustered_embeddings = None

                viz_dir = os.path.join(run_folder, "visualizations")
                os.makedirs(viz_dir, exist_ok=True)

                temporal_unit = detect_temporal_unit(embeddings)
                theme_context = context


                if aggregation_mode == "average":

                    polarization_metrics_df = calculate_polarization_metrics(
                        embeddings, run_folder, theme_context, temporal_unit
                    )


                    semantic_plot = create_semantic_space_visualization(
                        embeddings, viz_dir, theme_context, temporal_unit,
                        keyword, keyword_group=None
                    )
                    if semantic_plot:
                        pass


                    polarization_plot = plot_polarization_over_time(
                        polarization_metrics_df, viz_dir, theme_context,
                        temporal_unit, keyword, keyword_group=None
                    )
                    if polarization_plot:
                        pass


                if aggregation_mode == "kmeans" and clustered_embeddings is not None:

                    if args.force_new_themes:

                        cache_filename = f"clustering_vectors_{model}_sans_{keyword}.pkl"
                        cache_path = os.path.join(run_folder, "data", cache_filename)
                        if os.path.exists(cache_path):
                            os.remove(cache_path)


                    dynamic_theme_vectors = get_dynamic_theme_vectors(keyword, model, run_folder)

                    macro_cluster_map, macro_theme_labels, tsne_data = perform_meta_clustering_analysis(
                        clustered_embeddings, sentences,
                        theme_concept_vectors=dynamic_theme_vectors,
                        target_keyword=keyword, model_name=model, run_folder=viz_dir,
                        k_min=2, k_max=10, silhouette_threshold=0.25
                    )


                    constellation_plot = create_constellation_visualization(
                        clustered_embeddings, viz_dir, temporal_unit, target_keyword=keyword,
                        macro_cluster_map=macro_cluster_map, macro_theme_labels=macro_theme_labels
                    )
                    if constellation_plot:
                        pass

                    # # Generate metacluster spider chart
                    # spider_plot = plot_metacluster_spider_chart(
                    #     clustered_embeddings, macro_cluster_map, macro_theme_labels,
                    #     viz_dir, temporal_unit, keyword, theme_name=context
                    # )
                    # if spider_plot:
                    #     pass
                    # Generate cluster diversity analysis
                    cluster_analysis_results = analyze_cluster_diversity(
                        clustered_embeddings, sentences, viz_dir, theme_name=context, target_keyword=keyword
                    )

                    if cluster_analysis_results:

                        cluster_plot_path = plot_cluster_diversity(
                            cluster_analysis_results['group_diversity'],
                            cluster_analysis_results['temporal_evolution'],
                            viz_dir, theme_name=context, target_keyword=keyword
                        )

                    absolute_polarization_df = calculate_absolute_advanced_metrics(
                        clustered_embeddings, temporal_unit
                    )

                    if not absolute_polarization_df.empty:
                        profile_plot_path = plot_polarization_profile(
                            absolute_polarization_df, viz_dir, temporal_unit,
                            keyword, theme_name=context
                        )


                        party_pairs = parse_party_pairs(args.party_pairs)

                        for party1, party2 in party_pairs:
                            # Note: Polarization spectrum plot disabled for absolute metrics
                            # (spectrum plot is designed for relative pair-based analysis)
                            # spectrum_plot = plot_polarization_spectrum(
                            #     absolute_polarization_df, party1, party2, viz_dir,
                            #     temporal_unit, keyword, theme_name=context
                            # )

                            # # Generate comparative constellation plot
                            # constellation_plot = plot_comparative_constellation(
                            #     clustered_embeddings, party1, party2, viz_dir,
                            #     temporal_unit, keyword, theme_name=context
                            # )
                            # if constellation_plot:
                            #     pass
                            pass
                    else:
                        pass
                else:
                    pass
            except Exception as e:
                continue


    single_keyword_mode = (args.keyword or 
                          (args.keywords and len(args.keywords) == 1) or
                          (args.interactive and len(selected_analysis_keys) == 1))
    
    if combine_keywords and len(all_polarization_data) > 1:
        combined_df = pd.concat(all_polarization_data, ignore_index=True)

        comp_viz_dir = os.path.join(run_folder, "comparative_visualizations")
        os.makedirs(comp_viz_dir, exist_ok=True)


        selected_keywords = list(set(keywords_to_process))
        combined_name = "_".join(selected_keywords[:3])
        if len(selected_keywords) > 3:
            combined_name += f"_and_{len(selected_keywords)-3}_more"

        plot_semantic_fingerprint(combined_df, output_dir=comp_viz_dir, target_keyword=combined_name, theme_name="selective_combined")
        
    elif not single_keyword_mode and len(all_polarization_data) > 2:
        combined_df = pd.concat(all_polarization_data, ignore_index=True)

        comp_viz_dir = os.path.join(run_folder, "comparative_visualizations")
        os.makedirs(comp_viz_dir, exist_ok=True)

        plot_semantic_fingerprint(combined_df, output_dir=comp_viz_dir, target_keyword=None, theme_name="comparative")
        
    elif single_keyword_mode:
        pass
    else:
        pass

    print(f"\n[SUCCESS] Visualization generation complete!")
    print(f"[INFO] Processed {len(target_analyses)} analyses for keywords: {', '.join(keywords_to_process)}")
    if combine_keywords:
        print("[COMBINE] Generated combined comparative visualization")
    
    viz_dir = os.path.join(run_folder, "visualizations")
    print(f"[FOLDER] Individual visualizations: {viz_dir}")
    
    if not single_keyword_mode and len(all_polarization_data) > 2:
        comp_viz_dir = os.path.join(run_folder, "comparative_visualizations")
        print(f"[FOLDER] Comparative visualizations: {comp_viz_dir}")



if __name__ == "__main__":
    main()