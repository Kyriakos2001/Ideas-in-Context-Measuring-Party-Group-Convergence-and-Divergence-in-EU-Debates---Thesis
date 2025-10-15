import subprocess
import sys
import os
from datetime import datetime


sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from analysis_tools import calculate_absolute_polarization, generate_summary_report
from visualization_suite import plot_absolute_polarization, plot_individual_party_analysis, generate_word_clouds
import pickle
import numpy as np

from poc_phase1_data_prep import THEMATIC_FILTERS

def run_script_with_run_folder(script_name, description, phase_num, total_phases, run_folder, theme_name=None, theme_keywords=None, temporal_unit='term', target_keyword='security', model_name='roberta', keyword_group=None, min_sentences=0, aggregation_mode='average', k_min=2, k_max=8, silhouette_threshold=0.25):
    context_info = f" ({theme_name})" if theme_name else ""
    temporal_info = f" [{temporal_unit}]" if temporal_unit == 'year' else ""
    keyword_info = f" [grouped: {'+'.join(keyword_group)}]" if keyword_group else f" [single: {target_keyword}]"
    print(f"\n[{phase_num}/{total_phases}] {description}{context_info}{temporal_info}{keyword_info}")
    
    try:
        if 'phase1' in script_name:
            from poc_phase1_data_prep import main
            if theme_name is not None:
                main(run_folder, theme_name, theme_keywords, temporal_unit, target_keyword, keyword_group=keyword_group, min_sentence_threshold=min_sentences)
            else:
                main(run_folder, temporal_unit=temporal_unit, target_keyword=target_keyword, keyword_group=keyword_group, min_sentence_threshold=min_sentences)
        elif 'phase2' in script_name:
            from poc_phase2_embeddings import main
            if theme_name is not None:
                main(run_folder, theme_name, temporal_unit, target_keyword, model_name, keyword_group=keyword_group)
            else:
                main(run_folder, temporal_unit=temporal_unit, target_keyword=target_keyword, model_name=model_name, keyword_group=keyword_group)
        elif 'phase3' in script_name:
            from poc_phase3_aggregation import main
            if theme_name is not None:
                main(run_folder, aggregation_mode, target_keyword, theme_name, model_name, keyword_group, k_min=k_min, k_max=k_max, silhouette_threshold=silhouette_threshold)
            else:
                main(run_folder, aggregation_mode, target_keyword, model_name=model_name, keyword_group=keyword_group, k_min=k_min, k_max=k_max, silhouette_threshold=silhouette_threshold)
        else:
            result = subprocess.run([sys.executable, script_name],
                                  capture_output=True, text=True, check=True)
            print(result.stdout)
            if result.stderr:
                print("WARNINGS/ERRORS:", result.stderr)
        
        context_info = f" ({theme_name})" if theme_name else ""
        print(f"[SUCCESS] {description}{context_info} completed")
        return True
    except Exception as e:
        print(f"[ERROR] Error: {str(e)}")
        return False


def create_run_specific_folder():

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    base_dir = os.path.abspath(".")
    results_dir = os.path.join(base_dir, "results")
    runs_dir = os.path.join(results_dir, "runs")
    run_folder = os.path.join(runs_dir, f"run_{timestamp}")

    # Create directories step by step
    try:
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(runs_dir, exist_ok=True)
        os.makedirs(run_folder, exist_ok=True)
        os.makedirs(os.path.join(run_folder, "logs"), exist_ok=True)
        os.makedirs(os.path.join(run_folder, "visualizations"), exist_ok=True)
        os.makedirs(os.path.join(run_folder, "data"), exist_ok=True)
    except OSError as e:
        print(f"Error creating directories: {e}")
        # Fallback to current directory
        run_folder = f"run_{timestamp}"
        os.makedirs(run_folder, exist_ok=True)
        os.makedirs(f"{run_folder}/logs", exist_ok=True)
        os.makedirs(f"{run_folder}/visualizations", exist_ok=True)
        os.makedirs(f"{run_folder}/data", exist_ok=True)
        print(f"Using fallback folder in current directory: {run_folder}")


    return run_folder


def get_latest_run_folder():

    runs_dir = os.path.join("results", "runs")
    if not os.path.exists(runs_dir):
        return None
    
    run_folders = [d for d in os.listdir(runs_dir) 
                   if d.startswith("run_") and os.path.isdir(os.path.join(runs_dir, d))]
    
    if not run_folders:
        return None
    
    latest_folder = sorted(run_folders, reverse=True)[0]
    return latest_folder

def detect_completed_analyses(run_folder, target_keyword='security', model_name='roberta', keyword_group=None, aggregation_mode='average'):

    completed = {}

    if keyword_group:
        keyword_part = "_".join(keyword_group)
    else:
        keyword_part = target_keyword

    if aggregation_mode == "average":
        phase3_file = os.path.join(run_folder, "data", f"poc_{keyword_part}_{model_name}_avg_embeddings.pkl")
    elif aggregation_mode == "kmeans":
        phase3_file = os.path.join(run_folder, "data", f"poc_{keyword_part}_{model_name}_clustered_data.pkl")
    else:

        phase3_file = [
            os.path.join(run_folder, "data", f"poc_{keyword_part}_{model_name}_avg_embeddings.pkl"),
            os.path.join(run_folder, "data", f"poc_{keyword_part}_{model_name}_clustered_data.pkl")
        ]


    baseline_files = {
        'phase1': os.path.join(run_folder, "data", f"poc_{keyword_part}_sentences.pkl"),
        'phase2': os.path.join(run_folder, "data", f"poc_{keyword_part}_{model_name}_raw_embeddings.pkl"),
        'phase3': phase3_file
    }

    baseline_completed = {}
    for phase, filepath in baseline_files.items():
        if isinstance(filepath, list):

            baseline_completed[phase] = any(os.path.exists(f) for f in filepath)
        else:
            baseline_completed[phase] = os.path.exists(filepath)

    if any(baseline_completed.values()):
        completed['baseline'] = baseline_completed

    if aggregation_mode == "average":
        combined_phase3_file = os.path.join(run_folder, "data", f"poc_{keyword_part}_combined_{model_name}_avg_embeddings.pkl")
    elif aggregation_mode == "kmeans":
        combined_phase3_file = os.path.join(run_folder, "data", f"poc_{keyword_part}_combined_{model_name}_clustered_data.pkl")
    else:

        combined_phase3_file = [
            os.path.join(run_folder, "data", f"poc_{keyword_part}_combined_{model_name}_avg_embeddings.pkl"),
            os.path.join(run_folder, "data", f"poc_{keyword_part}_combined_{model_name}_clustered_data.pkl")
        ]

    combined_files = {
        'phase1': os.path.join(run_folder, "data", f"poc_{keyword_part}_combined_sentences.pkl"),
        'phase2': os.path.join(run_folder, "data", f"poc_{keyword_part}_combined_{model_name}_raw_embeddings.pkl"),
        'phase3': combined_phase3_file
    }

    combined_completed = {}
    for phase, filepath in combined_files.items():
        if isinstance(filepath, list):

            combined_completed[phase] = any(os.path.exists(f) for f in filepath)
        else:
            combined_completed[phase] = os.path.exists(filepath)

    if any(combined_completed.values()):
        completed['combined'] = combined_completed


    for theme_name in THEMATIC_FILTERS.keys():

        if aggregation_mode == "average":
            theme_phase3_file = os.path.join(run_folder, "data", f"poc_{keyword_part}_{theme_name}_{model_name}_avg_embeddings.pkl")
        elif aggregation_mode == "kmeans":
            theme_phase3_file = os.path.join(run_folder, "data", f"poc_{keyword_part}_{theme_name}_{model_name}_clustered_data.pkl")
        else:

            theme_phase3_file = [
                os.path.join(run_folder, "data", f"poc_{keyword_part}_{theme_name}_{model_name}_avg_embeddings.pkl"),
                os.path.join(run_folder, "data", f"poc_{keyword_part}_{theme_name}_{model_name}_clustered_data.pkl")
            ]

        theme_files = {
            'phase1': os.path.join(run_folder, "data", f"poc_{keyword_part}_{theme_name}_sentences.pkl"),
            'phase2': os.path.join(run_folder, "data", f"poc_{keyword_part}_{theme_name}_{model_name}_raw_embeddings.pkl"),
            'phase3': theme_phase3_file
        }

        theme_completed = {}
        for phase, filepath in theme_files.items():
            if isinstance(filepath, list):

                theme_completed[phase] = any(os.path.exists(f) for f in filepath)
            else:
                theme_completed[phase] = os.path.exists(filepath)

        if any(theme_completed.values()):
            completed[theme_name] = theme_completed

    return completed

def detect_all_completed_analyses(run_folder, analysis_items, selected_themes, run_combined, run_baseline, model_name, aggregation_mode):

    all_completed = {}
    
    for analysis_item in analysis_items:
        target_keyword = analysis_item['keyword']
        keyword_group = analysis_item['keyword_group']
        display_name = analysis_item['display_name']
        

        completed_analyses = detect_completed_analyses(
            run_folder, target_keyword, model_name, keyword_group, aggregation_mode
        )
        

        if run_baseline:
            baseline_status = completed_analyses.get('baseline', {'phase1': False, 'phase2': False, 'phase3': False})
            all_completed[(display_name, 'baseline')] = baseline_status
        

        if run_combined:
            combined_status = completed_analyses.get('combined', {'phase1': False, 'phase2': False, 'phase3': False})
            all_completed[(display_name, 'combined')] = combined_status
        

        for theme_name in selected_themes:
            theme_status = completed_analyses.get(theme_name, {'phase1': False, 'phase2': False, 'phase3': False})
            all_completed[(display_name, theme_name)] = theme_status
    
    return all_completed

def check_prerequisites():

    required_scripts = ['src/poc_phase1_data_prep.py', 'src/poc_phase2_embeddings.py', 'src/poc_phase3_aggregation.py']
    missing_scripts = [script for script in required_scripts if not os.path.exists(script)]

    if missing_scripts:
        print(f"[ERROR] Missing scripts: {', '.join(missing_scripts)}")
        return False
    return True

def run_baseline_analysis(run_folder, completed_phases=None, temporal_unit='term', target_keyword='security', model_name='roberta', keyword_group=None, min_sentences=0, aggregation_mode='average', k_min=2, k_max=8, silhouette_threshold=0.25):

    if completed_phases is None:
        completed_phases = {}

    print("\n" + "=" * 80)
    print("RUNNING BASELINE ANALYSIS (No thematic filtering)")
    print("=" * 80)


    baseline_phases = [
        ("phase1", "src/poc_phase1_data_prep.py", "Baseline Phase 1: Data Preparation"),
        ("phase2", "src/poc_phase2_embeddings.py", "Baseline Phase 2: Raw Embedding Extraction"),
        ("phase3", "src/poc_phase3_aggregation.py", "Baseline Phase 3: Aggregation")
    ]

    success_count = 0
    total_phases = len(baseline_phases)
    phases_to_run = []


    for phase_id, script, description in baseline_phases:
        if completed_phases.get(phase_id, False):
            print(f"[SKIP] Skipping {description} (already completed)")
            success_count += 1
        else:
            phases_to_run.append((phase_id, script, description))


    for phase_num, (phase_id, script, description) in enumerate(phases_to_run, success_count + 1):
        if run_script_with_run_folder(script, description, phase_num, total_phases, run_folder,
                                    temporal_unit=temporal_unit, target_keyword=target_keyword, model_name=model_name, keyword_group=keyword_group, min_sentences=min_sentences, aggregation_mode=aggregation_mode, k_min=k_min, k_max=k_max, silhouette_threshold=silhouette_threshold):
            success_count += 1
        else:
            return False

    return success_count == total_phases

def run_thematic_analysis(run_folder, theme_name, theme_keywords, completed_phases=None, temporal_unit='term', target_keyword='security', model_name='roberta', min_sentences=0, aggregation_mode='average', k_min=2, k_max=8, silhouette_threshold=0.25):

    if completed_phases is None:
        completed_phases = {}

    print(f"\n" + "=" * 80)
    print(f"RUNNING THEMATIC ANALYSIS: {theme_name.upper()} CONTEXT")
    print(f"Keywords: {', '.join(theme_keywords)}")
    print("=" * 80)


    thematic_phases = [
        ("phase1", "src/poc_phase1_data_prep.py", f"{theme_name.title()} Phase 1: Filtered Data Preparation"),
        ("phase2", "src/poc_phase2_embeddings.py", f"{theme_name.title()} Phase 2: Context-Specific Raw Embeddings"),
        ("phase3", "src/poc_phase3_aggregation.py", f"{theme_name.title()} Phase 3: Context-Specific Aggregation")
    ]

    success_count = 0
    total_phases = len(thematic_phases)
    phases_to_run = []


    for phase_id, script, description in thematic_phases:
        if completed_phases.get(phase_id, False):
            print(f"[SKIP] Skipping {description} (already completed)")
            success_count += 1
        else:
            phases_to_run.append((phase_id, script, description))

    for phase_num, (phase_id, script, description) in enumerate(phases_to_run, success_count + 1):
        if run_script_with_run_folder(script, description, phase_num, total_phases,
                                    run_folder, theme_name, theme_keywords, temporal_unit, target_keyword, model_name, min_sentences=min_sentences, aggregation_mode=aggregation_mode, k_min=k_min, k_max=k_max, silhouette_threshold=silhouette_threshold):
            success_count += 1
        else:
            return False

    return success_count == total_phases

def create_comparative_analysis(run_folder, successful_themes):

    print(f"\n" + "=" * 80)
    print("CREATING COMPARATIVE ANALYSIS")
    print(f"Comparing {len(successful_themes)} contexts: {', '.join(successful_themes)}")
    print("=" * 80)

    try:

        comparative_folder = os.path.join(run_folder, "comparative_analysis")
        os.makedirs(comparative_folder, exist_ok=True)

        theme_data = {}

        for theme in successful_themes:
            try:

                if theme == "baseline":
                    stats_file = os.path.join(run_folder, "phase3_polarization_calculation_stats.txt")
                else:
                    stats_file = os.path.join(run_folder, f"phase3_polarization_calculation_stats_{theme}.txt")

                if os.path.exists(stats_file):

                    with open(stats_file, 'r') as f:
                        content = f.read()

                        if "Average distance:" in content:
                            for line in content.split('\n'):
                                if "Average distance:" in line:
                                    avg_dist = float(line.split(':')[1].strip())
                                    theme_data[theme] = avg_dist
                                    break
                        else:
                            theme_data[theme] = 0.0
            except Exception as e:
                theme_data[theme] = 0.0

        report_path = os.path.join(comparative_folder, "theme_comparison_report.txt")
        with open(report_path, 'w') as f:
            f.write("THEMATIC ANALYSIS COMPARISON REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Themes Analyzed: {len(successful_themes)}\n")
            f.write(f"Themes: {', '.join(successful_themes)}\n\n")

            f.write("POLARIZATION COMPARISON (Average Distances):\n")
            f.write("-" * 40 + "\n")


            sorted_themes = sorted(theme_data.items(), key=lambda x: x[1], reverse=True)

            for i, (theme, avg_dist) in enumerate(sorted_themes, 1):
                f.write(f"{i}. {theme.title()}: {avg_dist:.4f}\n")

            f.write("\nKEY INSIGHTS:\n")
            if sorted_themes:
                most_polarized = sorted_themes[0]
                least_polarized = sorted_themes[-1]
                f.write(f"• Most polarizing context: {most_polarized[0].title()} ({most_polarized[1]:.4f})\n")
                f.write(f"• Least polarizing context: {least_polarized[0].title()} ({least_polarized[1]:.4f})\n")

                if len(sorted_themes) > 1:
                    polarization_range = most_polarized[1] - least_polarized[1]
                    f.write(f"• Polarization range across contexts: {polarization_range:.4f}\n")

            f.write("\nCONTEXT-SPECIFIC FINDINGS:\n")
            for theme in successful_themes:
                if theme == "baseline":
                    f.write(f"• Baseline: Represents general keyword discourse without thematic filtering\n")
                else:
                    f.write(f"• {theme.title()}: Keyword discourse filtered for {theme}-related keywords\n")

        print(f"[SUCCESS] Comparative analysis completed!")
        print(f"Report saved to: {report_path}")

        return True

    except Exception as e:
        print(f"[ERROR] Comparative analysis failed: {str(e)}")
        return False

def main(resume_folder=None, temporal_unit='term', keywords=['security'], model_name='roberta', themes=['all'], keyword_groups=None, weighted_centroid=False, min_sentences=0, aggregation_mode='average', k_min=2, k_max=8, silhouette_threshold=0.25):

    start_time = datetime.now()


    if resume_folder:

        if resume_folder.lower() == "latest":
            latest_run = get_latest_run_folder()
            if not latest_run:
                print("[ERROR] No existing run folders found to resume from")
                return
            resume_folder = latest_run
            print(f"📍 Found latest run: {resume_folder}")


        if not os.path.isabs(resume_folder):

            run_folder = os.path.join("results", "runs", resume_folder)
        else:
            run_folder = resume_folder

        if not os.path.exists(run_folder):
            print(f"[ERROR] Resume folder not found: {run_folder}")
            return


        existing_logs = [f for f in os.listdir(os.path.join(run_folder, "logs")) if f.startswith("poc_pipeline_master_")]
        if existing_logs:
            timestamp = existing_logs[0].replace("poc_pipeline_master_", "").replace(".log", "")
        else:
            timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    else:

        run_folder = create_run_specific_folder()
        timestamp = start_time.strftime("%Y%m%d_%H%M%S")

    _run_pipeline_with_logging(run_folder, timestamp, start_time, resume_folder, temporal_unit, keywords, model_name, themes, keyword_groups, weighted_centroid, min_sentences, aggregation_mode, k_min, k_max, silhouette_threshold)


def _run_pipeline_with_logging(run_folder, timestamp, start_time, resume_folder, temporal_unit='term', keywords=['security'], model_name='roberta', themes=['all'], keyword_groups=None, weighted_centroid=False, min_sentences=0, aggregation_mode='average', k_min=2, k_max=8, silhouette_threshold=0.25):

    if resume_folder:
        print(f"\n[RESUME] RESUMING RUN from: {run_folder}")
        print("Analyzing completion status for all requested analyses...")
    else:
        print(f"\n[START] Created run-specific folder: {run_folder}")

    log_file = f"{run_folder}/logs/poc_pipeline_master_{timestamp}.log"



    if not check_prerequisites():
        return


    selected_themes = set()
    run_combined = False
    run_baseline = False

    for theme in themes:
        theme = theme.lower()
        if theme == 'all':
            selected_themes.update(THEMATIC_FILTERS.keys())
            run_combined = True
            run_baseline = True
        elif theme == 'combined':
            run_combined = True
        elif theme == 'baseline':
            run_baseline = True
        elif theme in THEMATIC_FILTERS:
            selected_themes.add(theme)
        else:
            print(f"[WARNING] Unknown theme '{theme}' - available themes: baseline, combined, {', '.join(sorted(THEMATIC_FILTERS.keys()))}")


    analysis_items = []


    if keyword_groups:

        for group_idx, keyword_group in enumerate(keyword_groups):
            group_name = "_".join(keyword_group)
            analysis_items.append({
                'type': 'grouped',
                'keyword': keyword_group[0],
                'keyword_group': keyword_group,
                'display_name': " + ".join(keyword_group)
            })
    else:

        for keyword in keywords:
            analysis_items.append({
                'type': 'single',
                'keyword': keyword,
                'keyword_group': None,
                'display_name': keyword
            })

    print(f"\n[CONFIG] MULTI-KEYWORD ANALYSIS CONFIGURATION")
    analysis_display = [item['display_name'] for item in analysis_items]
    print(f"Analysis items: {', '.join(analysis_display)}")
    print(f"Embedding model: {model_name}")
    print(f"Available themes: {list(THEMATIC_FILTERS.keys())}")
    print(f"Selected themes: {', '.join(sorted(selected_themes)) if selected_themes else 'None'}")
    print(f"Run combined analysis: {run_combined}")
    print(f"Run baseline analysis: {run_baseline}")
    total_contexts = len(selected_themes) + (1 if run_combined else 0) + (1 if run_baseline else 0)
    print(f"Total contexts per item: {total_contexts} (including baseline)")
    temporal_desc = "5-year parliamentary terms" if temporal_unit == 'term' else "individual years"
    print(f"Temporal resolution: {temporal_desc}")

    all_completed_status = {}
    if resume_folder:
        print(f"\n[STATUS] Analyzing completion status for all requested analyses...")
        all_completed_status = detect_all_completed_analyses(
            run_folder, analysis_items, selected_themes, run_combined, run_baseline, model_name, aggregation_mode
        )
        

        if all_completed_status:
            print(f"\n[STATUS] Detailed completion status:")
            for (display_name, context), phases in all_completed_status.items():
                phase_symbols = []
                for phase_num in [1, 2, 3]:
                    phase_key = f'phase{phase_num}'
                    symbol = '✓' if phases.get(phase_key, False) else '✗'
                    phase_symbols.append(f"Phase {phase_num} {symbol}")
                status_line = f"  {display_name} ({context}): {', '.join(phase_symbols)}"
                print(status_line)
        else:
            print("  No completed analyses found - starting fresh analysis")

    all_successful_analyses = []
    for analysis_idx, analysis_item in enumerate(analysis_items):
        target_keyword = analysis_item['keyword']
        keyword_group = analysis_item['keyword_group']
        display_name = analysis_item['display_name']
        analysis_type = analysis_item['type']

        print(f"\n{'='*80}")
        print(f"ANALYZING {analysis_type.upper()} {analysis_idx+1}/{len(analysis_items)}: '{display_name.upper()}'")
        print(f"{'='*80}")


        combined_keywords = []
        for theme_keywords in THEMATIC_FILTERS.values():
            combined_keywords.extend(theme_keywords)

        combined_keywords = list(dict.fromkeys(combined_keywords))
        print(f"Combined analysis uses {len(combined_keywords)} unique keywords from all themes")


        successful_themes = []

        if run_combined:
            combined_completed = all_completed_status.get((display_name, "combined"), {'phase1': False, 'phase2': False, 'phase3': False})
            if all(combined_completed.values()) and any(combined_completed.values()):
                print(f"\n[SKIP] Combined thematic analysis for '{display_name}' already completed - skipping")
                successful_themes.append("combined")
            else:
                print(f"\n[START] Starting COMBINED thematic analysis ({display_name} + any topic keyword)...")
                if run_thematic_analysis(run_folder, "combined", combined_keywords, combined_completed, temporal_unit, target_keyword, model_name, min_sentences, aggregation_mode, k_min, k_max, silhouette_threshold):
                    successful_themes.append("combined")
                    print(f"\n[SUCCESS] Combined thematic analysis completed successfully!")
                else:
                    print(f"\n[ERROR] Combined thematic analysis failed - continuing with remaining analyses...")
        else:
            print(f"\n[SKIP] Combined thematic analysis not selected")


        if run_baseline:
            baseline_completed = all_completed_status.get((display_name, "baseline"), {'phase1': False, 'phase2': False, 'phase3': False})
            if all(baseline_completed.values()) and any(baseline_completed.values()):
                print(f"\n[SKIP] Baseline analysis for '{display_name}' already completed - skipping")
                successful_themes.append("baseline")
            elif run_baseline_analysis(run_folder, baseline_completed, temporal_unit, target_keyword, model_name, keyword_group, min_sentences, aggregation_mode, k_min, k_max, silhouette_threshold):
                successful_themes.append("baseline")
                print(f"\n[SUCCESS] Baseline analysis completed successfully!")
            else:
                print(f"\n[ERROR] Baseline analysis failed - continuing with thematic analyses...")
        else:
            print(f"\n[SKIP] Baseline analysis not selected")

        for theme_name, theme_keywords in THEMATIC_FILTERS.items():
            if theme_name not in selected_themes:
                print(f"\n[SKIP] {theme_name.title()} analysis not selected")
                continue

            theme_completed = all_completed_status.get((display_name, theme_name), {'phase1': False, 'phase2': False, 'phase3': False})

            if all(theme_completed.values()) and any(theme_completed.values()):
                print(f"\n[SKIP] {theme_name.title()} analysis for '{display_name}' already completed - skipping")
                successful_themes.append(theme_name)
            else:
                print(f"\n[START] Starting analysis for {theme_name} context for '{display_name}'...")
                if run_thematic_analysis(run_folder, theme_name, theme_keywords, theme_completed, temporal_unit, target_keyword, model_name, min_sentences, aggregation_mode, k_min, k_max, silhouette_threshold):
                    successful_themes.append(theme_name)
                    print(f"\n[SUCCESS] {theme_name.title()} analysis completed successfully!")
                else:
                    print(f"\n[ERROR] {theme_name.title()} analysis failed - continuing with remaining themes...")


        if len(successful_themes) > 1:
            print(f"\n🔍 Creating comparative analysis for {len(successful_themes)} successful contexts...")
            create_comparative_analysis(run_folder, successful_themes)
        else:
            pass

        if successful_themes:
            all_successful_analyses.append(display_name)
        else:
            pass

    end_time = datetime.now()
    duration = end_time - start_time


    if successful_themes:

        print(f"\n[RESULTS] All data processing completed and saved to: {run_folder}")
        print(f"[RESULTS] Data files available in: {run_folder}/data/")
        print(f"[RESULTS] Comparative analysis in: {run_folder}/comparative_analysis/")
        print(f"[RESULTS] Check logs in: {run_folder}/logs/")

        print(f"\n{'='*80}")
        print(f"📊 DATA PROCESSING COMPLETE - READY FOR VISUALIZATION")
        print(f"{'='*80}")
        print(f"To generate comprehensive visualizations, run:")
        print(f"    python run_visualizations_only.py {os.path.basename(run_folder)}")
        print(f"")
        print(f"Available visualization options:")
        print(f"    --keywords {' '.join(keywords)} --aggregation-mode average")
        print(f"    --keywords {' '.join(keywords)} --aggregation-mode kmeans")
        print(f"    --list                           # List available keywords")
        print(f"    --interactive                    # Interactive keyword selection")
        print(f"{'='*80}")

        if len(successful_themes) > 1:
            print(f"🔍 {len(successful_themes)} contexts analyzed: {', '.join(successful_themes)}")
        else:
            print(f"[WARNING] Only {len(successful_themes)} context analyzed - comparative analysis limited")

    else:
        print(f"\n[ERROR] Pipeline failed - no successful analyses")
        print(f"Check the logs for detailed error information")

    # Final summary
    if successful_themes:
        print(f"\n[SUCCESS] DATA PROCESSING PIPELINE COMPLETED")
        print(f"Successfully analyzed {len(successful_themes)}/{total_contexts} contexts")
        print(f"Contexts: {', '.join(successful_themes)}")
        if len(successful_themes) > 1:
            print(f"Comparative analysis available in: {run_folder}/comparative_analysis/")
    else:
        print(f"\n[ERROR] PIPELINE FAILED - No successful analyses")




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run EU Parliament Discourse Analysis Pipeline with Thematic Filtering')
    parser.add_argument('--resume', '-r', type=str, help='Resume from existing run folder (e.g., run_20250827_191459, full path, or "latest" for most recent)')
    parser.add_argument('--list', '-l', action='store_true', help='List available run folders to resume from')
    parser.add_argument('--temporal', choices=['term', 'year'], default='term',
                       help='Temporal grouping: term (5-year periods) or year (default: term)')
    parser.add_argument('--keywords', '-k', nargs='+', default=['security'],
                       help='List of keywords to analyze (default: security). Example: --keywords security defense migration')
    parser.add_argument('--model', '-m', choices=['roberta', 'distilroberta'], default='roberta',
                       help='Embedding model to use: roberta (high accuracy) or distilroberta (fast & efficient) (default: roberta)')
    parser.add_argument('--themes', '-t', nargs='+', default=['all'],
                       help='Themes to analyze: "all" (all themes + combined), "baseline" (only baseline analysis), "combined" (only combined), or specific theme names. '
                            f'Available themes: baseline, combined, {", ".join(sorted(THEMATIC_FILTERS.keys()))}. '
                            'Example: --themes baseline immigration | --themes combined security health')
    parser.add_argument('--keyword-groups', '-g', action='append',
                       help='Define keyword groups for grouped analysis. Use multiple times for multiple groups. '
                            'Format: comma-separated keywords within each group. '
                            'Example: --keyword-groups "security,defense,military" --keyword-groups "migration,immigration,refugee"')
    parser.add_argument('--weighted-centroid', '-w', action='store_true',
                       help='Use weighted centroid calculation for absolute polarization metrics. '
                            'Each political group contributes equally to the semantic center regardless of sentence count.')
    parser.add_argument('--min-sentences', '-s', type=int, default=0,
                       help='Minimum number of sentences required for a group-timestep combination to be included in the analysis. Default: 0 (no filtering).')
    parser.add_argument('--aggregation-mode', '-a', choices=['average', 'kmeans'], default='average',
                       help='Aggregation mode for Phase 3: average (simple averaging) or kmeans (clustering). Default: average.')
    parser.add_argument('--k-min', type=int, default=2,
                       help='Minimum number of clusters to test for Dynamic K-means (only used with --aggregation-mode kmeans). Default: 2.')
    parser.add_argument('--k-max', type=int, default=8,
                       help='Maximum number of clusters to test for Dynamic K-means (only used with --aggregation-mode kmeans). Default: 8.')
    parser.add_argument('--silhouette-threshold', type=float, default=0.25,
                       help='Minimum silhouette score threshold for meaningful clusters in Dynamic K-means (only used with --aggregation-mode kmeans). Default: 0.25.')

    args = parser.parse_args()


    keyword_groups = None
    if args.keyword_groups:
        keyword_groups = []
        for group_str in args.keyword_groups:
            group = [kw.strip() for kw in group_str.split(',') if kw.strip()]
            if group:
                keyword_groups.append(group)

    if args.list:
        runs_dir = os.path.join("results", "runs")
        if os.path.exists(runs_dir):
            run_folders = [d for d in os.listdir(runs_dir) if d.startswith("run_") and os.path.isdir(os.path.join(runs_dir, d))]
            if run_folders:
                print("\nAvailable run folders:")
                for folder in sorted(run_folders, reverse=True):
                    folder_path = os.path.join(runs_dir, folder)
                    completed = detect_completed_analyses(folder_path, 'security', 'roberta', None, 'average')  # Default for listing
                    total_analyses = (len(THEMATIC_FILTERS) + 2) * 3
                    completed_phases = sum(sum(phases.values()) for phases in completed.values())
                    completed_count = completed_phases
                    print(f"  {folder} ({completed_count}/{total_analyses} analyses completed)")
            else:
                print("No run folders found in results/runs/")
        else:
            print("results/runs/ directory not found")
    elif args.resume:
        main(resume_folder=args.resume, temporal_unit=args.temporal, keywords=args.keywords, model_name=args.model, themes=args.themes, keyword_groups=keyword_groups, weighted_centroid=args.weighted_centroid, min_sentences=args.min_sentences, aggregation_mode=args.aggregation_mode, k_min=args.k_min, k_max=args.k_max, silhouette_threshold=args.silhouette_threshold)
    else:
        main(temporal_unit=args.temporal, keywords=args.keywords, model_name=args.model, themes=args.themes, keyword_groups=keyword_groups, weighted_centroid=args.weighted_centroid, min_sentences=args.min_sentences, aggregation_mode=args.aggregation_mode, k_min=args.k_min, k_max=args.k_max, silhouette_threshold=args.silhouette_threshold)