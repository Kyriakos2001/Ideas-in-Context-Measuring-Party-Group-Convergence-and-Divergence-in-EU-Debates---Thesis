
"""
Batch process Phase 3 aggregations for all Phase 2 outputs in a run folder. very usefull for speeding up phase 3 on multicore machines (like colab tpu's - 96 core).
Automatically detects what needs processing and skips existing outputs.
Shows preview and asks for approval before processing.

Usage:
    python batch_process_phase3.py <run_folder> [options]
Examples:
    # Process everything with default settings
    python batch_process_phase3.py results/runs/...

"""

import os
import sys
import pickle
import argparse
import re
from datetime import datetime
from collections import defaultdict

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from poc_phase3_aggregation import aggregate_average, aggregate_kmeans, save_aggregated_embeddings
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure you're running this script from the project root directory.")
    sys.exit(1)

def parse_phase2_filename(filename):

    name = filename.replace('poc_', '').replace('_raw_embeddings.pkl', '')

    parts = name.split('_')
    model = parts[-1]

    keyword_context = '_'.join(parts[:-1])

    known_themes = [
        'digital', 'health', 'immigration', 'war', 'economy', 'environment',
        'energy', 'education', 'security', 'defence', 'enlargement', 'trade',
        'transportation', 'integration'
    ]

    if 'combined' in keyword_context:
        context = 'combined'
        keyword = keyword_context.replace('_combined', '')
    else:
        context = 'baseline'
        keyword = keyword_context

        for theme in known_themes:
            if f'_{theme}' in f'_{keyword_context}_':
                parts = keyword_context.split('_')
                if theme in parts:
                    theme_idx = parts.index(theme)
                    if theme_idx > 0:
                        context = theme
                        keyword = '_'.join(parts[:theme_idx] + parts[theme_idx+1:])
                        break

    return {
        'keyword': keyword,
        'context': context,
        'model': model,
        'keyword_context': keyword_context
    }

def find_phase2_outputs(data_dir):
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        return []

    phase2_files = []

    for filename in os.listdir(data_dir):
        if filename.startswith('poc_') and filename.endswith('_raw_embeddings.pkl'):
            try:
                parsed = parse_phase2_filename(filename)
                parsed['filename'] = filename
                phase2_files.append(parsed)
            except Exception as e:
                print(f"Warning: Could not parse filename {filename}: {e}")

    return sorted(phase2_files, key=lambda x: (x['keyword'], x['context'], x['model']))

def check_phase3_exists(data_dir, keyword_context, model, mode):
    if mode == 'average':
        filename = f'poc_{keyword_context}_{model}_avg_embeddings.pkl'
    elif mode == 'kmeans':
        filename = f'poc_{keyword_context}_{model}_clustered_data.pkl'
    else:
        return False

    return os.path.exists(os.path.join(data_dir, filename))

def get_phase3_filename(keyword_context, model, mode):
    if mode == 'average':
        return f'poc_{keyword_context}_{model}_avg_embeddings.pkl'
    elif mode == 'kmeans':
        return f'poc_{keyword_context}_{model}_clustered_data.pkl'
    else:
        return None

def display_processing_plan(to_process, data_dir, force_mode=False):
    print("\n" + "="*120)
    print("PROCESSING PLAN - FILES TO BE GENERATED")
    print("="*120)

    file_count = 0
    new_count = 0
    overwrite_count = 0

    print(f"{'#':<3} {'Keyword':<12} {'Context':<12} {'Model':<12} {'Mode':<7} {'Status':<9} {'Output File'}")
    print("-" * 120)

    for item in to_process:
        analysis = item['analysis']
        modes = item['modes']

        for mode in modes:
            file_count += 1
            filename = get_phase3_filename(
                analysis['keyword_context'],
                analysis['model'],
                mode
            )

            file_exists = os.path.exists(os.path.join(data_dir, filename))

            if force_mode and file_exists:
                status = "OVERWRITE"
                overwrite_count += 1
            else:
                status = "NEW"
                new_count += 1

            print(f"{file_count:<3} {analysis['keyword']:<12} {analysis['context']:<12} {analysis['model']:<12} {mode:<7} {status:<9} {filename}")

    print("-" * 120)

    print(f"\nSUMMARY:")
    print(f" Total files to generate: {file_count}")
    print(f" Unique keywords: {len(set(item['analysis']['keyword'] for item in to_process))}")
    print(f"New files: {new_count}")
    if overwrite_count > 0:
        print(f"  • Files to overwrite: {overwrite_count}")

    return file_count

def ask_for_confirmation(auto_yes=False):
    if auto_yes:
        print("\nAuto-approval enabled (--yes flag)")
        return True

    print("\n" + "="*120)
    print("CONFIRMATION REQUIRED")
    print("="*120)

    while True:
        response = input("\n  Proceed with generating these files? (yes/no/y/n): ").strip().lower()

        if response in ['yes', 'y']:
            print("\nProcessing approved. Starting generation...")
            return True
        elif response in ['no', 'n']:
            print("\nProcessing cancelled by user.")
            return False
        else:
            print("Please enter 'yes' or 'no' (or 'y' or 'n')")

def load_raw_embeddings_direct(filepath):
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        raise Exception(f"Failed to load raw embeddings from {filepath}: {str(e)}")

def process_single_analysis(run_folder, analysis_info, modes_to_process, k_min, k_max, silhouette_threshold):
    data_dir = os.path.join(run_folder, 'data')

    keyword = analysis_info['keyword']
    context = analysis_info['context']
    model = analysis_info['model']
    keyword_context = analysis_info['keyword_context']

    raw_embeddings_path = os.path.join(data_dir, analysis_info['filename'])
    raw_embeddings = load_raw_embeddings_direct(raw_embeddings_path)

    results = []

    if 'average' in modes_to_process:
        aggregated_data = aggregate_average(raw_embeddings)

        theme_name = None if context == 'baseline' else context

        output_file = save_aggregated_embeddings(
            aggregated_data, run_folder, 'average',
            keyword, theme_name, model
        )
        results.append(('average', output_file))

    if 'kmeans' in modes_to_process:
        aggregated_data = aggregate_kmeans(raw_embeddings, k_min, k_max, silhouette_threshold)

        theme_name = None if context == 'baseline' else context

        output_file = save_aggregated_embeddings(
            aggregated_data, run_folder, 'kmeans',
            keyword, theme_name, model
        )
        results.append(('kmeans', output_file))

    return results

def main():
    parser = argparse.ArgumentParser(
        description='Batch process Phase 3 aggregations for all Phase 2 outputs',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('run_folder', help='Path to the run folder')
    parser.add_argument('--mode', choices=['both', 'average', 'kmeans'], default='both',
                       help='Which aggregation modes to process (default: both)')
    parser.add_argument('--force', action='store_true',
                       help='Force reprocessing even if outputs exist')
    parser.add_argument('--k-min', type=int, default=2,
                       help='Minimum k for Dynamic K-means (default: 2)')
    parser.add_argument('--k-max', type=int, default=4,
                       help='Maximum k for Dynamic K-means (default: 4)')
    parser.add_argument('--silhouette-threshold', type=float, default=0.25,
                       help='Silhouette threshold for Dynamic K-means (default: 0.25)')
    parser.add_argument('--yes', '-y', action='store_true',
                       help='Auto-approve without confirmation prompt')

    args = parser.parse_args()

    if not os.path.exists(args.run_folder):
        print(f"Error: Run folder not found: {args.run_folder}")
        sys.exit(1)

    log_dir = os.path.join(args.run_folder, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'batch_phase3_{timestamp}.log')

    if args.mode in ['both', 'kmeans']:
        pass
    data_dir = os.path.join(args.run_folder, 'data')
    phase2_outputs = find_phase2_outputs(data_dir)

    print(f"\nSCANNING: {data_dir}")
    print(f"Found {len(phase2_outputs)} Phase 2 outputs")

    if not phase2_outputs:
        print("No Phase 2 outputs found. Nothing to process.")
        return

    to_process = []
    skip_count = defaultdict(int)

    for analysis in phase2_outputs:
        keyword_context = analysis['keyword_context']
        model = analysis['model']
        modes_needed = []


        if args.mode in ['both', 'average']:
            if args.force or not check_phase3_exists(data_dir, keyword_context, model, 'average'):
                modes_needed.append('average')
            else:
                skip_count['average'] += 1

        if args.mode in ['both', 'kmeans']:
            if args.force or not check_phase3_exists(data_dir, keyword_context, model, 'kmeans'):
                modes_needed.append('kmeans')
            else:
                skip_count['kmeans'] += 1

        if modes_needed:
            to_process.append({
                'analysis': analysis,
                'modes': modes_needed
            })


    if skip_count['average'] > 0 or skip_count['kmeans'] > 0:
        print(f"\nEXISTING FILES (will be skipped unless --force):")
        if skip_count['average'] > 0:
            print(f"  • {skip_count['average']} average embeddings already exist")
        if skip_count['kmeans'] > 0:
            print(f"  • {skip_count['kmeans']} k-means clusterings already exist")

    if not to_process:
        print("\nAll Phase 3 outputs already exist. Nothing to process.")
        print("   Use --force to regenerate existing files.")
        return


    file_count = display_processing_plan(to_process, data_dir, args.force)


    if not ask_for_confirmation(args.yes):
        return


    print(f"\nPROCESSING {file_count} files...")
    print("-" * 120)

    success_count = 0
    error_count = 0

    for i, item in enumerate(to_process, 1):
        analysis = item['analysis']
        modes = item['modes']

        print(f"\n[{i}/{len(to_process)}] {analysis['keyword']} ({analysis['context']}) - {analysis['model']}")
        print(f"     Modes: {', '.join(modes)}")

        try:
            results = process_single_analysis(
                args.run_folder, analysis, modes,
                args.k_min, args.k_max, args.silhouette_threshold
            )

            for mode, output_file in results:
                print(f"     ✓ {mode}: {os.path.basename(output_file)}")

            success_count += 1

        except Exception as e:
            print(f"     ✗ Error: {str(e)}")
            error_count += 1


    print("\n" + "="*120)
    print("PROCESSING COMPLETE")
    print("="*120)
    print(f"✓ Successful: {success_count} analyses")
    if error_count > 0:
        print(f"✗ Errors: {error_count} analyses")

    total_files = sum(len(item['modes']) for item in to_process[:success_count])
    print(f"Files created: {total_files}")
    print(f"Output directory: {data_dir}")



if __name__ == "__main__":
    main()