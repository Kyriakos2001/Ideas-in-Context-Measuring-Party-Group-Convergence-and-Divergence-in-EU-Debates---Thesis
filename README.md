# EU Parliament Discourse Analysis


## Quick Start

1. **Install dependencies:**
   ```bash
   python tools/install_poc_requirements.py
   ```

2. **Run your first analysis:**
   ```bash
   python run_poc_pipeline.py --keywords security
   ```

3. **View results:** Check the `results/runs/` folder for visualizations and analysis

## What It Does

- Analyzes how different political groups use language around keywords
- Measures semantic polarization using RoBERTa/DistilRoBERTa embeddings
- Creates visualizations showing political discourse patterns over time
- Supports both averaging and clustering analysis methods

## Basic Usage

### Analyze a keyword:
```bash
python run_poc_pipeline.py --keywords migration
```

### Analyze multiple keywords:
```bash
python run_poc_pipeline.py --keywords security defense migration
```

### Generate only visualizations (after running analysis):
```bash
python run_visualizations_only.py latest --keywords security
```

### Resume interrupted analysis / useful for adding keywords:
```bash
python run_poc_pipeline.py --resume latest
```

## Advanced Options

### Pipeline Arguments (`run_poc_pipeline.py`)

**Model Selection:**
```bash
# Use RoBERTa (high accuracy, slower)
python run_poc_pipeline.py --keywords security --model roberta

# Use DistilRoBERTa (faster, slightly lower accuracy ~96%)
python run_poc_pipeline.py --keywords security --model distilroberta
```

**Temporal Grouping:**
```bash
# Group by 5-year parliamentary terms (default)
python run_poc_pipeline.py --keywords security --temporal term

# Group by individual years
python run_poc_pipeline.py --keywords security --temporal year
```

**Aggregation Methods:**
```bash
# Simple averaging (default)
python run_poc_pipeline.py --keywords security --aggregation-mode average

# K-means clustering analysis
python run_poc_pipeline.py --keywords security --aggregation-mode kmeans
```

**Keyword Groups (analyze related keywords together):**
```bash
python run_poc_pipeline.py --keyword-groups "security,defense,military" --keyword-groups "migration,immigration,refugee"
```

**Thematic Filtering:**
```bash
# Analyze specific themes
python run_poc_pipeline.py --themes security health immigration

# All themes + combined analysis
python run_poc_pipeline.py --themes all

# Only baseline (no thematic filtering)
python run_poc_pipeline.py --themes baseline
```

**Quality Filters:**
```bash
# Minimum sentences per group-term combination
python run_poc_pipeline.py --keywords security --min-sentences 10

# Weighted polarization calculation
python run_poc_pipeline.py --keywords security --weighted-centroid
```

**K-means Parameters:**
```bash
python run_poc_pipeline.py --keywords security --aggregation-mode kmeans --k-min 3 --k-max 12 --silhouette-threshold 0.3
```

**Resume & List Options:**
```bash
# Resume from specific run
python run_poc_pipeline.py --resume run_20250914_133748

# List available runs
python run_poc_pipeline.py --list
```

### Visualization Options (`run_visualizations_only.py`)

**Keyword Selection:**
```bash
# Specific keywords
python run_visualizations_only.py latest --keywords security migration

# Interactive selection
python run_visualizations_only.py latest --interactive

# List available keywords
python run_visualizations_only.py latest --list
```

**Analysis Options:**
```bash
# Weighted polarization
python run_visualizations_only.py latest --keywords security --weighted

# Force specific aggregation mode
python run_visualizations_only.py latest --keywords security --aggregation-mode kmeans

# Combine multiple keywords in comparison
python run_visualizations_only.py latest --keywords security migration --combine
```

**Advanced Visualization:**
```bash
# Custom party pairs for analysis
python run_visualizations_only.py latest --keywords security --party-pairs "EPP vs S&D" "ALDE vs ECR"

# Force regeneration of theme analysis
python run_visualizations_only.py latest --keywords security --force-new-themes
```

## Available Themes

The system supports 14 predefined themes for contextual analysis:

- **security**
- **health**
- **war** 
- **enlargement**
- **defence**
- **environment** 
- **immigration** 
- **integration**
- **economy**
- **energy** 
- **digital** 
- **trade** 
- **transportation** 

## Output

Results are saved in timestamped folders under `results/runs/`:
- **Visualizations:** Charts and graphs (PNG files)
- **Data:** Processed embeddings and analysis (pickle files)

## Installation Options


**Full pipeline (required for analysis):**
```bash
python tools/install_poc_requirements.py
```

## Documentation

- **Available themes and parameters:** Run with `--help` flag

## Requirements

- EUPDCorp_1999-2024_v1.RDS in `data/` folder
- Optional: CUDA-compatible GPU for faster processing (strongly recommended)
- Optional: sentence_df_cache.pkl, stem_to_lemma_map.pkl (strongly recommended)

## Required Project Structure

```
EU_Discourse_Project/
├── run_poc_pipeline.py         # Main data processing entry point
├── run_visualizations_only.py  # Visualization generation entry point
├── src/                        # Core logic modules
│   ├── poc_phase1_data_prep.py
│   ├── poc_phase2_embeddings.py
│   ├── poc_phase3_aggregation.py
│   ├── analysis_tools.py
│   ├── visualization_suite.py
│   ├── clustering_themes.py
│   └── utils/
│       └── plotting_utils.py
├── data/
│   └── EUPDCorp_1999-2024_v1.RDS
├── sentence_df_cache.pkl
└── stem_to_lemma_map.pkl
    
```

## Closing Notes

- To run the pipeline and visualization scripts efficiently, its highly recommended you run them on a powerful GPU. A good way to do that is running them on google colab using the provided notebook. For the notebook to work, you will have to copy the folder that was shared with you to your own drive (root), including all dataset & pkl files. Pkl files are technically optional but will require significant time to be generated, even on A100 GPU / multiprocessed on 96 core machine.