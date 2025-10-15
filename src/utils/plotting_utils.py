import os
import matplotlib.pyplot as plt

PARTY_COLORS = {
    'ALDERE': '#FFFF00',
    'ECR': '#00008B',
    'EPP': '#87CEEB',
    'GEFA': '#00FF00',
    'GUENGL': '#8B0000',
    'IND': '#808080',
    'NAT': '#800080',
    'SOCPESPASD': '#FF6B6B',
}
DEFAULT_COLOR = '#000000'

def filter_temporal_data(embeddings, temporal_unit, max_year=2024):
    if temporal_unit == 'year':
        return {(t, g): e for (t, g), e in embeddings.items() if t < max_year}
    return embeddings

def filter_temporal_dataframe(df, temporal_unit, max_year=2024):
    if temporal_unit == 'year' and 'term' in df.columns:
        return df[df['term'] < max_year]
    return df

def generate_colors_for_groups(groups):
    color_map = {}
    for group in groups:
        color_map[group] = PARTY_COLORS.get(group, DEFAULT_COLOR)

    return color_map

def generate_colormap_names_for_groups(groups):
    party_colormaps = {
        'ALDERE': 'YlOrBr',
        'ECR': 'Blues',
        'EPP': 'Blues',
        'GEFA': 'Greens',
        'GUENGL': 'Reds',
        'IND': 'Greys',
        'NAT': 'Purples',
        'SOCPESPASD': 'Reds',
    }

    default_colormap = 'Greys'

    color_map = {}
    for group in groups:
        color_map[group] = party_colormaps.get(group, default_colormap)

    return color_map

def setup_plot_aesthetics(ax, title, xlabel, ylabel, grid=True, ylim_bottom=None):
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    if grid:
        ax.grid(True, alpha=0.3)

    if ylim_bottom is not None:
        ax.set_ylim(bottom=ylim_bottom)

def save_figure(fig, output_path, dpi=300, bbox_inches='tight', show=True):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches)

    if show:
        plt.show()

