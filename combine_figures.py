import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
from PIL import Image


def create_figure(
    image_paths, 
    labels, 
    output_filename, 
    figsize, 
    label_fontsize=20,
    label_offset_x=-0.00,
    label_offset_y=-0.00):
    
    print(f"Figure creation : {output_filename}...")

    output_dir = os.path.dirname(output_filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    nrows = len(image_paths)
    ncols = len(image_paths[0])

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    
    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r, c]
            img_path = image_paths[r][c]
            if img_path is None:
                ax.axis('off')
            else:
                try:
                    if img_path.lower().endswith('.png'):
                        # For png files, use PIL to open and convert to a format matplotlib can display
                        img = Image.open(img_path)
                        ax.imshow(img)
                    else:
                        # For other image formats (like PNG), use mpimg.imread
                        img = mpimg.imread(img_path)
                        ax.imshow(img)
                except FileNotFoundError:
                    print(f"file '{img_path}' not found, displaying error message.")
                    ax.text(0.5, 0.5, f"File not found:\n{os.path.basename(img_path)}", 
                            ha='center', va='center', wrap=True, color='red')
                except Exception as e:
                    print(f"Error loading '{img_path}': {e}")
                    ax.text(0.5, 0.5, f"Error loading:\n{os.path.basename(img_path)}\n({e})", 
                            ha='center', va='center', wrap=True, color='red')
                ax.axis('off')

    plt.tight_layout(pad=1.0, h_pad=2.0, w_pad=2.0)

    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r, c]
            label = labels[r][c]
            if label:
                pos = ax.get_position()
                fig.text(
                    pos.x0 + label_offset_x, 
                    pos.y1 + label_offset_y, 
                    label, 
                    transform=fig.transFigure, 
                    fontsize=label_fontsize, 
                    fontweight='bold', 
                    va='bottom', 
                    ha='left'
                )

    plt.savefig(output_filename, dpi=600, bbox_inches='tight', format='png')
    plt.close(fig)
    print(f"  -> Figure saved to '{output_filename}'")

# FIGURE 1 (2x2 carré)
create_figure(
    image_paths=[
        ['results/global_correlation_plot.png', 'results/per_drug_correlation_grid.png'],
        ['results/r2_comparison_barplot.png', None]
    ],
    labels=[['A', 'B'], ['C', '']],
    output_filename='results/Figure1.png',
    figsize=(12, 12)
)

# FIGURE 2 (2x2)
create_figure(
    image_paths=[
        ['results/umap_embeddings_by_stop_type.png', 'results/umap_embeddings_by_rt.png'],
        ['results/umap_embeddings_by_drug.png', 'results/drug_similarity_clustermap.png']
    ],
    labels=[['A', 'B'], ['C', 'D']],
    output_filename='results/Figure2.png',
    figsize=(14, 12)
)

# FIGURE 3 (1x2)
create_figure(
    image_paths=[
        ['results/saturation_mutagenesis_heatmap_DAP_uga.png', 'results/epistasis_heatmap_DAP_uga.png']
    ],
    labels=[['A', 'B']],
    output_filename='results/Figure3.png',
    figsize=(14, 6)
)

# FIGURE 4 (1x2)
create_figure(
    image_paths=[
        ['results/best_drug_sunburst_inverted.png', 'results/drug_profile_raincloud_plot_custom.png']
    ],
    labels=[['A', 'B']],
    output_filename='results/Figure4.png',
    figsize=(16, 7)
)

# FIGURE 5 (2x2)
create_figure(
    image_paths=[
        ['results/best_drug_confusion_matrix.png', 'results/disagreement_gain_distribution_log.png'],
        ['results/high_gain_stop_type_analysis.png', 'results/high_gain_drug_switch_analysis.png']
    ],
    labels=[['A', 'B'], ['C', 'D']],
    output_filename='results/Figure5.png',
    figsize=(14, 12)
)

# FIGURE 6 (1x1)
create_figure(
    image_paths=[
        ['results/cftr_therapeutic_profiles_heatmap.png']
    ],
    labels=[['']], 
    output_filename='results/Figure6.png',
    figsize=(8, 7)
)

print("\n--- All figures have been successfully generated. ---")
