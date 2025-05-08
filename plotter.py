import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from sklearn.manifold import TSNE

from utility import get_encoding_from_image


def tsne_training_recognizable_unrecognizable_original_images_having_different_colors(original_training_images, recognizable_training_images, unrecognizable_training_images, ui_centroid=None, title=""):
    all_embeddings = []
    labels = []
    # Original embeddings (label: 'original')
    for original_img in original_training_images:
        enc = get_encoding_from_image(original_img)
        if enc is not None:
            all_embeddings.append(enc)
            labels.append('original')
    # Recognizable embeddings (label: 'recognizable')
    for _, enc, _, _ in recognizable_training_images:
        if enc is not None:
            all_embeddings.append(enc)
            labels.append('recognizable')
    # Unrecognizable embeddings (label: 'unrecognizable')
    for _, enc, _, _ in unrecognizable_training_images:
        if enc is not None:
            all_embeddings.append(enc)
            labels.append('unrecognizable')
    # Add the centroid embedding (if exists) to the embeddings list
    if ui_centroid is not None:
        all_embeddings.append(ui_centroid)
        labels.append('ui_centroid')
    # Convert to NumPy array
    all_embeddings = np.array(all_embeddings)
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(all_embeddings)
    # Plot
    plt.figure(figsize=(10, 7))
    colors = {'original': 'black', 'recognizable': 'green', 'unrecognizable': 'red', 'ui_centroid': 'purple'}
    for label in set(labels) - {'original', 'ui_centroid'}:
        indexes = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(embeddings_2d[indexes, 0], embeddings_2d[indexes, 1], label=label, alpha=0.6, c=colors[label])
    # Plot 'original' group on top
    original_indexes = [i for i, l in enumerate(labels) if l == 'original']
    plt.scatter(embeddings_2d[original_indexes, 0], embeddings_2d[original_indexes, 1], label='original', alpha=0.8, c=colors['original'], zorder=4)
    # Highlight the UI centroid
    if ui_centroid is not None:
        centroid_idx = len(all_embeddings) - 1  # The last point is the centroid
        plt.scatter(embeddings_2d[centroid_idx, 0], embeddings_2d[centroid_idx, 1], color='purple', s=250, marker='X', label='UI Centroid', zorder=5)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def tsne_test_images_originated_from_same_original_image_having_same_colors(all_test_embeddings, labels, ui_centroid=None, title=""):
    color_map = cm.get_cmap('tab20', 20)
    # Add UI centroid if needed
    if ui_centroid is not None:
        all_test_embeddings.append(ui_centroid)
        labels.append('ui_centroid')
    all_test_embeddings = np.array(all_test_embeddings)
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(all_test_embeddings)
    # Plot
    plt.figure(figsize=(12, 8))
    unique_groups = list(set(labels))
    for i, group_id in enumerate(unique_groups):
        if group_id not in {'ui_centroid', 'original'}:
            indexes = [j for j, l in enumerate(labels) if l == group_id]
            plt.scatter(embeddings_2d[indexes, 0], embeddings_2d[indexes, 1], color=color_map(i % 20), alpha=0.6, label=None)
    # Plot 'original' group on top
    original_indexes = [j for j, l in enumerate(labels) if l == 'original']
    if original_indexes:
        plt.scatter(embeddings_2d[original_indexes, 0], embeddings_2d[original_indexes, 1], color='black', alpha=0.8, label='original', zorder=4)
    # Highlight the UI centroid
    if ui_centroid is not None:
        centroid_indexes = [j for j, l in enumerate(labels) if l == 'ui_centroid']
        plt.scatter(embeddings_2d[centroid_indexes, 0], embeddings_2d[centroid_indexes, 1], color='purple', s=250, marker='X', label='UI Centroid', zorder=5)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
