from collections import defaultdict

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.cluster import DBSCAN


def moving_average(x, y, window_size=15):
    if len(x) < window_size:
        return x, y
    y_smooth = np.convolve(y, np.ones(window_size) / window_size, mode='valid')
    x_smooth = x[window_size // 2: -(window_size // 2)] if len(x) > window_size else x[:len(y_smooth)]
    return x_smooth, y_smooth


def tsne_with_clustered_ui_centroids(original_training_embeddings, recognizable_training_images, unrecognizable_training_images, eps=0.5, min_samples=5, title=""):
    all_embeddings = []
    labels = []

    # Add original embeddings
    for emb in original_training_embeddings:
        if emb is not None:
            all_embeddings.append(emb)
            labels.append('original')

    # Add recognizable embeddings
    for _, enc, _, _ in recognizable_training_images:
        if enc is not None:
            all_embeddings.append(enc)
            labels.append('recognizable')

    # Extract unrecognizable embeddings separately
    unrec_embs = [enc for _, enc, _, _ in unrecognizable_training_images if enc is not None]

    # Cluster unrecognizable embeddings
    cluster_labels = []
    ui_centroids = []
    if len(unrec_embs) > 0:
        X = np.array(unrec_embs)
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        cluster_labels = db.labels_
        for cluster_id in set(cluster_labels):
            if cluster_id == -1:  # Noise
                continue
            cluster_points = X[cluster_labels == cluster_id]
            centroid = cluster_points.mean(axis=0)
            ui_centroids.append(centroid)

    # Add clustered unrecognizable embeddings to the plot
    for i, emb in enumerate(unrec_embs):
        label = f'unrecognizable_{cluster_labels[i]}' if cluster_labels[i] != -1 else 'unrecognizable_noise'
        all_embeddings.append(emb)
        labels.append(label)

    # Add UI centroids to the plot
    for i, centroid in enumerate(ui_centroids):
        all_embeddings.append(centroid)
        labels.append(f'ui_centroid_{i}')

    # t-SNE
    all_embeddings = np.array(all_embeddings)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(all_embeddings)

    # Plotting
    plt.figure(figsize=(10, 7))
    unique_labels = set(labels)
    colors = {
        'original': 'black',
        'recognizable': 'green',
    }

    for label in unique_labels:
        indices = [i for i, l in enumerate(labels) if l == label]
        if 'ui_centroid' in label:
            plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], label=label, marker='X', s=100, zorder=5)
        elif 'unrecognizable_noise' in label:
            plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], label='unrecognizable (noise)', color='gray', s=5, alpha=0.4)
        elif 'unrecognizable_' in label:
            cluster_id = label.split('_')[-1]
            plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], label=f'unrecognizable (cluster {cluster_id})', s=5, alpha=0.6)
        else:
            plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], label=label, color=colors.get(label, None), s=5, alpha=0.6)

    plt.title(title)
    plt.legend(markerscale=1, bbox_to_anchor=(1.05, 1), loc='upper left')  # Move legend outside
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return ui_centroids  # So you can use them for ERS or similarity filtering


def tsne_with_clustered_ui_centroids_hdbscan(original_training_embeddings, unrecognizable_training_images, min_cluster_size=10, title=""):
    all_embeddings = []
    labels = []

    # Add original embeddings
    for emb in original_training_embeddings:
        if emb is not None:
            all_embeddings.append(emb)
            labels.append('original')

    # Extract unrecognizable embeddings
    unrec_embs = [enc for _, enc, _, _ in unrecognizable_training_images if enc is not None]

    # Cluster with HDBSCAN
    cluster_labels = []
    ui_centroids = []
    cluster_id_to_color = {}

    if len(unrec_embs) > 0:
        X = np.array(unrec_embs)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        cluster_labels = clusterer.fit_predict(X)

        # Generate unique colors for clusters
        unique_cluster_ids = sorted(set(cluster_labels) - {-1})
        color_map = cm.get_cmap('tab20', len(unique_cluster_ids))
        for i, cluster_id in enumerate(unique_cluster_ids):
            cluster_id_to_color[cluster_id] = color_map(i)

        for cluster_id in unique_cluster_ids:
            cluster_points = X[cluster_labels == cluster_id]
            centroid = cluster_points.mean(axis=0)
            ui_centroids.append(centroid)

    # Add clustered unrecognizable embeddings to plot
    for i, emb in enumerate(unrec_embs):
        label = f'u{cluster_labels[i]}' if cluster_labels[i] != -1 else 'u_n'
        all_embeddings.append(emb)
        labels.append(label)

    # Add UI centroids
    for i, centroid in enumerate(ui_centroids):
        all_embeddings.append(centroid)
        labels.append(f'ui_c_{i}')

    # Apply t-SNE
    all_embeddings = np.array(all_embeddings)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(all_embeddings)

    # Prepare color mapping
    colors = {'original': 'black', 'u_n': 'gray'}
    for i, cluster_id in enumerate(unique_cluster_ids):
        colors[f'u{cluster_id}'] = cluster_id_to_color[cluster_id]
        colors[f'ui_c_{i}'] = cluster_id_to_color[cluster_id]

    # Plot
    plt.figure(figsize=(12, 8))  # Increase width if needed
    scatter_handles = {}
    for label in set(labels):
        indices = [i for i, l in enumerate(labels) if l == label]
        if not indices:
            continue

        color = colors.get(label, 'blue')
        display_label = label  # What will appear in legend

        if label.startswith('ui_c_'):
            scatter = plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1],
                                  color=color, marker='X', s=100, zorder=5)
        elif label == 'u_n':
            scatter = plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1],
                                  color=color, s=5, alpha=0.4)
            display_label = 'noise'
        elif label.startswith('u'):
            scatter = plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1],
                                  color=color, s=5, alpha=0.6)
        else:  # 'original'
            scatter = plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1],
                                  color=color, s=5, alpha=0.6)

        if display_label not in scatter_handles:
            scatter_handles[display_label] = scatter

    plt.subplots_adjust(right=0.75)  # Make room for legend on the right
    plt.title(title)
    plt.grid(True)
    plt.legend(scatter_handles.values(), scatter_handles.keys(),
               markerscale=1.5, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.show()

    return ui_centroids


def tsne_training_recognizable_unrecognizable_original_images_having_different_colors(original_training_embeddings, recognizable_training_images, unrecognizable_training_images, ui_centroid=None, title=""):
    all_embeddings = []
    labels = []
    # Original embeddings (label: 'original')
    for original_emb in original_training_embeddings:
        if original_emb is not None:
            all_embeddings.append(original_emb)
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

    if len(all_embeddings) < 2:
        print("[t-SNE] Not enough data points to visualize.")
        return

    safe_perplexity = min(30, max(2, len(all_embeddings) - 1))

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=safe_perplexity)
    embeddings_2d = tsne.fit_transform(all_embeddings)
    # Plot
    plt.figure(figsize=(10, 7))
    colors = {'original': 'black', 'recognizable': 'green', 'unrecognizable': 'red', 'ui_centroid': 'purple'}
    for label in set(labels) - {'original', 'ui_centroid'}:
        indexes = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(embeddings_2d[indexes, 0], embeddings_2d[indexes, 1], label=label, s=5, alpha=0.6, c=colors[label])
    # Plot 'original' group on top
    original_indexes = [i for i, l in enumerate(labels) if l == 'original']
    plt.scatter(embeddings_2d[original_indexes, 0], embeddings_2d[original_indexes, 1], label='original', s=5, alpha=0.8, c=colors['original'], zorder=4)
    # Highlight the UI centroid
    if ui_centroid is not None:
        centroid_idx = len(all_embeddings) - 1  # The last point is the centroid
        plt.scatter(embeddings_2d[centroid_idx, 0], embeddings_2d[centroid_idx, 1], color='purple', s=100, marker='X', label='UI Centroid', zorder=5)
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
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    all_test_embeddings = np.array(all_test_embeddings)
    embeddings_2d = tsne.fit_transform(all_test_embeddings)
    # Plot
    plt.figure(figsize=(12, 8))
    unique_groups = list(set(labels))
    for i, group_id in enumerate(unique_groups):
        if group_id not in {'ui_centroid', 'original'}:
            indexes = [j for j, l in enumerate(labels) if l == group_id]
            plt.scatter(embeddings_2d[indexes, 0], embeddings_2d[indexes, 1], color=color_map(i % 20), s=5, alpha=0.6, label=None)
    # Plot 'original' group on top
    original_indexes = [j for j, l in enumerate(labels) if l == 'original']
    if original_indexes:
        plt.scatter(embeddings_2d[original_indexes, 0], embeddings_2d[original_indexes, 1], color='black', s=5, alpha=0.8, label='original', zorder=4)
    # Highlight the UI centroid
    if ui_centroid is not None:
        centroid_indexes = [j for j, l in enumerate(labels) if l == 'ui_centroid']
        plt.scatter(embeddings_2d[centroid_indexes, 0], embeddings_2d[centroid_indexes, 1], color='purple', s=100, marker='X', label='UI Centroid', zorder=5)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_binned_bar_chart(data, title, bin_width=0.01, max_value=2.0):
    bins = np.arange(0, max_value + bin_width, bin_width)
    bin_labels = [f"{b:.1f}-{b + bin_width:.1f}" for b in bins[:-1]]

    # Count frequencies in each bin for both tuple positions
    first_counts = defaultdict(int)
    second_counts = defaultdict(int)

    for val1, val2, _ in data:
        for val, counts in zip((val1, val2), (first_counts, second_counts)):
            bin_index = int(val // bin_width)
            if bin_index < len(bins) - 1:
                counts[bin_index] += 1

    first_heights = [first_counts[i] for i in range(len(bins) - 1)]
    second_heights = [second_counts[i] for i in range(len(bins) - 1)]

    x = np.arange(len(bin_labels))
    width = 0.4

    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, first_heights, width=width, label='ERS', color='blue')
    plt.bar(x + width / 2, second_heights, width=width, label='Similarity', color='orange')

    plt.xticks(x[::5], bin_labels[::5], rotation=45)
    plt.xlabel("Value Ranges (0.01 intervals)")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.show()


def plot_ers_similarity_scatter(combined_data, title="ERS vs Similarity: Match (S) vs Mismatch (F)", ers_thresh=None, sim_thresh=None):
    ers_S = [ers for ers, sim, label in combined_data if label == 'S']
    sim_S = [sim for ers, sim, label in combined_data if label == 'S']

    ers_F = [ers for ers, sim, label in combined_data if label == 'F']
    sim_F = [sim for ers, sim, label in combined_data if label == 'F']

    plt.figure(figsize=(10, 6))
    plt.scatter(ers_S, sim_S, color='green', label='Match (S)', s=5, alpha=0.6)
    plt.scatter(ers_F, sim_F, color='red', label='Mismatch (F)', s=5, alpha=0.6)

    # Threshold lines (optional)
    if ers_thresh is not None:
        plt.axvline(x=ers_thresh, color='blue', linestyle='--', label=f'ERS Threshold ({ers_thresh:.2f})')
    if sim_thresh is not None:
        plt.axhline(y=sim_thresh, color='orange', linestyle='--', label=f'Similarity Threshold ({sim_thresh:.2f})')

    plt.ylim(0, 1)
    plt.xlim(0.5, 1.25)
    plt.xlabel("ERS (Euclidean distance from UI centroid)")
    plt.ylabel("Cosine Similarity")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_ers_similarity_binned(combined_data, title="ERS vs Similarity (Binned Averages)", ers_thresh=None, sim_thresh=None, bin_width=0.01):
    # Separate data by label
    ers_S = np.array([ers for ers, sim, label in combined_data if label == 'S'])
    sim_S = np.array([sim for ers, sim, label in combined_data if label == 'S'])

    ers_F = np.array([ers for ers, sim, label in combined_data if label == 'F'])
    sim_F = np.array([sim for ers, sim, label in combined_data if label == 'F'])

    def binned_avg(x, y, bin_width):
        if len(x) == 0 or len(y) == 0:
            return [], []
        bins = np.arange(x.min(), x.max() + bin_width, bin_width)
        bin_centers = []
        bin_means = []
        for i in range(len(bins) - 1):
            mask = (x >= bins[i]) & (x < bins[i + 1])
            if np.any(mask):
                bin_centers.append((bins[i] + bins[i + 1]) / 2)
                bin_means.append(np.mean(y[mask]))
        return bin_centers, bin_means

    # Apply binning (with safety checks)
    x_S, y_S = binned_avg(ers_S, sim_S, bin_width)
    x_F, y_F = binned_avg(ers_F, sim_F, bin_width)

    plt.figure(figsize=(10, 6))

    """if len(ers_S) > 0:
        plt.scatter(ers_S, sim_S, color='green', s=5, alpha=0.3, label='Match (S) Raw')
    if len(ers_F) > 0:
        plt.scatter(ers_F, sim_F, color='red', s=5, alpha=0.3, label='Mismatch (F) Raw')"""
    if x_S:
        plt.plot(x_S, y_S, color='green', linewidth=2, label='Match (S) Avg')
    if x_F:
        plt.plot(x_F, y_F, color='red', linewidth=2, label='Mismatch (F) Avg')

    if ers_thresh is not None:
        plt.axvline(x=ers_thresh, color='blue', linestyle='--', label=f'ERS Threshold ({ers_thresh:.2f})')
    if sim_thresh is not None:
        plt.axhline(y=sim_thresh, color='orange', linestyle='--', label=f'Similarity Threshold ({sim_thresh:.2f})')

    plt.xlim(0.5, 1.25)
    plt.ylim(0, 1)
    plt.xlabel("ERS (Euclidean distance from UI centroid)")
    plt.ylabel("Cosine Similarity")
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


def  plot_roc_curve(data, similarity_title="ROC Curve - Similarity", ers_title="ROC Curve - ERS"):
    # Assume your data is like this:
    # combined_data = [(ers, sim, 'S' or 'F')]
    labels = [1 if label == 'S' else 0 for _, _, label in data]
    similarity_scores = [sim for _, sim, _ in data]
    ers_scores = [ers for ers, _, _ in data]
    fpr, tpr, thresholds = roc_curve(labels, similarity_scores)
    roc_auc = auc(fpr, tpr)

    # Find the best threshold (Youden’s J statistic)
    j_scores = tpr - fpr
    best_idx = j_scores.argmax()
    best_sim_thresh = thresholds[best_idx]

    print(f"Best Similarity Threshold: {best_sim_thresh:.4f}")
    inverted_ers_scores = [-x for x in ers_scores]
    fpr_ers, tpr_ers, thresholds_ers = roc_curve(labels, inverted_ers_scores)
    roc_auc_ers = auc(fpr_ers, tpr_ers)

    # Best ERS threshold
    j_scores_ers = tpr_ers - fpr_ers
    best_idx_ers = j_scores_ers.argmax()
    best_ers_thresh = -thresholds_ers[best_idx_ers]  # invert back

    print(f"Best ERS Threshold: {best_ers_thresh:.4f}")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f'Similarity ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(similarity_title)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(fpr_ers, tpr_ers, label=f'ERS ROC (AUC = {roc_auc_ers:.2f})', color='orange')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(ers_title)
    plt.legend()

    plt.tight_layout()
    plt.show()
    precision, recall, pr_thresholds = precision_recall_curve(labels, similarity_scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
    best_pr_idx = f1_scores.argmax()
    best_pr_thresh = pr_thresholds[best_pr_idx]

    print(f"Best PR Similarity Threshold (F1): {best_pr_thresh:.4f}")
