import os
import umap
import umap.plot
import numpy as np
import matplotlib.pyplot as plt
from ldm.evaluation.dataset_features import location, img_haralick, expand_feature, get_feature_values
from ldm.evaluation.dataset_utils import get_common_features
from scipy.special import kl_div
import seaborn as sns
import pandas as pd
import math
import warnings
from sklearn.preprocessing import StandardScaler


def plot_haralick_umap(dataset, dataset_name, locations, logdir, sample_size=100):
    """Plot UMAP embedding of dataset features."""
    # plot u-map of haralick features with localization as color
    # plot only the ones that are common between datasets
    # if proteins multilocalize, they can be plotted for either or both locations
    # if there aren't enough samples for a location they are sampled with replacement for simplicity
    # this is going to over sampling multilocalizing proteins
    # but we're trying to sample over a given protein, in which case it may be fine
    # subsample locations to have a balanced, manageable UMAP dataset
    if img_haralick.name not in dataset.columns:
        raise ValueError(f"Need haralick features in dataset for UMAP.")

    location_samples = []
    for loc in locations:
        try:
            location_samples.extend(
                dataset.loc[
                    lambda df: df[expand_feature(location, loc)], 
                    [location.name, img_haralick.name]
                ].sample(sample_size).values
            )
        except ValueError:
            print(f"Skipping {loc} because there aren't enough samples for it.")

    if len(location_samples) == 0:
        print("No samples for any location. Skipping UMAP.")
        return

    # convert the multilocalizing proteins to their own class
    loc_hara_list = np.stack(location_samples)
    for i, loc in enumerate(loc_hara_list[:, 0]):
        if len(loc) > 1:
            loc_hara_list[i, 0] = "multilocalizing"
        else:
            loc_hara_list[i, 0] = loc[0]

    # plot the UMAP
    scaled_loc_hara = StandardScaler().fit_transform(np.stack(loc_hara_list[:, 1]))
    transform = umap.UMAP().fit(np.stack(scaled_loc_hara)) # haralick features
    plt.clf()
    plot = umap.plot.points(transform, labels=loc_hara_list[:, 0])
    plot.get_legend().set(bbox_to_anchor=(1.05, 1.0))
    plt.tight_layout()
    plt.savefig(os.path.join(logdir, f"{dataset_name}_texture_umap.png"))


# Plot histograms of the features requested from the counts
def plot_profile(features, counts, filename, logdir):
    plt.clf()
    # features = [f for f in features if f.countable]
    features = list(filter(lambda f: f.countable, features))
    fig, axs = plt.subplots(1, len(features))
    fig.suptitle('Histograms of Feature Counts')
    for i, feature in enumerate(features):
        axs[i].hist(counts[feature.name].values(), bins=10, density=True)
        axs[i].title.set_text(feature.name)
    plt.tight_layout()
    plt.savefig(os.path.join(logdir, f'{filename}.png'))


def plot_conditional_dist(pred_feature, cond_feature, datasets, source_data, target_data,
                          logdir, summary=None, dims=None, bins=5, sample_size=None, use_prior=True):
    common = get_common_features({source_data: datasets[source_data], target_data: datasets[target_data]}, [cond_feature])
    kl_divs = {pred_feature.name:[], "kl_div": []}
    df = pd.DataFrame()

    for feature in common[cond_feature.name]:
        # combine the lists of pred_feature values for the current feature
        indices = None
        if not cond_feature.multiple:
            source_indices = datasets[source_data][cond_feature.name] == feature
            target_indices = datasets[target_data][cond_feature.name] == feature
        else:
            lookup_feature = expand_feature(cond_feature, feature)
            source_indices = datasets[source_data][lookup_feature] == 1
            target_indices = datasets[target_data][lookup_feature] == 1

        Q = get_feature_values(pred_feature, datasets[source_data], source_indices)
        P = get_feature_values(pred_feature, datasets[target_data], target_indices)

        if dims is None:
            dims = 1 if Q[0].ndim == 0 else Q[0].ndim
        if len(Q.shape) == 1:
            Q = Q.reshape(-1, 1)
            P = P.reshape(-1, 1)
        Q = Q[:, :dims].reshape(-1, dims)
        P = P[:, :dims].reshape(-1, dims)

        # TODO: this is a hack so far
        df = df.append([{cond_feature.name: feature, "dataset": source_data, pred_feature.name: q} for q in Q[:, 0]])
        df = df.append([{cond_feature.name: feature, "dataset": target_data, pred_feature.name: p} for p in P[:, 0]])

        # TODO: maybe these functions can be separated
        # sample elements of P and Q to get a better estimate of the KL divergence
        min_sample = (bins + 1) ** (dims + 1)
        if sample_size is None:
            sample_size = min_sample
        elif  sample_size < min_sample and use_prior:
            warnings.warn("Sample size is less than the virtual samples in the prior")

        if use_prior:
            data_min = np.where(P.min(axis=0) < Q.min(axis=0), P.min(axis=0), Q.min(axis=0))
            data_max = np.where(P.max(axis=0) > Q.max(axis=0), P.max(axis=0), Q.max(axis=0))
            prior = [np.linspace(data_min[i], data_max[i], bins) for i in range(dims)]
            prior = np.meshgrid(*prior)
        else:
            prior = [[] for _ in range(dims)]
            raise warnings.warn("Warning, not using prior is untested.")

        for _ in range(30):
            P_sample = P[np.random.choice(P.shape[0], size=sample_size, replace=True)]
            Q_sample = Q[np.random.choice(Q.shape[0], size=sample_size, replace=True)]
            P_i = np.swapaxes(np.array([np.concatenate([P_sample[:, i], prior[i].flatten()]) for i in range(dims)]), 0, 1)
            Q_i = np.swapaxes(np.array([np.concatenate([Q_sample[:, i], prior[i].flatten()]) for i in range(dims)]), 0, 1)

            P_hist, _ = np.histogramdd(P_i, bins=bins, density=True)
            P_hist /= P_hist.sum()
            Q_hist, _ = np.histogramdd(Q_i, bins=bins, density=True)
            Q_hist /= Q_hist.sum()

            kl = np.sum(kl_div(P_hist, Q_hist))
            if math.isinf(kl):
                raise ValueError("KL divergence is infinite.")
            kl_divs[pred_feature.name].append(feature)
            kl_divs["kl_div"].append(kl)

    kl_df = pd.DataFrame(kl_divs)

    plt.clf()
    if summary == 'scatter':
        return NotImplementedError("Scatter plots not implemented yet.")
    elif summary == 'histogram':
        plt.clf()
        sns.violinplot(data=df, x=cond_feature.name, y=pred_feature.name, hue="dataset", split=True)
        plt.tight_layout()
    elif summary == 'umap':
        return NotImplementedError("UMAP plots not implemented yet.")
    if summary is not None:
        plt.savefig(os.path.join(logdir, f'Q={source_data},P={target_data}_{pred_feature.name}|{cond_feature.name}_{summary}.png'))

    # plot bar chart from kl_divs dictionary
    plt.clf()
    sns.barplot(data=kl_df, x=pred_feature.name, y="kl_div")
    plt.tight_layout()
    plt.savefig(os.path.join(logdir, f'Q={source_data},P={target_data}_{pred_feature.name}|{cond_feature.name}_KL_Divs.png'))

    return common