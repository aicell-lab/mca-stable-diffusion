# TODO rerun analysis from calculated stats on existing folder
    # Need to save indices of each dataset for thi
# TODO timestamp data_caches so that the Explorer class can lazily regenerate stuff
# TODO these need to be abstracted so they can be used over model outputs at inferece time
# TODO and so that they all just deal with the dataset structure or external defined datatypes
# TODO use the streaming module instead of loading into memory?
# TODO protein names contain a list of them and their isoforms, so they are brittle to lookup by other names
# TODO add metrics on the other input data as well? Like for the cell line and protein encoding?

# TODO: Classes for datasets with methods to analyze them
# TODO: Classes for datasets that are constructed using another one have constructors made accordingly

import os
import sys
import numpy as np
import prettyprinter as pp
import matplotlib.pyplot as plt

from ldm.evaluation.dataset_features import protein, cell_line, location, img_hash, int_mean, int_var, img_haralick, nuc_cyto
from ldm.evaluation.dataset_utils import load_config, allocate_logdir
from ldm.evaluation.dataset_utils import HPAToPandas, get_counts, get_common_features, get_total_features
from ldm.evaluation.dataset_vis import plot_profile, plot_haralick_umap, plot_conditional_dist

"""
checkpoint:
logs
2023-01-17T01-02-06_hpa-ldm-vq-4-hybrid-protein-ishan
checkpoints
epochs=000073.ckpt
"""
# model_checkpoint = os.path.join("logs", "2023-01-17T01-02-06_hpa-ldm-vq-4-hybrid-protein-ishan", "checkpoints", "epochs=000073.ckpt")

config, opt = load_config()
logdir = allocate_logdir(config, opt)

if not opt.debug:
    log_file = os.path.join(logdir, f"dataset_explorer.log")
    print(f"Saving logs to {log_file}")
    sys.stdout = open(log_file, "w")

features = [img_hash, protein, cell_line, location, int_mean, int_var, img_haralick, nuc_cyto]
datasets = HPAToPandas(config, features, expand="total", cache=True, logdir=logdir)
counts = get_counts(datasets, features)

for dataset_counts, dataset_name in zip(counts, datasets.keys()):
    plot_profile(features, dataset_counts, dataset_name, logdir)

# calculate counts of common proteins, cell-lines, and localizations between datasets
# for each dataset, get the intersection of proteins, cell-lines, and localizations
# express them as percent shared out of total and for each dataset's set of classes
common = get_common_features(datasets, features, logdir)
total = get_total_features(datasets, features, logdir)
common_counts = {feature: len(vals) for feature, vals in common.items()}
total_counts = {feature: len(vals) for feature, vals in total.items()}
print("\n")
pp.pprint(f"Common features: {common_counts}")
pp.pprint(f"Total features: {total_counts}")

for dataset_name, dataset in datasets.items():
    plot_haralick_umap(dataset, dataset_name, common[location.name],
        logdir, sample_size=10)
    
# Get prior to smooth distribution and reduce chance of accidental infs
plot_conditional_dist(nuc_cyto, cell_line, datasets, 'train', 'validation', logdir, summary='histogram', dims=1, bins=5)

# get binned histogram of intensities using a beta distribution centered at -0.5 with number of samples=bins
# TODO: ending up just needing a uniform prior over the distribution, otherwise the KL stuff becomes ill-defined...
# would've liked to use a beta distribution but no guarantee of covering the whole range
# I could do it but then the total number of points I would need to sample would go up so there is an integer number in the bin with the lowest probability
plot_conditional_dist(int_mean, location, datasets, 'train', 'validation', logdir, summary='histogram', bins=10, sample_size=15)

plot_conditional_dist(img_haralick, location, datasets, 'train', 'validation', logdir, dims=4, bins=5, sample_size=15)

# # report KL-divergence between attribute distributions
# # need ps to get list across all keys, not just those present in one dataset or the other
# div_smoothing = {feature: [] for feature, _ in profile_features}
# smoothing_range = [0.1 ** i for i in range(0, 10)]
# for smoothing in smoothing_range:
#     for feature in div_smoothing:
#         # TODO: way to do this programatically? for train and validation?
#         div_smoothing[feature].append(data_exp.kl_div('validation', 'train', feature, smoothing))

# # TODO: maybe we should just make lists and filter/match things later
# # like this workflow below doesn't fit easily into the profile structure
# # get the localizations of each protein
# # TODO: could add an option for counts or just dictionary rule or smthg
# protein_localizations = {}
# for dataset, samples in data_exp:
#     for sample in samples:
#         prot = get_protein(sample)[0]
#         loc = get_location(sample)[0]
#         if prot not in protein_localizations:
#             protein_localizations[prot] = set()
#         protein_localizations[prot].update(loc)

# # count the number of localizations for each protein
# protein_localization_counts = {}
# for protein, localizations in protein_localizations.items():
#     protein_localization_counts[protein] = len(localizations)

# # plot counts of localizations for each protein
# plt.clf()
# x = protein_localization_counts.values()
# plt.hist(x, bins=10)
# plt.title("Protein Localization Counts")
# plt.savefig(os.path.join(logdir, 'protein_localization_counts.png'))

# # print protein localization counts by dataset
# for k in data_profile:
#     multilocalizing = 0
#     for protein in data_profile[k]["Protein Counts"]:
#         if protein_localization_counts[protein] > 1:
#             multilocalizing += 1
#     print(f"{k} total proteins {len(data_profile[k]['Protein Counts'])}")
#     print(f"{k} has {multilocalizing} multilocalizing proteins")

# # histograms of dataset image intensities
# def median(x): return np.median(x)
# def mean(x): return np.mean(x)
# def var(x): return np.var(x)
# stats = [median, mean, var]
# for stat in stats:
#     plt.clf()
#     fig, axs = plt.subplots(1, len(data.datasets), sharex=True, sharey=True)
#     fig.suptitle(f'{stat.__name__} Intensity Histograms (PMF)')
#     start_index = 0
#     for i, k in enumerate(data.datasets):
#         offset = len(data.datasets[k])
#         dataset = data.datasets[k].samples[start_index : start_index + offset] # TODO
#         start_index += offset

#         intensities = []
#         for sample in dataset:
#             intensities.append(stat(sample['image']))

#         intensities = np.asarray(intensities)
#         intensities = (intensities + 1) / 2 # normalize to [0, 1]
#         axs[i].hist(intensities, bins=10, density=True)
#         axs[i].title.set_text(k)

#     plt.savefig(os.path.join(logdir, f'{stat.__name__}_intensity_histograms.png'))
