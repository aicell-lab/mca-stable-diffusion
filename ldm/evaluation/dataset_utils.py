import os
import sys
# import csv
# import copy
import json
import shutil
import datetime
# import itertools
# import numpy as np
from collections import Counter
import pandas as pd
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
# from scipy.special import kl_div
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer

from ldm.parse import get_parser
from ldm.util import instantiate_from_config
from ldm.evaluation.dataset_features import expand_feature, get_features


def load_config():
    # add cwd for convenience
    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()

    seed_everything(opt.seed)

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    return config, opt


def allocate_logdir(config, opt):
    config, opt = load_config()

    # setup logging environment
    # if opt.name:
    #     name = "_" + opt.name
    # elif opt.base:
    #     cfg_fname = os.path.split(opt.base[0])[-1]
    #     cfg_name = os.path.splitext(cfg_fname)[0]
    #     name = "_" + cfg_name
    # else:
    #     name = ""

    # now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    # nowname = now + name + opt.postfix
    # logdir = os.path.join(opt.logdir, nowname)
    # setup logger to send output to file
    if opt.debug:
        logdir = os.path.join(opt.logdir, "eval_debug_logs")
    else:
        # now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        logdir = os.path.join(opt.logdir, f"eval_logs")
    os.makedirs(logdir, exist_ok=True)

    # if opt.dev:
    #     for root, dirs, files in os.walk(logdir):
    #         for f in files:
    #             os.unlink(os.path.join(root, f))
    #         for d in dirs:
    #             shutil.rmtree(os.path.join(root, d))

    return logdir


# returns a dictionary of datasets, where each dataset is a pandas dataframe of the passed in features
# config file passed in from command line
# features are a list of column names and functions to extract the feature value from a given sample
# TODO: segmentation and downstream tasks should go here
# if there is a list of column names, need to iterate through the extracted values and add them
# TODO: need to be able to get the dataset without the config? Intent with that was so you can just pass in the config
# you're using for experiments. Not sure how useful that really is.
def HPAToPandas(config, features, debug, expand=None, cache=False, logdir=None, recompute=[]):
    if expand is not None and expand not in ["common", "total"]:
        raise ValueError("expand must be either 'common' or 'total'")

    data_explorer = DatasetExplorer(config, debug)

    datasets = {}
    column_names = [feature.name for feature in features]
    for dataset_name, samples, dataset_size in data_explorer:
        print(f"Getting features for {dataset_name}")
        data_features = get_features(samples, features, dataset_size, cache=cache, logdir=logdir, dataset_name=dataset_name, recompute=recompute)
        print(f"Done getting features for {dataset_name}")
        datasets[dataset_name] = pd.DataFrame.from_dict(data_features, orient="columns")

    # TODO factor out the expansion logic to a separate function
    if expand is None:
        return datasets

    # expand multi-valued features
    if expand == "common":
        common = get_common_features(datasets, features)
        raise NotImplementedError("Expansion over common features not implemented yet.")

    if expand == "total":
        total = get_total_features(datasets, features)

        for feature in filter(lambda f: f.multiple, features):
            for dataset_name, dataset in datasets.items():
                # add a column for each value in the total set of values for this feature
                # in total we should get a one-hot encoding over the columns prefixed by the feature name
                for value in total[feature.name]:
                    dataset[expand_feature(feature, value)] = dataset[feature.name].apply(
                        lambda x: value in x
                    )

    return datasets


def get_common_features(datasets, features, logdir=None):
    common = {
        feature.name:
        set.intersection(
            *[set(instances(d[feature.name], expandLists=True)) for d in datasets.values()]
        )
        for feature in features if feature.countable
    }

    if logdir is not None:
        common_write = {key: list(val) for key, val in common.items()}
        with open(os.path.join(logdir, "common.json"), "w") as f:
            json.dump(common_write, f)
    return common


def get_total_features(datasets, features, logdir=None):
    total = {
        feature.name:
        set.union(
            *[set(instances(d[feature.name], expandLists=True)) for d in datasets.values()]
        )
        for feature in features if feature.countable
    }

    if logdir is not None:
        total_write = {key: list(val) for key, val in total.items()}
        with open(os.path.join(logdir, "total.json"), "w") as f:
            json.dump(total_write, f)

    return total


# dictionary of feature value counts
def get_counts(datasets, features):
    # Retrieve protein, cell_line, and location counts
    counts = []
    features = list(filter(lambda f: f.countable, features))
    for dataset_name, data in datasets.items():
        print(f"\nDataset: {dataset_name}")
        dataset_counts = {}
        for feature in features:
            dataset_counts[feature.name] = Counter(instances(data[feature.name], expandLists=True))
            printTab(f"Top 5 {feature}s: {dataset_counts[feature.name].most_common(5)}")
        counts.append(dataset_counts)
        print(dataset_counts)
    return counts


# Helper for features that contain lists of the actual data instances that want to be accessed
def instances(data, expandLists=False):
    if type(data[0]) == list and not expandLists:
        raise TypeError ("Data is a list of lists. Set expandLists=True to flatten.")
    elif type(data[0]) == list:
        instances = []
        for x in data:
            instances.extend(x)
    else: 
        instances = data
    return instances


def view_top_k_images(dataset, filename, logdir, k=10):
    images = []


def printTab(*args):
    args = ("\t",)+args
    print(*args)


class DatasetExplorer:
    def __init__(self, config, debug):
        # Retrieve dataset from config
        # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
        # calling these ourselves should not be necessary but it is.
        # lightning still takes care of proper multiprocessing though
        self.data = instantiate_from_config(config.data)
        self.data.prepare_data()
        self.data.setup()
        self.debug = debug

    def __iter__(self):
        return DatasetIterator(self.data.datasets, self.debug)

    # returns list[dataset_names] to index into profiles
    def dataset_names(self):
        return list(self.data.datasets.keys())

    # returns list[DatasetProfile]
    # def get_profiles(self, features):
    #     self.features = [feature_name for feature_name, _ in features]
    #     self.profiles = {}
    #     for dataset_name, samples in self:
    #         class_name = self.data.datasets[dataset_name].__class__.__name__
    #         print(f"{dataset_name}, {class_name}, {len(samples)}")
    #         self.profiles[dataset_name] = FeatureCountProfile(samples, features, self.logdir)
    #     return self.profiles
    
    # returns dict{features : set(labels)}
    def get_common_labels(self):
        raise NotImplementedError
        # if self.profiles is None:
        #     raise ValueError("Must call get_profiles first")
        # common = {}
        # for feature in self.features:
        #     feature_labels = [set(profile.label_list(feature)) for profile in self.profiles.values()]
        #     common[feature] = set.intersection(*feature_labels)
        # return common
    
    # returns dict{features : set(labels)}
    def get_total_labels(self):
        raise NotImplementedError
        # if self.profiles is None:
        #     raise ValueError("Must call get_profiles first")
        # total = {}
        # for feature in self.features:
        #     feature_labels = [set(profile.label_list(feature)) for profile in self.profiles.values()]
        #     total[feature] = set.union(*feature_labels)
        # return total

    def get_feature_pmf(self, dataset, feature, labels, smoothing):
        raise NotImplementedError
        # counts = [self.profiles[dataset].get_feature(feature)[label] for label in labels[feature]]
        # pmf = np.asarray(counts) + smoothing * len(self.data.datasets[dataset])
        # norm_pmf = pmf / np.sum(pmf)
        # return norm_pmf
        

    def kl_div(self, dataset_p, dataset_q, feature, smoothing):
        raise NotImplementedError
        # labels = self.get_total_labels()
        # ps = self.get_feature_pmf(dataset_p, feature, labels, smoothing)
        # qs = self.get_feature_pmf(dataset_q, feature, labels, smoothing)
        # return np.sum(kl_div(ps, qs))


def iterable_dataset(map_style_dataset, debug):
    for i in range(len(map_style_dataset)):
        if not debug or i < 4:
            yield map_style_dataset[i]


class DatasetIterator:
    def __init__(self, datasets, debug):
        self.datasets = datasets
        self.i, self.keys = 0, list(datasets.keys())
        self.offset = 0
        self.debug = debug

    # iterator that goes through each dataset in data
    # the samples list is a list of samples in the dataset
    # so if training has 10 samples and validation has 15
    # then validation starts at data.dataset[10]
    def __next__(self):
        if self.i == len(self.keys):
            raise StopIteration
        dataset_name = self.keys[self.i]
        dataset = self.datasets[dataset_name]

        n = len(dataset)
        # assert len(dataset.samples) >= self.offset + n
        # samples = dataset.samples[self.offset:self.offset + n]
        samples = iterable_dataset(dataset, self.debug)

        self.i += 1
        # self.offset += n

        return dataset_name, samples, n

# class FeatureCountProfile:
#     # features at a label for that statistic and a function that takes
#     # a list of samples and returns that statistic
#     def __init__(self, samples, features, logdir):
#         self.prof = {}
#         for feature_name, get_feature in features:
#             self.prof[feature_name] = FeatureCountProfile.get_counts(get_feature, samples)
#         self.logdir = logdir
    
#     def get_counts(extract_func, samples):
#         counts = Counter()
#         for sample in samples:
#             counts.update(extract_func(sample))
#         return counts

#     def top_percents(self, name, top=5, digits=2):
#         counts = self.prof[name]
#         total = 0
#         for k in counts:
#             total += counts[k]
        
#         percents = {}
#         for k, v in counts.most_common(top):
#             percents[k] = f"{round(100 * v / total, digits)}%"
#         return percents

#     def write_to_csv(self, features, fileprefix):
#         for feature, _ in features:
#             with open(os.path.join(self.logdir, f"{fileprefix}_{feature}.csv"), "w", newline="") as f:
#                 fieldnames = [feature, "count"]
#                 writer = csv.writer(f)
#                 writer.writerow(fieldnames)
#                 for key, val in self.prof[feature].items():
#                     writer.writerow([key, val])   

#     # passing features manually most places so it is forwards compatibile
#     # if we don't care about some of the features later on
#     # features is the original 
#     # TODO: feel like this feature passing interface might be more confusing
#     # TODO: maybe manage the list of features in the DatasetExplorer
#     def plot_profile(self, features, filename):
#         plt.clf()
#         fig, axs = plt.subplots(1, len(features))
#         fig.suptitle('Histograms of feature categories')
#         for i, (feature, _) in enumerate(features):
#             counts = self.prof[feature].values()
#             axs[i].hist(counts, bins=10, density=True)
#             axs[i].title.set_text(feature)
#         plt.savefig(os.path.join(self.logdir, f'{filename}.png'))
    
#     # feature is the string label
#     # really for use by the DatasetExplorer
#     # TODO: feel like this feature passing interface might be more confusing
#     def label_list(self, feature):
#         return list(self.prof[feature].keys())

#     def get_feature(self, feature):
#         return self.prof[feature]

# config should have the path to the desired validation dataset
# should also specify the model checkpoint to use
def generate_checkpoint_predictions(config):
    data_explorer = DatasetExplorer(config)
    if "analysis" not in data_explorer.data.datasets:
        data = data_explorer.data.datasets["validation"]
    else:
        data = data_explorer.data.datasets["analysis"]
    model = instantiate_from_config(config.model)
    checkpoint = torch.load(config.checkpoint)
    model.load_state_dict(checkpoint)
    model.eval()
    predictions = model.predict(data, config.logdir)
    print("bleh")