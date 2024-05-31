from fld.features.DINOv2FeatureExtractor import DINOv2FeatureExtractor
from fld.features.InceptionFeatureExtractor import InceptionFeatureExtractor
from fld.features.CLIPFeatureExtractor import CLIPFeatureExtractor
from sklearn.decomposition import PCA
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN
import matplotlib.pyplot as plt
import re
import glob
import os
import umap
import argparse

"""
Example commands to launch:

`python average_embedding.py different_guidance --guide /path/to/directory/with/differnt/guidance/images --dir_regex regex_to_find_subdirs_in_guide --gt /path/to/dir/of/ground/truth --save_path optional/path/to/cache/features --extractor Inception --labelregex regex_for_labels_from_dirnames_optional`

 `python average_embedding.py different_conditions --function umap --different_cond z --image_type gen --extractor Inception --dir /path/to/dir/with/subdirs --regex regex_to_find_subdirs_with_images_in_dir --save_path /path/to/cache/features/optional`

"""

def pretty_labels_in_plot(labels, type_to_research):
    """
    Replace the labels from the images to labels that are easier to read in the plot.
    A header with the constant values and the varying values is added to the labels.
    input: 
    labels: list of labels from the images
    type_to_research: type of varying image, either z, mst or poi

    output:
    header: str with constant values
    labels: list of labels with the varying values
    """
    if type_to_research == 'z':
        mst=labels[0].split("_")[1]
        poi=labels[0].split("_")[5]
        header = f"MST: {mst}, POI: {poi}"
        new_labels = [f"Z: {label.split('_')[3]}" for label in labels]
        return header, new_labels
    elif type_to_research == 'mst':
        z=labels[0].split("_")[3]
        poi=labels[0].split("_")[5]
        header = f"Z: {z}, POI: {poi}"
        new_labels = [f"MST: {label.split('_')[1]}" for label in labels]
        return header, new_labels
    elif type_to_research == 'poi':
        z=labels[0].split("_")[3]
        mst=labels[0].split("_")[1]
        header = f"Z: {z}, MST: {mst}"
        new_labels = [f"POI: {label.split('_')[5]}" for label in labels]
        return header, new_labels

def find_images_dir_path_and_label(directory, dirregex, labelregex="(mst[A-Za-z\d_]*)"):
    """
    Find the images directory path and label from the subdirectories in the directory.

    Args:
        directory (str): The directory path where the subdirectories are located.
        dirregex (str): Regular expression pattern to match the subdirectory names.
        labelregex (str, optional): Regular expression pattern to extract the label from the directory name.
            Defaults to "(mst[A-Za-z\d_]*)".

    Returns:
        tuple: A tuple containing two lists - image_dirs and labels.
            - image_dirs (list): A list of directory paths that match the dirregex pattern.
            - labels (list): A list of labels extracted from the directory names.

    Example:
        >>> directory = "/path/to/directory"
        >>> dirregex = "subdir_.*"
        >>> labelregex = "(mst[A-Za-z\d_]*)"
        >>> image_dirs, labels = find_images_dir_path_and_label(directory, dirregex, labelregex)
        >>> print(image_dirs)
        ['/path/to/directory/subdir_1/', '/path/to/directory/subdir_2/']
        >>> print(labels)
        ['mst_10_z_15_poi_TPR', 'mst_20_z_25_poi_FPR']
    """
    # Find all subdirectories in the directory
    dirs = glob.glob(directory + "/*/")
    image_dirs = []
    labels = []
    for dir in dirs:
        match = re.search(dirregex, dir)
        if match:
            image_dirs.append(dir)
            # Extract the label from the directory name, e.g., "mst_10_z_15_poi_TPR"
            dirname = os.path.dirname(dir).split("/")[-1]
            match = re.search(labelregex, dirname)
            labels.append(match.group(1))
            
    return image_dirs, labels

def extract_features(paths, labels, save_path, type='generated', feature_extractor='DINOv2'):
    """
    Extract features from the images using a feature extractor and save the features to the save_path.

    Args:
        paths (list): List of paths to the image directories.
        labels (list): List of labels corresponding to each directory. It is assumed images with the same condition are in the same directory.
        save_path (str): Path to cache the extracted features.
        type (str, optional): Type of features to extract. Defaults to 'generated'.
        feature_extractor (str, optional): Name of the feature extractor to use. 
            Supported options: 'DINOv2', 'Inception', 'CLIP'. Defaults to 'DINOv2'.

    Returns:
        tuple: A tuple containing the concatenated features and concatenated feature labels.

    Raises:
        AssertionError: If the feature_extractor is not one of the supported options.

    """
    features = []
    feature_labels = []
    assert feature_extractor in ['DINOv2', 'Inception', 'CLIP']
    if feature_extractor == 'DINOv2':
        feature_extractor = DINOv2FeatureExtractor(save_path=save_path)
    elif feature_extractor == 'Inception':
        feature_extractor = InceptionFeatureExtractor(save_path=save_path)
    elif feature_extractor == 'CLIP':
        feature_extractor = CLIPFeatureExtractor(save_path=save_path)
    for i in range(len(paths)):
        features.append(feature_extractor.get_dir_features(paths[i], extension="png", name=f"{type}_features_{i}"))
        print(i, features[-1].shape, 'features shape in feature extraction')
        feature_labels.append(np.repeat(labels[i], len(features[-1])))
    return np.concatenate(features), np.concatenate(feature_labels)

    
def PCA_cluster_images_with_conditions(paths, labels, cluster_algorithm, save_path, type='generated', 
                                   pca_components=32, name="PCA_components_kmeans_clustering.png",
                                    extractor='DINOv2', **kwargs):
    """
    Take images of one type (generated or from ground truth) and cluster them using K means and PCA.
    Plot the first two PCA components. Colors indicate the cluster and markers indicate the condition type.

    :param paths: list of paths to the directories containing the images
    :param labels: list of labels for each condition, each label correspond to a certain combination of conditions
    :param cluster_algorithm: algorithm to use for clustering, either 'Kmeans' or 'DBSCAN'
    :param save_path: path to save the features, used for caching the feature extraction
    :param type: type of the images, either 'generated' or 'ground_truth'
    :param pca_components: number of PCA components to keep
    :param kmeans_clusters: number of clusters to use for K means
    :param name: name of the plot when saved

    """
    
    assert len(paths) <= 14, "Too many classes to plot markers for each class"

    all_feats, feat_labels = extract_features(paths, labels, save_path, type, feature_extractor=extractor)
       

    # run PCA before K means to reduce the dimensionality of the features
    pca = PCA(n_components=pca_components)
    all_feats = pca.fit_transform(all_feats)

    #all_feats = np.concatenate(feats)
    
    # perform K means clustering
    if cluster_algorithm == 'Kmeans':
        if kwargs['n_clusters'] is None:
            num_clusters = len(paths) # if kmeans_clusters is not specified, use the number of conditions as the number of clusters
        else:
            num_clusters = kwargs['n_clusters']
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        cluster_labels = kmeans.fit_predict(all_feats)
        
    elif cluster_algorithm == 'DBSCAN':
        dbscan = DBSCAN(eps=kwargs['eps'], min_samples=kwargs['min_samples'])
        cluster_labels = dbscan.fit_predict(all_feats)
        num_clusters = len(np.unique(cluster_labels))
    else:
        raise NotImplementedError("Only K means and DBSCAN are supported for clustering")
    
    plot_clusters_in_2D(num_clusters, all_feats, cluster_labels, feat_labels, name, type, cluster_algorithm, pca_components, extractor)


def umap_images_with_conditions(paths, labels, save_path, type_to_research, type='generated', n_components=2, name="umap.png", extractor='DINOv2'):
    """
    Apply umap to the features extracted using extractor.
    """
    print(len(paths), len(labels), 'Number of directories and labels found')
    features, feature_labels = extract_features(paths, labels, save_path, type, feature_extractor=extractor)
    print(features.shape, feature_labels.shape, 'features and labels shape')
    umap_model = umap.UMAP(n_components=n_components)
    umap_features = umap_model.fit_transform(features)
    unique_labels = np.unique(feature_labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels))).tolist()
    header, plot_labels = pretty_labels_in_plot(unique_labels, type_to_research)
    plt.figure(figsize=(10, 5))
    plt.scatter(umap_features[0, 0]+0.01, umap_features[0, 1]+0.01, marker='o', color='black', alpha=0.0, label=header)
    for i, label in enumerate(unique_labels):
        idx = np.where(feature_labels == label)[0]
        #plt.scatter(umap_features[idx, 0], umap_features[idx, 1], color=colors[i], label=fr'$\gamma$={label}' if i < len(unique_labels)-1 else 'Ground Truth')
        plt.scatter(umap_features[idx, 0], umap_features[idx, 1], color=colors[i], label=plot_labels[i])
    plt.title(f'UMAP with {n_components} components of {type.replace("_", " ")} image features extracted using {extractor}')
    #plt.title(f'UMAP with {n_components} components of image features extracted using {extractor}')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(name)

def plot_clusters_in_2D(num_clusters, all_features, cluster_labels, feature_labels, name, type, cluster_algorithm, pca_components, extractor):
    """
    Plot PCA1 and PCA2 components of the features and color the points based on the real number of classes and the clusters by their marker type.
    """
    #plot the clusters in PCA xy-plane using the labels from K means
    # I want to use different colors for each cluster and a different marker for each real class
    
    num_samples = all_features.shape[0]
    num_classes = len(np.unique(feature_labels))
    unique_labels = np.unique(feature_labels)
    assert num_clusters <= 14, "Too many clusters to plot different markers for each cluster"
    markers = generate_markers(num_clusters)
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes)).tolist()
    plt.figure(figsize=(10, 5))
    grid = plt.GridSpec(1, 3)
    
    plt.subplot(grid[0, :2])
    for i in range(num_samples):
        class_k = cluster_labels[i]
        idx = np.where(unique_labels == feature_labels[i])[0][0]
        plt.scatter(all_features[i, 0], all_features[i, 1], marker=markers[class_k], color=colors[idx])
        

    plt.title(f'PCA with {pca_components} components of {type.replace("_", " ")} image features extracted using {extractor}')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    

    plt.subplot(grid[0, 2]) # add sublot to have the legend
    for i in range(num_classes):
        #plt.scatter(0, i, marker=markers[i], label=labels[i].replace("_", " "), color='black')
        plt.scatter(0, num_classes-i, color=colors[i], label=unique_labels[i].replace("_", " "))
    plt.scatter(2, 0, marker='o', color='black', alpha=0.0, label=f'Marker style defines the\n{num_clusters} clusters given by\n{cluster_algorithm}.')
    plt.axis('off')
    plt.legend(loc='right')
    plt.tight_layout()
    plt.savefig(name)
    

def generate_markers(num_classes):
    markers = ['o', 's', 'D', '^', 'v', '>', '<', 'p', '*', 'h', 'H', 'd', 'P', 'X'] 
    return markers[:num_classes]


def one_condition_difference_analysis(function='umap', type_to_research='z', image_type='gen', extractor='Inception', dir=None, regex=None, cluster='Kmeans', save_path=None, **kwargs):
    """
    Perform analysis on different variants of images.

    Args:
        function (str, optional): The analysis function to use. Defaults to 'umap'.
        type_to_research (str, optional): The type of research to perform. Defaults to 'z'.
        image_type (str, optional): The type of images to analyze. Defaults to 'gen'.
        extractor (str, optional): The feature extractor to use. Defaults to 'Inception'.
        dir (str, optional): The directory path of the images. Defaults to None.
        regex (str, optional): The regular expression pattern to match image filenames. Defaults to None.
        cluster (str, optional): The clustering algorithm to use. Defaults to 'Kmeans'.
        save_path (str, optional): The path to save the extracted features. Defaults to None.
        **kwargs: Additional keyword arguments for the cluster algorithms Kmeans or DBSCAN function.

    Raises:
        AssertionError: If the input arguments are invalid.

    Returns:
        None
    """
    
    assert type_to_research in ['poi', 'mst', 'z'] #p options are mst, poi, z
    assert image_type in ['gen', 'gt'] #  options are gen gt
    assert extractor in ['Inception', 'DINOv2', 'CLIP'] # options are DINOv2, Inception, CLIP
    assert cluster in ['Kmeans', 'DBSCAN'] # options are Kmeans, DBSCAN

    if dir is None:
        if image_type == 'gt':
            dir = f"/proj/aicell/data/stable-diffusion/mca/logs/ldm-v1-round4-2024-04-19T15-48-43_mca_debug/normalized_gt_images_by_condition/{type_to_research}"
        else:
            dir = f"/proj/aicell/data/stable-diffusion/mca/logs/ldm-v1-round4-2024-04-19T15-48-43_mca_debug/{image_type}_imgs_different_{type_to_research}"
    if regex is None:
        regex = "mst*" if image_type == 'gt' else "gamma_1_mst*"
    dirs, labels = find_images_dir_path_and_label(dir, regex)
    
    type = 'ground_truth' if image_type == 'gt' else 'generated'
    assert function in ['umap', 'PCA']
    if function == 'umap':
        umap_images_with_conditions(dirs, labels, save_path=save_path, type=type, extractor=extractor, type_to_research=type_to_research,
                                         name=f'new_{extractor}_umap_{type}_{type_to_research}_one_difference.png')
    else:
        PCA_cluster_images_with_conditions(dirs, labels, save_path=save_path, cluster_algorithm=cluster, type=type, extractor=extractor, type_to_research=type_to_research,
                                         name=f'new_{extractor}_pca_{type}_{type_to_research}_one_difference.png', **kwargs)
        


def different_guidance(guide_dir, gt_dir, save_path=None, extractor='Inception', dir_regex="gamma_(1_|5|8|10|20)",labelregex="(gamma_\d*)"):
    """
    Compute embeddings for different guidance images and ground truth images, and generate a UMAP plot.

    Args:
        guide_dir (str): Directory containing the images with different guidance levels. Gamma = guidance level
        gt_dir (str): Directory containing the ground truth images.
        save_path (str, optional): Path to cache the extracted features.
        extractor (str, optional): Feature extractor to use. Can be 'Inception', 'DINOv2', or 'CLIP'. Defaults to 'Inception'.
        dir_regex (str, optional): Regular expression pattern to match the guidance image directories. Defaults to "gamma_(1_|5|8|10|20)".
        labelregex (str, optional): Regular expression pattern to match the guidance image labels. Defaults to "(gamma_\d*)".

    Raises:
        ValueError: If an invalid feature extractor is provided.

    Returns:
        None
    """
    assert extractor in ['Inception', 'DINOv2', 'CLIP']
    if extractor == 'CLIP':
        extractor = CLIPFeatureExtractor(save_path=save_path)
    elif extractor == 'Inception':
        extractor = InceptionFeatureExtractor(save_path=save_path)
    elif extractor == 'DINOv2':
        extractor = DINOv2FeatureExtractor(save_path=save_path)
    else:
        raise ValueError("Invalid feature extractor")

    dirs, labels = find_images_dir_path_and_label(guide_dir, dir_regex, labelregex=labelregex)
    
    dirs.append(gt_dir)
    labels.append('gt')
    umap_images_with_conditions(dirs, labels, save_path=save_path, type='different_guidance', extractor=extractor,
                                         name=f'umap_different_guidance_{extractor}.png')
    

        
def parse_args():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description='Script for running analysis on images')
    subparsers = parser.add_subparsers(dest='command', help='Choose a command to run')

    # Subparser for one_condition_difference_analysis command
    one_condition_parser = subparsers.add_parser('different_conditions', help='Perform analysis on different variants of images')
    one_condition_parser.add_argument('--function', choices=['umap', 'PCA'], default='umap', help='The analysis function to use')
    one_condition_parser.add_argument('--different_cond', choices=['poi', 'mst', 'z'], default='z', help='The type condition that varies in the images')
    one_condition_parser.add_argument('--image_type', choices=['gen', 'gt'], default='gen', help='The type of images to analyze')
    one_condition_parser.add_argument('--extractor', choices=['Inception', 'DINOv2', 'CLIP'], default='Inception', help='The feature extractor to use')
    one_condition_parser.add_argument('--dir', help='The directory path of the images')
    one_condition_parser.add_argument('--regex', help='The regular expression pattern to match image filenames in the directory')
    one_condition_parser.add_argument('--cluster', choices=['Kmeans', 'DBSCAN'], default='Kmeans', help='The clustering algorithm to use if using PCA')
    one_condition_parser.add_argument('--save_path', help='The path to cache the extracted features')

    # Subparser for different_guidance command
    different_guidance_parser = subparsers.add_parser('different_guidance', help='Compute embeddings for different guidance images and ground truth images')
    different_guidance_parser.add_argument('--guide', help='Directory containing the images with different guidance levels')
    different_guidance_parser.add_argument('--gt', help='Directory containing the ground truth images')
    different_guidance_parser.add_argument('--save_path', help='Path to cache the extracted features')
    different_guidance_parser.add_argument('--extractor', choices=['Inception', 'DINOv2', 'CLIP'], default='Inception', help='Feature extractor to use')
    different_guidance_parser.add_argument('--dir_regex', default='gamma_(1_|5|8|10|20)', help='Regular expression pattern to match the guidance image directories')
    different_guidance_parser.add_argument('--labelregex', default='(gamma_\d*)', help='Regular expression pattern to match the guidance image labels')

    return parser.parse_args()

def main():
    args = parse_args()

    if args.command == 'different_conditions':
        one_condition_difference_analysis(function=args.function, type_to_research=args.different_cond,
                                            image_type=args.image_type, extractor=args.extractor, dir=args.dir,
                                            regex=args.regex, cluster=args.cluster, save_path=args.save_path)
    elif args.command == 'different_guidance':
        different_guidance(args.guide, args.gt, save_path=args.save_path, extractor=args.extractor,
                            dir_regex=args.dir_regex, labelregex=args.labelregex)
    else:
        print('Invalid command')

if __name__ == "__main__":
    main()
