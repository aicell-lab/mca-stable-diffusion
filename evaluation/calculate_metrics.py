from fld.metrics.FLD import FLD
from fld.features.DINOv2FeatureExtractor import DINOv2FeatureExtractor
from fld.features.InceptionFeatureExtractor import InceptionFeatureExtractor
from fld.features.CLIPFeatureExtractor import CLIPFeatureExtractor
from fld.metrics.FID import FID
from fld.metrics.KID import KID
from fld.metrics.AuthPct import AuthPct
from fld.metrics.CTTest import CTTest
from fld.metrics.PrecisionRecall import PrecisionRecall
from fld.sample_evaluation import sample_memorization_scores
import argparse


"""
Example usage: python calculate_metrics.py -g /path/to/generated/images -t /path/to/training/images -f Inception --cache-path /path/to/cache
"""


def calculate_metrics(gen_imgs_path, train_imgs_path, feature_extractor, cache_path=None):
    """
    Calculates various evaluation metrics for generated images and prints them.

    Args:
        gen_imgs_path (str): The path to the directory containing the generated images.
        train_imgs_path (str): The path to the directory containing the training (real) images.
        feature_extractor (str): The name of the feature extractor to use. Supported options are "DINOv2", "Inception", and "CLIP".
        cache_path (str, optional): The path to the directory where the extracted features will be saved. Defaults to None.

    Raises:
        ValueError: If an invalid feature extractor is specified.

    Returns:
        None
    """
 
    # create the feature extractor
    if feature_extractor == "DINOv2":
        feature_extractor = DINOv2FeatureExtractor(save_path=cache_path)
    elif feature_extractor == "Inception":
        feature_extractor = InceptionFeatureExtractor(save_path=cache_path)
    elif feature_extractor == "CLIP":
        feature_extractor = CLIPFeatureExtractor(save_path=cache_path)
    else:
        raise ValueError("Invalid feature extractor")
    

    gen_feat = feature_extractor.get_dir_features(gen_imgs_path, extension="png", name="generated_features")
    train_feat = feature_extractor.get_dir_features(train_imgs_path, extension="png", name="train_features")

    fid_val = FID().compute_metric(train_feat, None, gen_feat)
    print(f"FID score: {fid_val:.3f}")

    test_feat = train_feat
    fld_val = FLD(eval_feat="train").compute_metric(train_feat, test_feat, gen_feat) # by passing eval_feat="train", we are evaluating the FLD score on the train set.
                                                                                    # the test_feat are not used but must be passed
    print(f"FLD score: {fld_val:.3f}")


    auth_pct_val = AuthPct().compute_metric(train_feat, None, gen_feat) # test samples not used by AuthPct
    print(f"AuthPct score: {auth_pct_val:.3f}")
    # computes the % of samples where the distance to the sample's nearest neighbour in the train set is smaller
    # than then distance between that train sample and its nearest train sample


    precision = PrecisionRecall(mode="Precision").compute_metric(train_feat, None, gen_feat) # test samples not used by PrecisionRecall
    print(f"Precision score: {precision:.3f}")

    recall = PrecisionRecall(mode="Recall").compute_metric(train_feat, None, gen_feat)
    print(f"Recall score: {recall:.3f}")
    


def parse_args():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Calculate metrics for generated and trained images")
    parser.add_argument("-g", "--gen-images-path", type=str, help="Path to the generated images")
    parser.add_argument("-t", "--train-images-path", type=str, help="Path to the trained images")
    parser.add_argument("-f", "--feature-extractor", type=str, help="Feature extractor to use", default="Inception", choices=["DINOv2", "Inception", "CLIP"])
    parser.add_argument("--cache-path", type=str, help="Path to save the feature cache", default=None)
    return parser.parse_args()


def main(args):
    """
    Example paths
    Paths to predicted and ground truth images
    gen_imgs_path = "/proj/aicell/data/stable-diffusion/mca/logs/ldm-v1-round4-2024-04-19T15-48-43_mca_debug/unconditional_images_set/predicted"
    train_imgs_path = "/proj/aicell/data/stable-diffusion/mca/logs/ldm-v1-round4-2024-04-19T15-48-43_mca_debug/unconditional_images_set/ground_truth"

    Calculate baseline score between two sets of real images
    gen_imgs_path = "/proj/aicell/data/stable-diffusion/mca/logs/ldm-v1-round4-2024-04-19T15-48-43_mca_debug/unconditional_images_set/gt_images_for_baseline/fake_gen_images"
    train_imgs_path="/proj/aicell/data/stable-diffusion/mca/logs/ldm-v1-round4-2024-04-19T15-48-43_mca_debug/unconditional_images_set/gt_images_for_baseline/real_images"  

    """
    
    calculate_metrics(args.gen_images_path, args.train_images_path, args.feature_extractor, args.cache_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)


    
