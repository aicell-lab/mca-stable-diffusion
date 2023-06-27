import matplotlib.pyplot as plt
import webdataset as wds
import pickle
import numpy as np
import json
from tqdm import tqdm

def main():
    url = "/data/wei/hpa-webdataset-all-composite/webdataset_embed.tar"
    dataset = wds.WebDataset(url).decode().to_tuple("__key__", 'desenet.pyd')
    dataset_iter = iter(dataset)
    TOTAL_LENGTH = 247678
    densent_features = []

    for index in tqdm(range(TOTAL_LENGTH), total=TOTAL_LENGTH):
        key, desenet = next(dataset_iter)
        assert desenet.shape == (1, 1024)
        densent_features.append(desenet)

    # with open("/data/wei/hpa-webdataset-all-composite/HPACombineDatasetInfo-densenet-features.pickle", "wb") as f:
    #     pickle.dump(np.concatenate(densent_features, axis=0), f)

    densent_features = np.concatenate(densent_features, axis=0)
    assert desenet.shape == (TOTAL_LENGTH, 1024)

    url = "/data/wei/hpa-webdataset-all-composite/webdataset_info.tar"
    info_dataset = wds.WebDataset(url).decode().to_tuple("__key__", "info.json")
    info_dataset_iter = iter(info_dataset)
    ensembl_ids = {}
    index = 0

    for index in tqdm(range(TOTAL_LENGTH), total=TOTAL_LENGTH):
        key, info = next(info_dataset_iter)
        gid = info['ensembl_ids']
        if not isinstance(gid, str):
            continue
        if gid not in ensembl_ids:
            ensembl_ids[gid] = [index]
        else:
            ensembl_ids[gid].append(index)
        index += 1

    idl = []
    for ids in list(ensembl_ids.keys()):
        idl.append(len(ensembl_ids[ids]))
    idl = np.array(idl)

    plt.hist(idl, bins=100)

    densent_features_avg = []
    multi_avg_embeddings = []

    url = "/data/wei/hpa-webdataset-all-composite/webdataset_info.tar"
    info_dataset = wds.WebDataset(url).decode().to_tuple("__key__", "info.json")
    info_dataset_iter = iter(info_dataset)

    for index in tqdm(range(TOTAL_LENGTH), total=TOTAL_LENGTH):
        key, info = next(info_dataset_iter)
        gid = info['ensembl_ids']
        if not isinstance(gid, str):
            # no emsembl id
            densent_features_avg.append(np.zeros([1, 1024], dtype='float32'))
            continue
        assert index in ensembl_ids[gid], f"{index} not in {ensembl_ids[gid]}"
        if len(ensembl_ids[gid]) == 1:
            densent_features_avg.append(densent_features[index][None, :])
        else:
            multi_avg_embeddings.append(index)
            ids = ensembl_ids[gid].copy()
            ids.remove(index)
            assert index in ensembl_ids[gid]
            densent_features_avg.append(densent_features[ids].mean(axis=0))

    # with open("/data/wei/hpa-webdataset-all-composite/HPACombineDatasetInfo-indexes-densenet-features-avg.json", "w") as f:
    #     # write the indexes of multi avg embeddings into a json file
    #     json.dump(multi_avg_embeddings, f)

    # with open("/data/wei/hpa-webdataset-all-composite/HPACombineDatasetInfo-densenet-features-avg.pickle", "wb") as f:
    #     pickle.dump(densent_features_avg, f)
    
    print('all done')
    print(TOTAL_LENGTH - len(multi_avg_embeddings))
    
main()