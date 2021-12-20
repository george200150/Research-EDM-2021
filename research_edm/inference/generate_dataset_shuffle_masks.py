import os
import numpy as np

from research_edm.dataloader.feature_extractor import get_features_labels
from research_edm.io.pickle_io import dump_data, get_mask
from research_edm.normalisation.postprocessing import Wrap, identic
from research_edm.configs.paths import datasets_base_path, mask_dump_base, dataset_listings_path


def get_shuffle_mask(dset):
    dset_name = dset.split("/")[-1].split(".")[0]
    data_type = "grades" if "note" in dset_name else "categories"

    features, labels = get_features_labels(
        data_file=os.path.join(datasets_base_path, dset),
        transform=Wrap(identic),
        mean=None,
        stdev=None,
        normalise=False,
        num_images=None  # consider all images
    )

    size = len(labels)
    mask = list(range(size))
    np.random.shuffle(mask)
    dump_data(mask, os.path.join(mask_dump_base, data_type, dset_name + ".pkl"))


def main_generate_masks():
    ds_fd = open(dataset_listings_path, "r")
    datasets = [x.strip() for x in ds_fd.readlines()]

    for dataset in datasets:
        get_shuffle_mask(dataset)


def test_shuffle_data():
    ds_fd = open(dataset_listings_path, "r")
    datasets = [x.strip() for x in ds_fd.readlines()]
    dset = datasets[0]

    dset_name = dset.split("/")[-1].split(".")[0]
    data_type = "grades" if "note" in dset_name else "categories"
    mask = get_mask(os.path.join(mask_dump_base, data_type, dset_name + ".pkl"))

    features, labels = get_features_labels(
        data_file=os.path.join(datasets_base_path, dset),
        transform=Wrap(identic),
        mean=None,
        stdev=None,
        normalise=False,
        num_images=None  # consider all images
    )

    print("BEFORE features.size = ", features.size)
    print("BEFORE labels = ", labels)

    features_shuffled = features[mask]
    labels_shuffled = list(np.asarray(labels)[mask])

    print("AFTER features.size = ", features_shuffled.size)
    print("AFTER features = ", features_shuffled)
    print("AFTER labels = ", labels_shuffled)


if __name__ == '__main__':
    main_generate_masks()
    # test_shuffle_data()
