import os
from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm

from research_edm.DATA.class_mapping import get_data_type
from research_edm.configs.paths import dataset_listings_path, datasets_base_path, mapping_dump_base
from research_edm.dataloader.feature_extractor import get_features_labels
from research_edm.io.pickle_io import dump_data
from research_edm.normalisation.postprocessing import default_t


def main_generate_mappings():
    ds_fd = open(dataset_listings_path, "r")
    datasets = [x.strip() for x in ds_fd.readlines()]

    for dataset in tqdm(datasets, desc='Generating datasets one-hot mappings...'):
        get_mapping(dataset)


def get_mapping(dset):
    dset_name = dset.split("/")[-1].split(".")[0]
    data_type = get_data_type(dset_name)

    _, labels = get_features_labels(
        data_file=os.path.join(datasets_base_path, dset),
        transform=default_t,
        mean=None,
        stdev=None,
        normalise=False,
        num_images=None  # consider all images
    )

    lb = LabelBinarizer()
    lb.fit(labels)
    one_hot_labels = lb.transform(labels)
    labels_back = lb.inverse_transform(one_hot_labels)

    assert labels == list(labels_back),\
        "Mapping between one-hot and categorical is not well defined for dataset {}!".format(dset)

    dump_data(lb, os.path.join(mapping_dump_base, data_type, dset_name + ".pkl"))


if __name__ == '__main__':
    main_generate_mappings()
