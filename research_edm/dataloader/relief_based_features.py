import os

import numpy as np
from skrebate import ReliefF

from research_edm.DATA.class_mapping import get_data_type
from research_edm.configs.paths import mask_dump_base, dset_mean_stdev_dump_base, datasets_base_path, \
    dataset_listings_path
from research_edm.dataloader.feature_extractor import get_features_labels
from research_edm.evaluation.classification_metrics import get_quality_metrix
from research_edm.io.pickle_io import get_mean_std, get_mask
from research_edm.normalisation.postprocessing import Wrap, identic

ds_fd = open(dataset_listings_path, "r")
datasets = [x.strip() for x in ds_fd.readlines()]
paths = []

dset = datasets[0]  # TODO: change dataset manually
# dset = datasets[1]

transform = Wrap(identic)
norm_flag = False

dset_name = dset.split("/")[-1].split(".")[0]
data_type = get_data_type(dset_name)
mean_stdev_pkl_name = "{}_mean_stdev.pkl".format(dset_name)

mean, stdev = get_mean_std(os.path.join(dset_mean_stdev_dump_base, mean_stdev_pkl_name))

mask = get_mask(os.path.join(mask_dump_base, data_type, dset_name + ".pkl"))

features, labels = get_features_labels(
    data_file=os.path.join(datasets_base_path, dset),
    transform=transform,
    mean=mean,
    stdev=stdev,
    normalise=norm_flag,
    num_images=None  # consider all images
)

n = len(features[0])
m = len(features)

labels = np.asarray([int(x) for x in labels])  # must have them integers

for j in range(25, m+1, 25):
    print("neighbours: ", j)
    for i in range(1, n + 1):
        print("features: ", i)

        q = get_quality_metrix(labels, ReliefF(n_features_to_select=i, n_neighbors=j).fit_transform(features, labels))
        print(f"Quality of dataset: {dset} with {i} FEATURES and {j} NEIGHBOURS is: {q}")
