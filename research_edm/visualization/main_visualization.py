import os

from research_edm.DATA.class_mapping import get_palette
from research_edm.clustering.visualization import visualize_3d_clustering
from research_edm.configs.paths import dataset_listings_path, dset_mean_stdev_dump_base, datasets_base_path
from research_edm.dataloader.feature_extractor import get_features_labels
from research_edm.io.pickle_io import get_mean_std
from research_edm.normalisation.postprocessing import default_t

import matplotlib as mpl

mpl.use('TkAgg')  # for plotting in separate window


def select_features(features, indices):
    return features[:, indices]


def main_visualization(no_classes):
    ds_fd = open(dataset_listings_path, "r")
    datasets = [x.strip() for x in ds_fd.readlines()]

    feature_indices_list = [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8]]
    transform = default_t
    norm_flag = False

    for dataset in datasets:
        color_scheme = get_palette(no_classes)

        dset_name = dataset.split("/")[-1].split(".")[0]
        mean_stdev_pkl_name = "{}_mean_stdev.pkl".format(dset_name)

        mean, stdev = get_mean_std(os.path.join(dset_mean_stdev_dump_base, mean_stdev_pkl_name))

        features, y_gts = get_features_labels(
            data_file=os.path.join(datasets_base_path, dataset),
            transform=transform,
            mean=mean,
            stdev=stdev,
            normalise=norm_flag,
            num_images=None  # consider all images
        )

        for feature_indices in feature_indices_list:
            title = dset_name + "_TRANSF_" + transform.name + "_NORM_" + str(norm_flag) + "_" + ","\
                .join([str(x) for x in feature_indices])
            x_points_sliced = select_features(features, feature_indices)

            visualize_3d_clustering(x_points_sliced, y_gts, title, color_scheme, three_d=True, savefig=False)


if __name__ == '__main__':
    main_visualization(7)
    # main_visualization(5)
    # main_visualization(2)
