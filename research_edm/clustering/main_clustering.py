import os

from sklearn.cluster import KMeans

from research_edm.DATA.class_mapping import unmap_category, get_data_type, grades_type
from research_edm.clustering.visualization import visualize_3d_clustering, generate_colors_per_class_7cls, \
    generate_colors_per_class_5cls, generate_colors_per_class_2cls
from research_edm.configs.paths import dset_mean_stdev_dump_base, datasets_base_path, plot_dump_base, \
    dataset_listings_path, clustering_dump_base
from research_edm.dataloader.feature_extractor import get_features_labels
from research_edm.evaluation.classification_metrics import get_quality_metrix
from research_edm.inference.model_instantiation import parse_cluster_params, instantiate_clustering_dryrun
from research_edm.io.pickle_io import get_mean_std, dump_data
from research_edm.normalisation.postprocessing import default_t


def plot_my_data(clusterings, clustering_names, color_scheme, dset_name, fresh_start, given_labels, savefig, title,
                 transform, wrapped_models):
    plts = [visualize_3d_clustering(clustering, given_labels, title=c_name + " " + title, colors_per_class=color_scheme,
                                    savefig=savefig) for clustering, c_name in zip(clusterings, clustering_names)]
    plot_paths = [os.path.join(plot_dump_base, transform.name, dset_name + wrapped_model.png_name) for
                  wrapped_model in wrapped_models]
    if fresh_start:
        for plot, path in zip(plts, plot_paths):
            if savefig:
                plot.savefig(path, dpi=500)
            else:
                plot.show()


def cluster_dataset(no_classes, dset, transform, normalisation, savefig, fresh_start, active_models, models_configs):
    dset_name = dset.split("/")[-1].split(".")[0]
    mean_stdev_pkl_name = "{}_mean_stdev.pkl".format(dset_name)
    if no_classes == 7:  # grades
        color_scheme = generate_colors_per_class_7cls()
        no_cls = no_classes
    elif no_classes == 5:  # E V G S F
        color_scheme = generate_colors_per_class_5cls()
        no_cls = no_classes
    elif no_classes == 2:  # P F
        color_scheme = generate_colors_per_class_2cls()
        no_cls = no_classes
    else:
        raise ValueError("No such class mapping!")

    mean, stdev = get_mean_std(os.path.join(dset_mean_stdev_dump_base, mean_stdev_pkl_name))

    features, labels = get_features_labels(
        data_file=os.path.join(datasets_base_path, dset),
        transform=transform,
        mean=mean,
        stdev=stdev,
        normalise=normalisation,
        num_images=None  # consider all images
    )
    paths = []

    title = dset_name + "_" + transform.name

    # instantiate the models
    if active_models is None or len(active_models) == 0:
        wrapped_models = instantiate_clustering_dryrun()
    else:
        wrapped_models = parse_cluster_params(models_configs, active_models)

    # initialize names for specific file extensions
    for wrapped_model in wrapped_models:
        wrapped_model.set_png_ending()
        wrapped_model.set_pkl_ending(transform, normalisation)

    clusterings = [wrapped_model.model.fit_transform(features) for wrapped_model in wrapped_models]
    clustering_names = [wrapped_model.name for wrapped_model in wrapped_models]
    plot_my_data(clusterings, clustering_names, color_scheme, dset_name, fresh_start, labels, savefig,
                 title + " Ground Truth", transform, wrapped_models)

    kmeans_labels = KMeans(n_clusters=no_cls).fit_predict(features)

    dump_paths = [os.path.join(clustering_dump_base, transform.name, dset_name + wrapped_model.complete_model_name)
                  for wrapped_model in wrapped_models]

    for clustering, dump_path in zip(clusterings, dump_paths):
        dump_data([clustering, kmeans_labels, labels], dump_path)
        paths.append(dump_path)

    if get_data_type(dset_name) == grades_type:
        kmeans_labels = list([str(x+4) for x in kmeans_labels])
    else:
        kmeans_labels = list([unmap_category(no_classes, x + 4) for x in kmeans_labels])

    # now that the k-means labels have been introduced, we must change the models' png endings
    for wrapped_model in wrapped_models:
        wrapped_model.set_png_ending(kmeans=True)

    for indx, clustering in enumerate(clusterings):
        print(f"[INFO]: For dataset {dset} and clustering {indx}, the quality is the following: "
              f"{get_quality_metrix(st=labels, a=clustering)}")

    plot_my_data(clusterings, clustering_names, color_scheme, dset_name, fresh_start, kmeans_labels, savefig,
                 title + " K-Means", transform, wrapped_models)

    return paths


def main_cluster(no_classes, transform, normalisation, savefig, fresh_start, active_models, models_configs):
    ds_fd = open(dataset_listings_path, "r")
    datasets = [x.strip() for x in ds_fd.readlines()]

    paths = []
    for dataset in datasets:
        paths += cluster_dataset(no_classes, dataset, transform, normalisation, savefig, fresh_start, active_models, models_configs)
    return paths


if __name__ == '__main__':
    paths = main_cluster(7, default_t, True, False, False, None, None)
    # paths = main_cluster(5, default_t, True, False, False, None, None)
    # paths = main_cluster(2, default_t, True, False, False, None, None)
    print(paths)
