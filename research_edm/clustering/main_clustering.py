import os

from sklearn.cluster import KMeans
import numpy as np
import random

from research_edm.clustering.visualization import visualize_3d_clustering, generate_colors_per_class_7cls, \
    generate_colors_per_class_4cls
from research_edm.dataloader.feature_extractor import get_features_labels
from research_edm.io.pickle_io import get_mean_std, dump_data
from research_edm.normalisation.postprocessing import identic, Wrap

from umap import UMAP
from sklearn.manifold import TSNE


datasets_base_path = r"C:\Users\George\PycharmProjects\research_edm\research_edm\DATA"
dataset_listings_path = r"C:\Users\George\PycharmProjects\research_edm\research_edm\DATA\dataset_listings.txt"
dset_mean_stdev_dump_base = \
    r"C:\Users\George\PycharmProjects\research_edm\research_edm\normalisation\datasets_mean_stdev_dumps"
clustering_dump_base = r"C:\Users\George\PycharmProjects\research_edm\research_edm\clustering\clustering_dump"
plot_dump_base = r"C:\Users\George\PycharmProjects\research_edm\research_edm\clustering\plot_dump"


def cluster_dataset(dset, transform):
    seed = 10
    random.seed(seed)
    # torch.manual_seed(seed)
    np.random.seed(seed)

    dset_name = dset.split("/")[-1].split(".")[0]
    mean_stdev_pkl_name = "{}_mean_stdev.pkl".format(dset_name)
    if "note" in dset:  # grades
        color_scheme = generate_colors_per_class_7cls()
        no_cls = 7
    else:
        color_scheme = generate_colors_per_class_4cls()
        no_cls = 4

    mean, stdev = get_mean_std(os.path.join(dset_mean_stdev_dump_base, mean_stdev_pkl_name))

    features, labels = get_features_labels(
        data_file=os.path.join(datasets_base_path, dset),
        transform=transform,
        mean=mean,
        stdev=stdev,
        normalise=True,
        num_images=None  # consider all images
    )
    paths = []

    title = dset_name + "_" + transform.name
    umap_clustering = UMAP(n_components=2).fit_transform(features)
    # umap_dump_path = os.path.join(clustering_dump_base, transform.name, title + "_umap.pkl")
    # dump_data([umap_clustering, labels], umap_dump_path)
    # paths.append(umap_dump_path)
    #
    tsne_clustering = TSNE(n_components=2, perplexity=15, early_exaggeration=10.0, learning_rate=150, n_iter=2000).fit_transform(features)
    # tsne_dump_path = os.path.join(clustering_dump_base, transform.name, title + "_tnse.pkl")
    # dump_data([tsne_clustering, labels], tsne_dump_path)
    # paths.append(tsne_dump_path)

    # kmeans_clustering = KMeans(n_clusters=no_cls).fit_transform(features)
    kmeans_labels = KMeans(n_clusters=no_cls).fit_predict(features)
    # kmeans_dump_path = os.path.join(clustering_dump_base, transform.name, title + "_kmeans.pkl")
    # dump_data([kmeans_clustering, labels], kmeans_dump_path)
    # paths.append(kmeans_dump_path)

    plt_umap = visualize_3d_clustering(umap_clustering, labels, title=title, colors_per_class=color_scheme, savefig=True)
    umap_plot_path = os.path.join(plot_dump_base, transform.name, title + "_umap.png")
    # plt_umap.savefig(umap_plot_path, dpi=500)
    plt_umap.show()
    #
    plt_tsne = visualize_3d_clustering(tsne_clustering, labels, title=title + " Ground Truth", colors_per_class=color_scheme, savefig=True)
    # tsne_plot_path = os.path.join(plot_dump_base, transform.name, title + "_tsne.png")
    # plt_tsne.savefig(tsne_plot_path, dpi=500)
    plt_tsne.show()

    plt_tsne = visualize_3d_clustering(tsne_clustering, [str(x+4) for x in kmeans_labels], title=title + " K-means", colors_per_class=color_scheme, savefig=True)
    plt_tsne.show()

    plt_kmeans = visualize_3d_clustering(umap_clustering, labels, title=title + " Ground Truth", colors_per_class=color_scheme, savefig=True)
    # kmeans_plot_path = os.path.join(plot_dump_base, transform.name, title + "_kmeans.png")
    # plt_kmeans.savefig(kmeans_plot_path, dpi=500)
    plt_kmeans.show()

    plt_kmeans = visualize_3d_clustering(umap_clustering, [str(x+4) for x in kmeans_labels], title=title + " K-means", colors_per_class=color_scheme, savefig=True)
    plt_kmeans.show()

    return paths


def main_cluster(transform):
    ds_fd = open(dataset_listings_path, "r")
    datasets = [x.strip() for x in ds_fd.readlines()]

    paths = []
    for dataset in datasets:
        paths.append(cluster_dataset(dataset, transform))
    return paths


if __name__ == '__main__':
    paths = main_cluster(Wrap(identic))
    print(paths)
