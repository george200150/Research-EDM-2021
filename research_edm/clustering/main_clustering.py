import os

from sklearn.cluster import KMeans

from research_edm.DATA.class_mapping import classes_grades, classes_categories
from research_edm.clustering.visualization import visualize_3d_clustering, generate_colors_per_class_7cls, \
    generate_colors_per_class_4cls
from research_edm.configs.paths import dset_mean_stdev_dump_base, datasets_base_path, plot_dump_base, \
    dataset_listings_path
from research_edm.dataloader.feature_extractor import get_features_labels
from research_edm.inference.model_instantiation import get_data_type, grades_type
from research_edm.io.pickle_io import get_mean_std, dump_data
from research_edm.normalisation.postprocessing import identic, Wrap

from umap import UMAP
from sklearn.manifold import TSNE


def cluster_dataset(dset, transform, normalisation):
    dset_name = dset.split("/")[-1].split(".")[0]
    mean_stdev_pkl_name = "{}_mean_stdev.pkl".format(dset_name)
    if get_data_type(dset) == grades_type:
        color_scheme = generate_colors_per_class_7cls()
        no_cls = len(classes_grades)
    else:
        color_scheme = generate_colors_per_class_4cls()
        no_cls = len(classes_categories)

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


def main_cluster(transform, normalisation):
    ds_fd = open(dataset_listings_path, "r")
    datasets = [x.strip() for x in ds_fd.readlines()]

    paths = []
    for dataset in datasets:
        paths.append(cluster_dataset(dataset, transform, normalisation))
    return paths


if __name__ == '__main__':
    paths = main_cluster(Wrap(identic), True)
    print(paths)
