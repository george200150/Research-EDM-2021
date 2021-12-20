import umap
# conda install -c conda-forge umap-learn


class Umap:
    def __init__(self):
        pass

    def cluster(self, xs):
        return umap.UMAP(n_neighbors=15,
                         n_components=2,  # n_components=3,
                         metric="euclidean",
                         metric_kwds=None,
                         output_metric="euclidean",
                         output_metric_kwds=None,
                         n_epochs=None,
                         learning_rate=1.0,
                         init="spectral",
                         min_dist=0.1,
                         spread=1.0,
                         low_memory=True,
                         n_jobs=-1,
                         set_op_mix_ratio=1.0,
                         local_connectivity=1.0,
                         repulsion_strength=1.0,
                         negative_sample_rate=5,
                         transform_queue_size=4.0,
                         a=None,
                         b=None,
                         random_state=None,
                         angular_rp_forest=False,
                         target_n_neighbors=-1,
                         target_metric="categorical",
                         target_metric_kwds=None,
                         target_weight=0.5,
                         transform_seed=42,
                         transform_mode="embedding",
                         force_approximation_algorithm=False,
                         verbose=False,
                         unique=False,
                         densmap=False,
                         dens_lambda=2.0,
                         dens_frac=0.3,
                         dens_var_shift=0.1,
                         output_dens=False,
                         disconnection_distance=None).fit_transform(xs)  # random_state=42
