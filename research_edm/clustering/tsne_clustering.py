from sklearn.manifold import TSNE


class Tsne:
    def __init__(self):
        pass

    def cluster(self, xs):
        return TSNE(n_components=2,
                    perplexity=30.0,
                    early_exaggeration=12.0,
                    learning_rate=200.0,
                    n_iter=1000,
                    n_iter_without_progress=300,
                    min_grad_norm=1e-7,
                    metric="euclidean",
                    init="random",
                    verbose=0,
                    random_state=None,
                    method='barnes_hut',
                    angle=0.5,
                    n_jobs=None,
                    square_distances='legacy').fit_transform(xs)
