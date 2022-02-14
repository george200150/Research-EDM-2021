from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, SGDRegressor, TweedieRegressor, LinearRegression
from sklearn.naive_bayes import CategoricalNB
from sklearn.neural_network import MLPClassifier

from umap import UMAP
from sklearn.manifold import TSNE


cls_task = "classification"
reg_task = "regression"
clustering_tasks = "clustering"


class ModelWrapper:
    def __init__(self, sklearn_model, model_name, task_type, data_type=None):
        self.model = sklearn_model
        self.task_type = task_type
        self.name = model_name
        self.complete_model_name = None
        self.png_name = None
        self.data_type = data_type

    def set_pkl_ending(self, transform, norm_flag):
        formatted_name = "_" + self.name + ".pkl"
        added_transform_name = "_" + transform.name + formatted_name
        if norm_flag:
            self.complete_model_name = "_norm" + added_transform_name

    def set_png_ending(self, kmeans=False):
        formatted_name = "_" + self.name
        if kmeans:
            formatted_name = formatted_name + "_kmeans"
        self.png_name = formatted_name + ".png"

    def set_trained_data_type(self, data_type):
        self.data_type = data_type


def get_active_models(wrapped_models, active_models):
    active_wrapped_models = [x for x in wrapped_models if x.name in active_models]
    return active_wrapped_models


def instantiate_clustering_dryrun():
    umap = UMAP(n_components=2)
    return [ModelWrapper(umap, 'umap', cls_task)]


def instantiate_default_dryrun():
    mlp = MLPClassifier(max_iter=5)
    return [ModelWrapper(mlp, 'mlp', cls_task)]


def parse_cluster_params(models_configs, active_models):
    umap_config = models_configs['umap']
    tnse_config = models_configs['t-sne']

    models = []

    umap = UMAP(**umap_config)
    models.append(ModelWrapper(umap, 'umap', clustering_tasks))

    tsne = TSNE(**tnse_config)
    models.append(ModelWrapper(tsne, 't-sne', clustering_tasks))

    return get_active_models(models, active_models)


def parse_ctor_params(cls_cf, reg_cf, active_models):
    models = []

    for klass_name, value_config in list(cls_cf.items()):
        klass = globals()[klass_name]
        models.append(ModelWrapper(klass(**value_config), klass_name, task_type=cls_task))

    for klass_name, value_config in list(reg_cf.items())[:-1]:  # Poly is different
        klass = globals()[klass_name]
        models.append(ModelWrapper(klass(**value_config), klass_name, task_type=reg_task))

    poly_config = reg_cf['Poly']  # TODO: need additional care...
    poly = Pipeline([
        ('Poly', PolynomialFeatures(**poly_config)),
        ('Linear', LinearRegression(fit_intercept=False))])
    models.append(ModelWrapper(poly, 'Poly', reg_task))

    return get_active_models(models, active_models)
