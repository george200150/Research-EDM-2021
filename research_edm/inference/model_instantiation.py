import ast

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, SGDRegressor, TweedieRegressor, LinearRegression
from sklearn.naive_bayes import CategoricalNB
from sklearn.neural_network import MLPClassifier


cls_task = "classification"
reg_task = "regression"

# TODO: add grades / categories information here

class ModelWrapper:
    def __init__(self, sklearn_model, model_name, task_type, data_type=None):
        self.model = sklearn_model
        self.task_type = task_type
        self.name = model_name
        self.complete_model_name = None
        self.data_type = data_type  # grades / categories

    def set_pkl_ending(self, transform, norm_flag):
        formatted_name = "_" + self.name + ".pkl"
        added_transform_name = "_" + transform.name + formatted_name
        if norm_flag:
            self.complete_model_name = "_norm" + added_transform_name

    def set_trained_data_type(self, data_type):
        self.data_type = data_type


def get_active_models(wrapped_models, active_models):
    active_wrapped_models = [x for x in wrapped_models if x.name in active_models]
    return active_wrapped_models


def instantiate_default_dryrun():
    mlp = MLPClassifier(max_iter=5)
    return ModelWrapper(mlp, 'mlp', cls_task)


def parse_ctor_params(models_configs, active_models):
    mlp_config = models_configs['mlp']
    nb_config = models_configs['nb']
    lr_config = models_configs['lr']

    sgdr_config = models_configs['sgdr']
    tr_config = models_configs['tr']
    poly_config = models_configs['poly']

    models = []

    mlp = MLPClassifier(hidden_layer_sizes=ast.literal_eval(mlp_config['hidden_layer_sizes']),
                        activation=mlp_config['activation'],
                        solver=mlp_config['solver'],
                        alpha=mlp_config['alpha'],
                        batch_size=mlp_config['batch_size'],
                        learning_rate=mlp_config['learning_rate'],
                        learning_rate_init=mlp_config['learning_rate_init'],
                        power_t=mlp_config['power_t'],
                        max_iter=mlp_config['max_iter'],
                        shuffle=mlp_config['shuffle'],
                        random_state=mlp_config['random_state'],
                        tol=float(mlp_config['tol']),
                        verbose=mlp_config['verbose'],
                        warm_start=mlp_config['warm_start'],
                        momentum=mlp_config['momentum'],
                        nesterovs_momentum=mlp_config['nesterovs_momentum'],
                        early_stopping=mlp_config['early_stopping'],
                        validation_fraction=mlp_config['validation_fraction'],
                        beta_1=mlp_config['beta_1'],
                        beta_2=mlp_config['beta_2'],
                        epsilon=float(mlp_config['epsilon']),
                        n_iter_no_change=mlp_config['n_iter_no_change'],
                        max_fun=mlp_config['max_fun'])
    models.append(ModelWrapper(mlp, 'mlp', cls_task))

    nb = CategoricalNB(alpha=nb_config['alpha'],
                       fit_prior=nb_config['fit_prior'],
                       class_prior=nb_config['class_prior'],
                       min_categories=nb_config['min_categories'])
    models.append(ModelWrapper(nb, 'nb', cls_task))

    lr = LogisticRegression(penalty=lr_config['penalty'],
                            dual=lr_config['dual'],
                            tol=float(lr_config['tol']),
                            C=lr_config['C'],
                            fit_intercept=lr_config['fit_intercept'],
                            intercept_scaling=lr_config['intercept_scaling'],
                            class_weight=lr_config['class_weight'],
                            random_state=lr_config['random_state'],
                            solver=lr_config['solver'],
                            max_iter=lr_config['max_iter'],
                            multi_class=lr_config['multi_class'],
                            verbose=lr_config['verbose'],
                            warm_start=lr_config['warm_start'],
                            n_jobs=lr_config['n_jobs'],
                            l1_ratio=lr_config['l1_ratio'])
    models.append(ModelWrapper(lr, 'lr', cls_task))

    sgdr = SGDRegressor(loss=sgdr_config['loss'],
                        penalty=sgdr_config['penalty'],
                        alpha=sgdr_config['alpha'],
                        l1_ratio=sgdr_config['l1_ratio'],
                        fit_intercept=sgdr_config['fit_intercept'],
                        max_iter=sgdr_config['max_iter'],
                        tol=sgdr_config['tol'],
                        shuffle=sgdr_config['shuffle'],
                        verbose=sgdr_config['verbose'],
                        epsilon=sgdr_config['epsilon'],
                        random_state=sgdr_config['random_state'],
                        learning_rate=sgdr_config['learning_rate'],
                        eta0=sgdr_config['eta0'],
                        power_t=sgdr_config['power_t'],
                        early_stopping=sgdr_config['early_stopping'],
                        validation_fraction=sgdr_config['validation_fraction'],
                        n_iter_no_change=sgdr_config['n_iter_no_change'],
                        warm_start=sgdr_config['warm_start'],
                        average=sgdr_config['average'])
    models.append(ModelWrapper(sgdr, 'sgdr', reg_task))

    tr = TweedieRegressor(power=tr_config['power'],
                          alpha=tr_config['alpha'],
                          fit_intercept=tr_config['fit_intercept'],
                          link=tr_config['link'],
                          max_iter=tr_config['max_iter'],
                          tol=tr_config['tol'],
                          warm_start=tr_config['warm_start'],
                          verbose=tr_config['verbose'])
    models.append(ModelWrapper(tr, 'tr', reg_task))

    poly = Pipeline([
        ('poly', PolynomialFeatures(degree=poly_config['degree'])),
        ('linear', LinearRegression(fit_intercept=False))])
    models.append(ModelWrapper(poly, 'poly', reg_task))

    return get_active_models(models, active_models)
