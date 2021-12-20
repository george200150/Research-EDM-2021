import os
import ast

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, SGDRegressor, TweedieRegressor, LinearRegression
from sklearn.naive_bayes import CategoricalNB
from sklearn.neural_network import MLPClassifier

from research_edm.dataloader.feature_extractor import get_features_labels
from research_edm.io.pickle_io import get_mean_std, dump_data, get_mask, get_labels_mapping
from research_edm.normalisation.postprocessing import Wrap, identic
from research_edm.configs.paths import mapping_dump_base, dset_mean_stdev_dump_base, mask_dump_base, \
    datasets_base_path, inference_dump_base, dataset_listings_path


def cross_train_model(model_meta, features, labels, test_size):
    ready_for_eval = []
    model, meta = model_meta
    for i in range(0, 10):
        print("---------------------------------" + str(i) + "---------------------------------------")
        x_train, x_test = train_test_split(features, test_size=test_size, random_state=i)
        y_train, y_test = train_test_split(labels, test_size=test_size, random_state=i)
        if meta == "classification":  # TODO: could wrap classifiers and regressors into a "MODEL_WRAPPER" class
            trained_classifier = model.fit(x_train, y_train)
        else:
            trained_classifier = model.fit(x_train, list([float(x) for x in y_train]))
            # TODO: NOT CATEGORIES (needs mapping)
        ready_for_eval.append([x_test, y_test, trained_classifier])
    return ready_for_eval


def parse_ctor_params(models_configs):
    mlp_config = models_configs['mlp']
    nb_config = models_configs['nb']
    lr_config = models_configs['lr']

    sgdr_config = models_configs['sgdr']
    tr_config = models_configs['tr']
    poly_config = models_configs['poly']

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

    nb = CategoricalNB(alpha=nb_config['alpha'],
                       fit_prior=nb_config['fit_prior'],
                       class_prior=nb_config['class_prior'],
                       min_categories=nb_config['min_categories'])

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

    tr = TweedieRegressor(power=tr_config['power'],
                          alpha=tr_config['alpha'],
                          fit_intercept=tr_config['fit_intercept'],
                          link=tr_config['link'],
                          max_iter=tr_config['max_iter'],
                          tol=tr_config['tol'],
                          warm_start=tr_config['warm_start'],
                          verbose=tr_config['verbose'])
    poly = Pipeline([
        ('poly', PolynomialFeatures(degree=poly_config['degree'])),
        ('linear', LinearRegression(fit_intercept=False))])

    return mlp, nb, lr, sgdr, tr, poly


def get_active_models(models_dict, active_classifiers):
    no_cls = 3
    no_reg = 3

    models_and_meta = [[] for _ in range(no_cls + no_reg)]  # empty list for each model

    if 'mlp' in active_classifiers:
        models_and_meta[0] = [models_dict['mlp'], 'classification']
    if 'nb' in active_classifiers:
        models_and_meta[1] = [models_dict['nb'], 'classification']
    if 'lr' in active_classifiers:
        models_and_meta[2] = [models_dict['lr'], 'classification']

    if 'sgdr' in active_classifiers:
        models_and_meta[no_cls + 0] = [models_dict['sgdr'], 'regression']
    if 'tr' in active_classifiers:
        models_and_meta[no_cls + 1] = [models_dict['tr'], 'regression']
    if 'poly' in active_classifiers:
        models_and_meta[no_cls + 2] = [models_dict['poly'], 'regression']
    return models_and_meta


def infer_dataset(active_models, models_configs, dset, transform, norm_flag):
    dset_name = dset.split("/")[-1].split(".")[0]
    data_type = "grades" if "note" in dset_name else "categories"
    mean_stdev_pkl_name = "{}_mean_stdev.pkl".format(dset_name)

    lb = get_labels_mapping(os.path.join(mapping_dump_base, data_type, dset_name + ".pkl"))

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

    features = features[mask]
    labels = list(np.asarray(labels)[mask])

    # instantiate the models
    if active_models is None or len(active_models) == 0:
        mlp = MLPClassifier(max_iter=5)
        # nb = CategoricalNB()
        # lr = LogisticRegression(max_iter=5)
        models_and_metadata = [[mlp, 'classification']]
    else:
        mlp, nb, lr, sgdr, tr, poly = parse_ctor_params(models_configs)
        models_dict = {'mlp': mlp, 'nb': nb, 'lr': lr, 'sgdr': sgdr, 'tr': tr, 'poly': poly}
        models_and_metadata = get_active_models(models_dict, active_models)

    model_names = ["_mlp.pkl", "_nb.pkl", "_lr.pkl", "_sgdr.pkl", "_tr.pkl", "_poly.pkl"]  # TODO: issue  when not enough models => shifted names
    model_names = ["_" + transform.name + x for x in model_names]
    if norm_flag:
        model_names = ["_norm" + x for x in model_names]

    # one-hot encode the labels (only for classification)
    one_hot_labels = lb.transform(labels)
    labels_back = lb.inverse_transform(one_hot_labels)
    assert labels == list(labels_back), "Mapping between one-hot and categorical is not well defined!"

    paths = []
    for idx, model_meta in enumerate(models_and_metadata):
        try:
            model, meta = model_meta
            if meta == 'classification':# TODO: could wrap classifiers and regressors into a "MODEL_WRAPPER" class
                ready_for_eval = cross_train_model(model_meta, features, one_hot_labels, test_size=0.1)
            else:  # multicollinearity problems for one-hot regression
                ready_for_eval = cross_train_model(model_meta, features, labels, test_size=0.1)
            dump_path = os.path.join(inference_dump_base, data_type, transform.name, dset_name + model_names[idx])  # TODO: issue  when not enough models => shifted names
            dump_data(ready_for_eval, dump_path)
            paths.append(dump_path)
        except ValueError as e:
            print(e)
            pass
    return paths


def main_inference(active_models, classifiers_configs, transform, norm_flag):
    ds_fd = open(dataset_listings_path, "r")
    datasets = [x.strip() for x in ds_fd.readlines()]

    paths = []
    for dataset in datasets:
        path = infer_dataset(active_models, classifiers_configs, dataset, transform, norm_flag)
        paths.append(path)
    return paths


if __name__ == '__main__':
    # paths = main_inference(None, None, Wrap(identic), norm_flag=False)
    paths = main_inference(None, None, Wrap(identic), norm_flag=True)

    print(paths)
