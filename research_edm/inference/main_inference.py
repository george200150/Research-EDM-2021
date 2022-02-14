import os

import numpy as np
from tqdm import tqdm

from research_edm.DATA.class_mapping import get_data_type
from research_edm.configs.paths import mapping_dump_base, dset_mean_stdev_dump_base, mask_dump_base, \
    datasets_base_path, inference_dump_base, dataset_listings_path
from research_edm.dataloader.feature_extractor import get_features_labels
from research_edm.inference.model_instantiation import instantiate_default_dryrun, parse_ctor_params, cls_task
from research_edm.io.pickle_io import get_mean_std, dump_data, get_mask, get_labels_mapping
from research_edm.normalisation.postprocessing import default_t


def train_test_split(features, test_size=0.1, slice=0):  # lists received by this function have already been shuffled
    sz = len(features) * test_size
    i = int(sz * slice)
    i_1 = int(sz*(slice+1))
    train = np.concatenate((features[:i], features[i_1:]), axis=0)
    test = features[i:i_1]
    return train, test


def cross_train_model(no_classes, wrapped_model, features, labels, test_size):
    ready_for_eval = []
    for i in tqdm(range(0, 10), desc="k-fold training model {}...".format(wrapped_model.name)):
        x_train, x_test = train_test_split(features, test_size=test_size, slice=i)
        y_train, y_test = train_test_split(labels, test_size=test_size, slice=i)
        if wrapped_model.task_type == cls_task:  # only classification uses one-hot labels
            # TODO: at the moment, the framework does not support training of different gt label mappings (only eval)
            # further implementations would use a remapping of the gt labels, considering no_classes
            wrapped_model.model = wrapped_model.model.fit(x_train, y_train)
            # TODO: naive_bayes - ValueError: y should be a 1d array, got an array of shape (151, 7) instead.
        else:  # regression models must be trained only on the integer labels
            y_train = list([float(x) for x in y_train])
            y_test = list([float(x) for x in y_test])
            wrapped_model.model = wrapped_model.model.fit(x_train, y_train)

        ready_for_eval.append([x_test, y_test, wrapped_model])
    return ready_for_eval


def infer_dataset(no_classes, active_models, cls_cf, reg_cf, dset, transform, norm_flag):
    dset_name = dset.split("/")[-1].split(".")[0]
    data_type = get_data_type(dset_name)
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
        wrapped_models = instantiate_default_dryrun()
    else:
        wrapped_models = parse_ctor_params(cls_cf, reg_cf, active_models)

    paths = []
    for wrapped_model in wrapped_models:
        wrapped_model.set_trained_data_type(data_type)
        if wrapped_model.task_type == cls_task:
            # one-hot encode the labels (only for classification)
            one_hot_labels = lb.transform(labels)
            labels_back = lb.inverse_transform(one_hot_labels)
            assert labels == list(labels_back), "Mapping between one-hot and categorical is not well defined!"
            ready_for_eval = cross_train_model(no_classes, wrapped_model, features, one_hot_labels, test_size=0.1)
        else:
            # multi-collinearity problems for one-hot regression
            ready_for_eval = cross_train_model(no_classes, wrapped_model, features, labels, test_size=0.1)
        wrapped_model.set_pkl_ending(transform, norm_flag)
        dump_path = os.path.join(inference_dump_base, data_type, transform.name,
                                 dset_name + wrapped_model.complete_model_name)
        dump_data(ready_for_eval, dump_path)
        paths.append(dump_path)
    return paths


def main_inference(no_classes, active_models, cls_cf, reg_cf, transform, norm_flag):
    ds_fd = open(dataset_listings_path, "r")
    datasets = [x.strip() for x in ds_fd.readlines()]

    paths = []
    for dataset in datasets:
        path = infer_dataset(no_classes, active_models, cls_cf, reg_cf, dataset, transform, norm_flag)
        paths.append(path)
    return paths


if __name__ == '__main__':
    # paths = main_inference(None, None, default_t, norm_flag=False)
    paths = main_inference(2, None, None, default_t, norm_flag=True)
    # paths = main_inference(5, None, None, default_t, norm_flag=True)
    # paths = main_inference(7, None, None, default_t, norm_flag=True)

    print(paths)
