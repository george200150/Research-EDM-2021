import random
import os
import numpy as np
import yaml

from research_edm.clustering.main_clustering import main_cluster
from research_edm.configs.paths import results_paths_dump_base, paths_filename
from research_edm.evaluation.main_evaluation import main_evaluation
from research_edm.inference.create_onehot_categorical_labels_mapping import main_generate_mappings
from research_edm.inference.generate_dataset_shuffle_masks import main_generate_masks
from research_edm.inference.main_inference import main_inference
from research_edm.io.pickle_io import dump_data, get_paths_list
from research_edm.normalisation.main_get_mean_stdev import main_norm
from research_edm.normalisation.postprocessing import Wrap, identic, asinh, log


def fix_random_seeds():
    seed = 10
    random.seed(seed)
    np.random.seed(seed)


def main_pipeline_unsupervised(no_classes, preprocessings, normalisation, savefig,
                               fresh_start, active_unsupervised_models, unsupervised_models_configs):
    fix_random_seeds()

    preprocessings = map_preproc_str_to_function(preprocessings)
    wrap_preprocs = [Wrap(x) for x in preprocessings]
    for fun in wrap_preprocs:
        results_paths = main_cluster(fun, normalisation, savefig,
                                     fresh_start, active_unsupervised_models, unsupervised_models_configs)
        main_evaluation(no_classes, results_paths, "unsupervised")


def main_pipeline_supervised_FIRST_RUN_ONLY(no_classes, preprocessings, normalisation, active_models, models_configs):
    fix_random_seeds()

    main_norm()
    main_generate_masks()
    main_generate_mappings()

    preprocessings = map_preproc_str_to_function(preprocessings)

    wrap_preprocs = [Wrap(x) for x in preprocessings]
    for fun in wrap_preprocs:
        results_paths = main_inference(no_classes, active_models, models_configs, fun, norm_flag=normalisation)
        dump_data(results_paths, os.path.join(results_paths_dump_base, paths_filename))
        # Momentan, este implementata doar varianta de normalizare (x - mean) / stdev.
        for paths_to_models in results_paths:
            main_evaluation(no_classes, paths_to_models, "supervised")


def map_preproc_str_to_function(preprocessings):
    res = []
    for preproc in preprocessings:
        if preproc == 'asinh':
            res.append(asinh)
        if preproc == 'identic':
            res.append(identic)
        if preproc == 'log':
            res.append(log)

    return res


def main_pipeline_supervised_TRAIN_OVERWRITE(no_classes, preprocessings, normalisation, active_models, models_configs):
    fix_random_seeds()

    preprocessings = map_preproc_str_to_function(preprocessings)

    wrap_preprocs = [Wrap(x) for x in preprocessings]
    for fun in wrap_preprocs:
        results_paths = main_inference(no_classes, active_models, models_configs, fun, norm_flag=normalisation)
        dump_data(results_paths, os.path.join(results_paths_dump_base, paths_filename))

        for paths_to_models in results_paths:
            main_evaluation(no_classes, paths_to_models, "supervised")


def main_pipeline_supervised_ONLY_EVAL(no_classes):
    fix_random_seeds()

    results_paths = get_paths_list(os.path.join(results_paths_dump_base, paths_filename))
    for paths_to_models in results_paths:
        main_evaluation(no_classes, paths_to_models, "supervised")


def parse_yml():
    with open("supervised_experiment_config.yml", "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def read_parsed_yml():
    yml_dict = parse_yml()
    print(yml_dict)

    experiment = yml_dict['experiment']
    no_classes = experiment['no_classes']
    preprocessings = experiment['preprocessings']
    normalisation = experiment['normalisation']

    experiment_supervised = experiment['supervised']
    supervised_models = experiment_supervised['models']
    active_supervised_models = supervised_models['active']
    supervised_models_configs = supervised_models['configs']

    experiment_unsupervised = experiment['unsupervised']
    fresh_start = experiment_unsupervised['fresh_start']
    savefig = experiment_unsupervised['savefig']
    unsupervised_models = experiment_unsupervised['models']
    active_unsupervised_models = unsupervised_models['active']
    unsupervised_models_configs = unsupervised_models['configs']

    return no_classes, preprocessings, normalisation, active_supervised_models, supervised_models_configs, savefig,\
           fresh_start, active_unsupervised_models, unsupervised_models_configs


if __name__ == '__main__':
    no_classes, preprocessings, normalisation, active_supervised_models, supervised_models_configs, savefig, \
    fresh_start, active_unsupervised_models, unsupervised_models_configs = read_parsed_yml()

    # main_pipeline_supervised_FIRST_RUN_ONLY(no_classes, preprocessings, normalisation, active_supervised_models, supervised_models_configs)
    # creates randomly generated masks, that are consistent cross-experiment

    main_pipeline_supervised_TRAIN_OVERWRITE(no_classes, preprocessings, normalisation, active_supervised_models, supervised_models_configs)
    # uses previously generated files; overwrites data, labels and models

    # main_pipeline_supervised_ONLY_EVAL(no_classes)
    # evaluates the already trained classifiers (double checking only)

    # main_pipeline_unsupervised(no_classes, preprocessings, normalisation, savefig, fresh_start, active_unsupervised_models, unsupervised_models_configs)
