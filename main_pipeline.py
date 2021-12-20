import random
import os
import numpy as np
import yaml
# import ruamel.yaml as yaml

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
    # torch.manual_seed(seed)
    np.random.seed(seed)


def main_pipeline_unsupervised():
    fix_random_seeds()

    preprocessings = [identic, asinh, log]
    wrap_preprocs = [Wrap(x) for x in preprocessings]
    for fun in wrap_preprocs:
        results_paths = main_cluster(fun)
        main_evaluation(results_paths, "unsupervised")


def main_pipeline_supervised_FIRST_RUN_ONLY(preprocessings, normalisation, active_models, models_configs):
    fix_random_seeds()

    # REMOVE THE 2 LINES BELOW, AS EACH RUN WOULD RESULT IN DIFFERENT RANDOM PATTERNS, MAKING RESULTS INCONSISTENT
    main_norm()  # create the norm/stdev statistics pkls
    main_generate_masks()  # generate a random shuffle that is consistent across experiments
    main_generate_mappings()  # instantiate the categorical to one-hot and back label mappers

    # preprocessings = [identic, asinh, log]  # Datele vor fi trecute prin functia de transformare dupa normalizare.
    # preprocessings = [identic, asinh]  # "log" poate avea probleme, intrucat nu este definit pe numere negative.
    preprocessings = map_preproc_str_to_function(preprocessings)

    # Daca nu se doreste asa ceva, doar "identic" va fi inclus in lista de mai sus, iar restul sterse.
    wrap_preprocs = [Wrap(x) for x in preprocessings]
    for fun in wrap_preprocs:
        results_paths = main_inference(active_models, models_configs, fun, norm_flag=normalisation)
        # norm_flag determina daca se va aplica sau nu normalizarea
        dump_data(results_paths, os.path.join(results_paths_dump_base, paths_filename))
        # Momentan, este implementata doar varianta de normalizare (x - mean) / stdev.
        for paths_to_models in results_paths:
            main_evaluation(paths_to_models, "supervised")
            # Inferenta returneaza path-urile catre clasificatoarele antrenate, salvate in format binar pkl.
            # Evaluarea incarca acele weight-uri folosind path-urile date si apoi face cross-validation.


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


def main_pipeline_supervised_TRAIN_OVERWRITE(preprocessings, normalisation, active_models, models_configs):
    fix_random_seeds()

    # preprocessings = [identic, asinh, log]
    preprocessings = map_preproc_str_to_function(preprocessings)

    wrap_preprocs = [Wrap(x) for x in preprocessings]
    for fun in wrap_preprocs:
        results_paths = main_inference(active_models, models_configs, fun, norm_flag=normalisation)
        # The line above will override the existing trained classifiers.
        # Moreover, the data in the pkls contains the (normalised (and transformed)) datapoints, labels and the models.
        # This is why evaluation does NOT need to know anything about the datasets.
        dump_data(results_paths, os.path.join(results_paths_dump_base, paths_filename))

        for paths_to_models in results_paths:
            main_evaluation(paths_to_models, "supervised")


def main_pipeline_supervised_ONLY_EVAL():
    fix_random_seeds()

    results_paths = get_paths_list(os.path.join(results_paths_dump_base, paths_filename))
    for paths_to_models in results_paths:
        main_evaluation(paths_to_models, "supervised")


def parse_yml():
    with open("supervised_experiment_config.yml", "r") as stream:
        try:
            return yaml.safe_load(stream)  # for vanilla yaml
            # return yaml.load(stream)  # for ruamel yaml
        except yaml.YAMLError as exc:
            print(exc)


def read_parsed_yml():
    # import sklearn
    # yaml = yaml.YAML(typ='safe')
    # yaml.register_class(sklearn.linear_model.LogisticRegression)
    # yaml.register_class(sklearn.naive_bayes.CategoricalNB)
    # yaml.register_class(sklearn.neural_network.MLPClassifier)  # Too beautiful to be true
    yml_dict = parse_yml()
    print(yml_dict)
    setup = yml_dict['experiment']
    preprocessings = setup['preprocessings']
    normalisation = setup['normalisation']
    models = setup['models']
    active_models = models['active']
    models_configs = models['configs']
    return preprocessings, normalisation, active_models, models_configs


if __name__ == '__main__':
    preprocessings, normalisation, active_models, models_configs = read_parsed_yml()

    # main_pipeline_supervised_FIRST_RUN_ONLY(preprocessings, normalisation, active_models, models_configs)
    # creates randomly generated masks, that are consistent cross-experiment

    main_pipeline_supervised_TRAIN_OVERWRITE(preprocessings, normalisation, active_models, models_configs)
    # uses previously generated files; overwrites data, labels and models

    # main_pipeline_supervised_ONLY_EVAL()
    # evaluates the already trained classifiers (double checking only)

    # main_pipeline_unsupervised()
