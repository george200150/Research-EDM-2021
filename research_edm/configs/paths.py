import os


# TODO: change to desired path
path_to_project = r"C:\Users\George\PycharmProjects\Research-EDM-2021"


results_paths_dump_base = os.path.join(path_to_project, r"results_paths_dump")

mapping_dump_base = os.path.join(path_to_project, r"research_edm\inference\labels_mapping_dump")
base_dump_xlxs = os.path.join(path_to_project, r"research_edm\evaluation\results")

datasets_base_path = os.path.join(path_to_project, r"research_edm\DATA")
dataset_listings_path = os.path.join(path_to_project, r"research_edm\DATA\dataset_listings.txt")
dset_mean_stdev_dump_base = os.path.join(path_to_project, r"research_edm\normalisation\datasets_mean_stdev_dumps")
inference_dump_base = os.path.join(path_to_project, r"research_edm\inference\trained_classifiers")
mask_dump_base = os.path.join(path_to_project, r"research_edm\inference\datasets_shuffle_mask")

paths_filename = "results_paths.pkl"
unsupervised_results_dir = "unsupervised"
supervised_results_dir = "supervised"
