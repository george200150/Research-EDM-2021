import os
import glob

from research_edm.configs.paths import results_paths_dump_base, base_dump_xlxs, mapping_dump_base, \
    dset_mean_stdev_dump_base, inference_dump_base, mask_dump_base, supervised_results_dir

type_folders = ["categories", "grades"]
preproc_folders = ["asinh", "identic", "log"]

all_save_paths = []
for type_folder in type_folders:
    for preproc_folders in preproc_folders:
        all_save_paths.append(os.path.join(base_dump_xlxs, supervised_results_dir, type_folder, preproc_folders))
        all_save_paths.append(os.path.join(inference_dump_base, type_folder, preproc_folders))
        pass
    all_save_paths.append(os.path.join(mapping_dump_base, type_folder))
    all_save_paths.append(os.path.join(mask_dump_base, type_folder))
    pass

all_save_paths.append(results_paths_dump_base)
all_save_paths.append(dset_mean_stdev_dump_base)


def main_delete():
    for path in all_save_paths:
        files = glob.glob(path + "/*")
        for f in files:
            os.remove(f)  # TODO: research_edm/inference/trained_classifiers/grades/identic is not seen ???
            print(f)


if __name__ == '__main__':
    main_delete()
