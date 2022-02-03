import os

from tqdm import tqdm
import numpy as np

from research_edm.configs.paths import dataset_listings_path, datasets_base_path, dset_mean_stdev_dump_base
from research_edm.dataloader.csv_data_loader import CsvDataLoader
from research_edm.io.pickle_io import dump_data


def main_norm():
    ds_fd = open(dataset_listings_path, "r")
    datasets = [x.strip() for x in ds_fd.readlines()]

    for dataset in datasets:
        dataloader = CsvDataLoader(os.path.join(datasets_base_path, dataset))

        features = None
        dset_name = dataset.split("/")[-1].split(".")[0]
        for idx, batch in enumerate(tqdm(dataloader, desc='Iterating over dataset {}...'.format(dset_name))):
            _, f = batch
            f = np.asarray([f], dtype=float)

            if features is not None:
                features = np.concatenate((features, f))
            else:
                features = f

        mean = np.mean(features, axis=0)
        stdev = np.std(features, axis=0)

        dataset_mean_stdev_dump_file = "{}_mean_stdev.pkl".format(dset_name)
        dump_data([mean, stdev], os.path.join(dset_mean_stdev_dump_base, dataset_mean_stdev_dump_file))


if __name__ == '__main__':
    main_norm()
