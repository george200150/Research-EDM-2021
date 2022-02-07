from skrebate import ReliefF
from tqdm import tqdm
import numpy as np

from research_edm.dataloader.csv_data_loader import CsvDataLoader


def get_features_labels(data_file, transform, mean=None, stdev=None, normalise=False, num_images=None):
    dataset = CsvDataLoader(data_file=data_file, transform=transform, mean=mean, stdev=stdev, normalise=normalise, num_images=num_images)
    dataloader = dataset

    features = None
    labels = []
    for idx, batch in enumerate(tqdm(dataloader, desc='Loading features...')):
        cls, f = batch
        f = np.asarray([f], dtype=float)

        assert len(labels) == (len(features) if features is not None else 0)
        labels.append(cls)

        current_features = f
        if features is not None:
            features = np.concatenate((features, current_features))
        else:
            features = current_features

    # TODO: add relief-based feature extraction
    # features = ReliefF(n_features_to_select=10, n_neighbors=150).fit_transform(features, labels)

    return features, labels
