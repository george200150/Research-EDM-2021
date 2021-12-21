import pickle


def get_mean_std(data_pkl):
    a_file = open(data_pkl, "rb")
    mean, stdev = pickle.load(a_file)
    a_file.close()
    return mean, stdev


def get_clustering(data_pkl):
    a_file = open(data_pkl, "rb")
    points, preds, labels = pickle.load(a_file)
    a_file.close()

    return points, preds, labels


def get_ready_for_eval(data_pkl):
    a_file = open(data_pkl, "rb")
    classifiers = pickle.load(a_file)
    a_file.close()
    return classifiers


def get_mask(data_pkl):
    a_file = open(data_pkl, "rb")
    mask = pickle.load(a_file)
    a_file.close()
    return mask


def get_labels_mapping(data_pkl):
    a_file = open(data_pkl, "rb")
    mapping = pickle.load(a_file)
    a_file.close()
    return mapping


def get_paths_list(data_pkl):
    a_file = open(data_pkl, "rb")
    paths_list = pickle.load(a_file)
    a_file.close()
    return paths_list


def dump_data(data, data_pkl):
    a_file = open(data_pkl, "wb")
    pickle.dump(data, a_file, protocol=4)  # allow >4Gb files
    a_file.close()
