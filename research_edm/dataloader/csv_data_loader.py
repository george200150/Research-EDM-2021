from research_edm.normalisation.postprocessing import default_t


class CsvDataLoader:
    def __init__(self, data_file, transform=default_t, mean=None, stdev=None, normalise=False, num_images=None):
        csv = open(data_file)
        lines = csv.readlines()
        self.data_size = len(lines) - 1
        data = [list(x.strip().split(",")) for x in lines[1:]]
        data_dicts = []
        for d in data:
            ddict = {}
            norm_data = None
            try:
                norm_data = [float(x) for x in d[1:]]
                if normalise and mean is not None and stdev is not None:
                    for i in range(len(norm_data)):
                        norm_data[i] = (norm_data[i] - mean[i]) / stdev[i]
                ddict = {"label": d[0],
                         "features": [transform(x) for x in norm_data]}
            except Exception as e:
                print(e)
                print(norm_data)
            data_dicts.append(ddict)
        self.data = data_dicts

        if num_images is not None:
            self.data_size = num_images
            self.data = self.data[:num_images]

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        d = self.data[index]
        return d["label"], d["features"]
