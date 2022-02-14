from research_edm.configs.paths import labels_yml_path
from research_edm.io.yml_io import parse_yml


yml_dict = parse_yml(labels_yml_path)
print(yml_dict)

labels = yml_dict['labels']
data_types = labels['data_types']
learning = labels['learning']
mappings = labels['mappings']
reverse_mappings = labels['reverse_mappings']

learning_classes = learning['categories']


# learning and non-learning types
grades_type = data_types['grades_type']['name']
categories_type = data_types['categories_type']['name']


def get_palette(no_classes):
    for mapping in list(mappings.keys()):
        mapping = mappings[mapping]
        if no_classes == mapping['no_classes']:
            return mapping['colors']
    raise ValueError("No such color scheme!")


def get_classes_by_number(no_classes):
    for mapping in list(mappings.keys()):
        mapping = mappings[mapping]
        if no_classes == mapping['no_classes']:
            return mapping['categories']
    raise ValueError("No such class mapping!")


def get_data_type_of_dataset(no_classes, labels):
    classes_categories = None
    for mapping in list(mappings.keys()):
        mapping = mappings[mapping]
        if no_classes == mapping['no_classes']:
            classes_categories = mapping['definition']

    if classes_categories is None:
        raise ValueError("No such class mapping!")

    for label in labels:
        if label in classes_categories:
            return categories_type
        elif label in learning_classes:
            return grades_type
    raise ValueError("No labels found!")


def get_data_type(dset_name):
    for data_type in list(data_types.keys()):
        data_type = data_types[data_type]
        for identifier in data_type['identification']:
            if identifier in dset_name:
                return data_type['name']
    raise ValueError("Dataset type not recognised!")


def map_category(no_classes, string_label):
    for mapping in list(mappings.keys()):
        mapping = mappings[mapping]
        if no_classes == mapping['no_classes']:
            return mapping['definition'][string_label]
    raise ValueError("No such class mapping!")


def unmap_category(no_classes, integer_label):
    integer_label = int(integer_label)  # safe check
    integer_label = min(max(integer_label, min(learning_classes)), max(learning_classes))

    for reverse_mapping in list(reverse_mappings.keys()):
        reverse_mapping = reverse_mappings[reverse_mapping]
        if no_classes == reverse_mapping['no_classes']:
            return reverse_mapping['definition'][integer_label]
    return integer_label
