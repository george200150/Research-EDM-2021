from research_edm.configs.paths import labels_yml_path
from research_edm.io.yml_io import parse_yml


yml_dict = parse_yml(labels_yml_path)
print(yml_dict)

labels = yml_dict['labels']
data_types = labels['data_types']
learning = labels['learning']
mappings = labels['mappings']
reverse_mappings = labels['reverse_mappings']

classes_grades = learning['categories']


obj_mapping_2 = mappings['mapping_2']
obj_mapping_5 = mappings['mapping_5']
obj_mapping_7 = mappings['mapping_7']

classes_categories_2 = obj_mapping_2['categories']
classes_categories_5 = obj_mapping_5['categories']
classes_categories_7 = obj_mapping_7['categories']

mapping_2 = obj_mapping_2['definition']
mapping_5 = obj_mapping_5['definition']
mapping_7 = obj_mapping_7['definition']


obj_reverse_mapping_2 = reverse_mappings['reverse_mapping_2']
obj_reverse_mapping_5 = reverse_mappings['reverse_mapping_5']
obj_reverse_mapping_7 = reverse_mappings['reverse_mapping_7']

reverse_mapping_2 = obj_reverse_mapping_2['definition']
reverse_mapping_5 = obj_reverse_mapping_5['definition']
reverse_mapping_7 = obj_reverse_mapping_7['definition']


grades_type = data_types['grades_type']['name']
categories_type = data_types['categories_type']['name']


def get_data_type_of_dataset(no_classes, labels):  # TODO: parametrise the strategy; create a listing of partitions
    if no_classes == 7:  # grades
        classes_categories = classes_categories_7  # TODO: for each partition, assign a lambda mapping function (in yml)
    elif no_classes == 5:  # E V G S F
        classes_categories = classes_categories_5
    elif no_classes == 2:  # P F
        classes_categories = classes_categories_2
    else:
        raise ValueError("No such class mapping!")

    for label in labels:
        if label in classes_categories:
            return categories_type
        elif label in classes_grades:
            return grades_type
    raise ValueError("No labels found!")


def get_data_type(dset_name):  # TODO: use yml config
    if "note" in dset_name:
        return grades_type
    elif "categorii" in dset_name:
        return categories_type
    elif "online_" in dset_name or "traditional_" in dset_name:
        return grades_type
    else:
        raise ValueError("Dataset type not recognised!")


def map_category(no_classes, string_label):  # TODO: use yml config
    if no_classes == 2:
        return mapping_2[string_label]
    if no_classes == 5:
        return mapping_5[string_label]
    if no_classes == 7:
        return mapping_7[string_label]
    raise ValueError("No such class mapping!")


def unmap_category(no_classes, integer_label):  # TODO: use yml config
    integer_label = int(integer_label)  # safe check
    integer_label = min(max(integer_label, 4), 10)

    if no_classes == 2:
        return reverse_mapping_2[integer_label]
    if no_classes == 5:
        return reverse_mapping_5[integer_label]
    if no_classes == 7:
        return integer_label
        # return reverse_mapping_7[integer_label]
    raise ValueError("No such class mapping!")
