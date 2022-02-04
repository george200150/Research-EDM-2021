classes_grades = [4, 5, 6, 7, 8, 9, 10]

classes_categories_2 = ["P", "F"]
classes_categories_4 = ["E", "G", "S", "F"]
classes_categories_5 = ["E", "V", "G", "S", "F"]
classes_categories_7 = [10, 9, 8, 7, 6, 5, 4]


mapping_2 = {"P": 10.0, "F": 1.0}
mapping_4 = {"E": 10.0, "G": 8.0, "S": 6.0, "F": 4.0}
mapping_5 = {"E": 10.0, "V": 9.0, "G": 8.0, "S": 6.0, "F": 4.0}
mapping_7 = {10: 10.0, 9: 9.0, 8: 8.0, 7: 7.0, 6: 6.0, 5: 5.0, 4: 4.0}

reverse_mapping_2 = {10: "P", 9: "P", 8: "P", 7: "P", 6: "P", 5: "P", 4: "F", 3: "F", 2: "F", 1: "F"}
reverse_mapping_4 = {10: "E", 9: "G", 8: "G", 7: "G", 6: "S", 5: "S", 4: "F", 3: "F", 2: "F", 1: "F"}
reverse_mapping_5 = {10: "E", 9: "V", 8: "G", 7: "G", 6: "S", 5: "S", 4: "F", 3: "F", 2: "F", 1: "F"}
reverse_mapping_7 = {10: "10", 9: "9", 8: "8", 7: "7", 6: "6", 5: "5", 4: "4", 3: "4", 2: "4", 1: "4"}

post_proc_remap_5 = {"E": "E", "V": "V", "G": "G", "S": "S", "F": "F"}
post_proc_remap_4 = {"E": "E", "V": "G", "G": "G", "S": "S", "F": "F"}
post_proc_remap_2 = {"E": "P", "V": "P", "G": "P", "S": "P", "F": "F"}

grades_type = "grades"
categories_type = "categories"


def get_data_type_of_dataset(no_classes, labels):
    if no_classes == 7:  # grades
        classes_categories = classes_categories_7
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


def get_data_type(dset_name):
    if "note" in dset_name:
        return grades_type
    elif "categorii" in dset_name:
        return categories_type
    elif "online_" in dset_name or "traditional_" in dset_name:
        return grades_type
    else:
        raise ValueError("Dataset type not recognised!")


def map_category(no_classes, string_label):
    if no_classes == 2:
        return mapping_2[string_label]
    if no_classes == 5:
        return mapping_5[string_label]
    if no_classes == 7:
        return mapping_7[string_label]
    raise ValueError("No such class mapping!")


def unmap_category(no_classes, integer_label):
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
