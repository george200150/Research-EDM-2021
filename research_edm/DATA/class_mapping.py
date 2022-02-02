classes_grades = ['4', '5', '6', '7', '8', '9', '10']
# classes_categories = ["E", "G", "S", "F"]
classes_categories = ["E", "V", "G", "S", "F"]

classes_categories_2 = ["P", "F"]
classes_categories_5 = ["E", "V" "G", "S", "F"]
classes_categories_7 = ["10", "9", "8", "7", "6", "5", "4"]


mapping = {"E": 10.0, "V": 9.0, "G": 8.0, "S": 6.0, "F": 4.0}

mapping_2 = {"P": 10.0, "F": 1.0}
mapping_5 = {"E": 10.0, "V": 9.0, "G": 8.0, "S": 6.0, "F": 4.0}
mapping_7 = {"E": 10.0, "V": 9.0, "G": 8.0, "S": 6.0, "F": 4.0}

reverse_mapping = {10: "E", 9: "E", 8: "G", 7: "G", 6: "S", 5: "S", 4: "F", 3: "F", 2: "F", 1: "F"}

reverse_mapping_2 =  {10: "P", 9: "P", 8: "P", 7: "P", 6: "P", 5: "P", 4: "F", 3: "F", 2: "F", 1: "F"}
reverse_mapping_5 =  {10: "E", 9: "V", 8: "G", 7: "G", 6: "S", 5: "S", 4: "F", 3: "F", 2: "F", 1: "F"}  # TODO: check if consistent with paper
reverse_mapping_7 = {10: "10", 9: "9", 8: "8", 7: "7", 6: "6", 5: "5", 4: "4", 3: "4", 2: "4", 1: "4"}


grades_type = "grades"
categories_type = "categories"


def get_data_type_of_dataset(labels):
    for label in labels:
        if label in classes_categories:
            return categories_type
        elif label in classes_grades:
            return grades_type


def get_data_type(dset_name):
    data_type = grades_type if "note" in dset_name else categories_type
    return data_type


def map_category(string_label):
    return mapping[string_label]


def unmap_category(integer_label):
    if integer_label > 10:
        return "E"
    if integer_label < 1:
        return "F"
    return reverse_mapping[integer_label]
