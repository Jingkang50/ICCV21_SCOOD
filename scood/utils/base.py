import numpy as np
import yaml


def load_yaml(path: str):
    with open(path, "r") as file:
        try:
            yaml_file = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    return yaml_file


def sort_array(old_array, index_array):
    sorted_array = np.ones_like(old_array)
    sorted_array[index_array] = old_array
    return sorted_array
