import yaml
import os
import pandas as pd
import numpy as np



def read_yaml(yaml_path):
    """
    Read yaml file and return a dictionary
    """
    with open(yaml_path, "r") as yaml_file:
        yaml_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)
    return yaml_dict




def create_df(IMAGE_PATH):
    name = []
    mask = []
    for dirname, _, filenames in os.walk(IMAGE_PATH): # given a directory iterates over the files
        for filename in filenames:
            f = filename.split('.')[0]
            name.append(f)

    return pd.DataFrame({'id': name}, index = np.arange(0, len(name))).sort_values('id').reset_index(drop=True)
