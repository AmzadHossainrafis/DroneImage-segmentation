import numpy as np
import pandas as pd
from DronVid.components.utils.common import read_yaml
from DronVid.components.utils.logger import logger
from tqdm import tqdm


config = read_yaml(r"config\config.yaml")


class DataIjection:

    """
    Data injection is responsible for slicing the data into train, test and validation set.
    default split is 80% train, 10% test and 10% validation.

    args:
        data_dir: path of the data directory
        injection_config: config of the data injection

    return:
     none

    """

    def __init__(
        self,
    ):
        self.data_dir = config["Dir"]["original_data_path"]
        self.injection_config = config["Data_injection"]

    def inject(self, X):
        """
        inject method is responsible for slicing the data into train, test and validation set.
        default split is 80% train, 10% test and 10% validation.

        args:
            X: list of images

        return:
            train: list of training images ids
            test: list of testing images ids
            val: list of validation images ids

        """
        try:
            logger.info("Data injection started ")
            # get split from config
            train_split = self.injection_config["train_split"]
            test_split = self.injection_config["test_split"]
            val_split = self.injection_config["val_split"]
            # add tqdm animation
            logger.info("Data injection started ")
            for i in tqdm(range(100), desc="Data injection in progress"):
                pass

            logger.info("Data injection in progress")
            train, test, val = np.split(
                X, [int(train_split * len(X)), int((train_split + test_split) * len(X))]
            )
            train_ds = pd.DataFrame(train, columns=["id"])
            train_ds.to_csv(self.injection_config["train_csv"], index=False)
            logger.info("train.csv created successfully")
            test_ds = pd.DataFrame(test, columns=["id"])
            test_ds.to_csv(self.injection_config["test_csv"], index=False)
            logger.info("test.csv created successfully")
            val_ds = pd.DataFrame(val, columns=["id"])
            val_ds.to_csv(self.injection_config["val_csv"], index=False)
            logger.info("val.csv created successfully")

            # logger.info("Data injection completed successfully")
            return train, test, val

        except Exception as e:
            # logger.error("Data injection failed")
            # logger.error(e)
            raise e
