"""
Code to download or generate data
"""

from pathlib import Path

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

from astropy.table import Table
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def main(
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
): ...


def load_data(file_name: Path = "image_table_qso.hdf5"):
    input_path: Path = RAW_DATA_DIR
    data = Table.read(Path.joinpath(input_path, file_name))

    # every img is 36x36px, each px is coded in 4 band values - u(ltraviolet), g(reen), r(ed), i(nfrared)
    images = np.array(data["image"]).reshape(
        (-1, 36, 36, 4)
    )  # reshape not neeeded in this cause, only 4 precaution
    labels = np.array(data["SPECTYPE"]).reshape((-1))

    return images, labels


def preprocess_data(images: np.ndarray, labels: np.ndarray):
    # asinh scalling
    images = np.arcsinh(images)

    # min max normalization
    scaler = MinMaxScaler()
    scaler.fit(images)

    return images, labels


def split_data(
    images: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
):
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        images,
        labels,
        test_size=test_size,
        random_state=random_state,
    )

    val_relative_size = val_size / (1 - test_size)

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=val_relative_size,
        random_state=random_state,
    )

    return x_train, x_val, x_test, y_train, y_val, y_test


if __name__ == "__main__":
    label_to_id = {"qso": 0, "star": 1, "galaxy": 2}
    imgs_qso, labels_qso = load_data()
    imgs_galaxy, labels_galaxy = load_data("image_table_galaxy.hdf5")
    imgs_star, labels_star = load_data("image_table_star.hdf5")
    imgs = np.concatenate(imgs_qso, imgs_galaxy, imgs_star)
    labels = np.concatenate(labels_qso, labels_galaxy, labels_star)
    labels = np.array([label_to_id[label.lower()] for label in labels], dtype=np.int64)

    x_train, x_val, x_test, y_train, y_val, y_test = split_data(imgs, labels)
    x_train, y_train = preprocess_data(x_train, y_train)
    x_val, y_val = preprocess_data(x_val, y_val)
    x_test, y_test = preprocess_data(x_test, y_test)

    print("Train:", x_train.shape, y_train.shape)
    print("Val:", x_val.shape, y_val.shape)
    print("Test:", x_test.shape, y_test.shape)
