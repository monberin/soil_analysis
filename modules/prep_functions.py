"""
This module contains helper functions for loading and preprocessing images and csv data for both datasets
"""

import glob
import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold


def create_images_list(directory_path, img_resolution, grayscale=False):
    """
    The function parses over the provided directory, and returns a list of images and a list of image labels.
    The images can be loaded in RGB or in grayscale;
    directory_path: string
    img_resolution: tuple(int)
    grayscale: bool
    """
    images_list = []
    labels_list = []
    for file in glob.glob(directory_path):
        labels_list.append(file.split('/')[-1])
        if grayscale:
            img = cv2.imread(file, 0)  # in grayscale
            img_scaled = cv2.resize(img, img_resolution)  # scale down to set resolution
            images_list.append(img_scaled)
        else:
            imgc = cv2.imread(file, 1)  # in colour
            imgc_scaled = cv2.resize(imgc, img_resolution)  # scale down to set resolution
            images_list.append(imgc_scaled)
    return images_list, labels_list


def extract_gh_values(labels, val_df):
    """
    Helper function for processing the Grieves House dataset, 
    extracting the values for the respective labels;
    returns a list of values.
    labels: list
    val_df: pandas dataframe
    """
    # cleaning up to match the labels from the csv file
    gh_labels = [el.split('_')[0][2:] for el in labels]
    # extracting values from the CSV files
    gh_df = val_df.rename(
        columns={"Sample": "sample", "Stable aggregates (%)": "WSA"})  # for easier access
    gh_values = []
    gh_values = [float(sum(gh_df.loc[gh_df['sample'] == el]['WSA'].values) / len(gh_df.loc[gh_df['sample']
                       == el]['WSA'].values)) for el in gh_labels]  # getting the mean from the spreadsheet
    return gh_values


def extract_lp_values(labels, val_df):
    """
    Helper function for processing the Lower Pilmore dataset, 
    extracting the values for the respective labels;
    returns a list of values.
    labels: list
    val_df: pandas dataframe
    """
    # cleaning up to match the labels from the csv file
    lp_labels = [el.split('_')[0] for el in labels]
    lp_df = val_df.rename(columns={
                          "Sample": "sample", "aggregates stability (%)": "WSA"})  # for easier access
    lp_values = []

    for l in lp_labels:
        val = lp_df.loc[lp_df['sample'] == l]['WSA'].values
        # this skips all values that are in spreadsheet but not in the images
        if len(val) != 0:
            # getting a mean of the values if there are multiple in the spreadsheet
            lp_values.append(float(sum(val) / len(val)))
    return lp_values


def circle_cut_out(img):
    """
    Helper function for cutting out the center circle area from the image (ROI); 
    returns the modified image.
    img: numpy array
    """
    height, width, _ = img.shape
    center = (width // 2, height // 2)
    small_radius = min(width, height) // 2 - 3
    small_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(small_mask, center, small_radius, 255, thickness=-1)
    thr_img = np.array(img, dtype=np.uint8)
    result = cv2.bitwise_and(img, img, mask=small_mask)
    return result


def scale_fit_transform(o_y_train, o_y_val):
    """
    Helper function for fitting and scaling the train values to have the mean of 0 and std of 1,
    and scaling the validation values to the same distribution as the train;
    returns two numpy arrays
    o_y_train: numpy array
    o_y_val: numpy array
    """
    scaler = StandardScaler()
    train = scaler.fit_transform(o_y_train)
    val = scaler.transform(o_y_val)
    return train, val


def extract_transform(vdf, val_name, labels):
    """
    Helper function for processing the Soil Cores dataset, 
    extracting the values for the respective labels for a specified quality indicator;
    returns an array of values.
    labels: list
    vdf: pandas dataframe
    val_name: string
    """
    arr = [float(vdf.loc[vdf['Sample'] == el][val_name].values[0])
           for el in labels]
    data = np.array(arr).reshape(-1, 1)
    return data


def custom_kfold_split(X_list, y_list, sample_names):
    """
    The function generates a list of indexes for k-fold interation;
    Each specified group of samples is together in the same fold;
    returns a list of index tuples.
    X_list: numpy array
    y_list: numpy array
    sample_names: list(string)
    """
    d = {}
    # the labels have a structure [XXXY], where XXX represents the sampling place and Y - the number of the sample;
    # the labels are stripped of the sample number and transformed into the number representation for GroupKFold
    groups = np.array([d.setdefault(x[:-1], len(d)) for x in sample_names])
    kf = GroupKFold(n_splits=5)
    kf.get_n_splits(X_list, y_list, groups)
    splits = []
    for i, (train_index, test_index) in enumerate(kf.split(X_list, y_list, groups)):
        print(f"Fold {i}:")
        print(f"  Train: group={set(groups[train_index])}")
        print(f"  Test:  group={set(groups[test_index])}")
        splits.append((train_index, test_index))

    return splits
