import glob
import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

def create_images_list(directory_path, img_resolution,grayscale=False):
  images_list = []
  labels_list = []
  for file in glob.glob(directory_path):
    labels_list.append(file.split('/')[-1])
    if grayscale:
      img = cv2.imread(file,0) #in grayscale
      img_scaled = cv2.resize(img, img_resolution) #scale down to 256
      images_list.append(img_scaled)
    else:
      imgc = cv2.imread(file,1) #in colour
      imgc_scaled = cv2.resize(imgc, img_resolution) #scale down to 256
      images_list.append(imgc_scaled)
  return images_list,labels_list

def extract_gh_values(labels,val_df):
  gh_labels = [el.split('_')[0][2:] for el in labels] # cleaning up to match the labels from the csv file
  # extracting values from the CSV files
  gh_df = val_df.rename(columns={"Sample": "sample", "Stable aggregates (%)" : "WSA"}) #for easier access
  gh_values = []
  gh_values = [float(sum(gh_df.loc[gh_df['sample'] == el]['WSA'].values) / len(gh_df.loc[gh_df['sample'] == el]['WSA'].values)) for el in gh_labels] # getting the mean from the spreadsheet
  return gh_values

def extract_lp_values(labels,val_df):
  lp_labels = [el.split('_')[0] for el in labels] # cleaning up to match the labels from the csv file
  lp_df = val_df.rename(columns={"Sample": "sample", "aggregates stability (%)" : "WSA"}) #for easier access
  lp_values = []

  for l in lp_labels:
    val = lp_df.loc[lp_df['sample'] == l]['WSA'].values
    #this skips all values that are in spreadsheet but not in the images
    if len(val) != 0:
      lp_values.append(float(sum(val) / len(val))) # getting a mean of the values if there are multiple in the spreadsheet
  return lp_values

def circle_cut_out(img):
  height, width, _ = img.shape
  center = (width // 2, height // 2)
  small_radius = min(width, height) // 2 - 3
  small_mask = np.zeros((height, width), dtype=np.uint8)
  cv2.circle(small_mask, center, small_radius, 255, thickness=-1)
  thr_img = np.array(img, dtype=np.uint8)
  result = cv2.bitwise_and(img, img, mask=small_mask)
  return result

def scale_fit_transform(o_y_train,o_y_val):
  scaler = StandardScaler()
  train = scaler.fit_transform(o_y_train)
  val = scaler.transform(o_y_val)
  return train,val

def extract_transform(vdf,val_name,labels):
  arr = [float(vdf.loc[vdf['Sample'] == el][val_name].values[0]) for el in labels]
  data = np.array(arr).reshape(-1, 1)
  return data

def custom_kfold_split(X_list,y_list,sample_names):
  d = {}
  groups = np.array([d.setdefault(x[:-1], len(d)) for x in sample_names])

  kf = GroupKFold(n_splits=5)

  kf.get_n_splits(X_list,y_list,groups)

  splits = []
  for i, (train_index, test_index) in enumerate(kf.split(X_list, y_list, groups)):
     print(f"Fold {i}:")
     print(f"  Train: group={set(groups[train_index])}")
     print(f"  Test:  group={set(groups[test_index])}")
     splits.append((train_index,test_index))

  return splits