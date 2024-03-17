import cv2
import numpy as np
import math
from skimage.feature import graycomatrix,graycoprops
from skimage.morphology import disk
from skimage.filters import rank
from skimage.feature import local_binary_pattern

def get_distribution(feature_list,num_bins):
  """
  function returns a distribution from values in the input list;
  (only takes into account values from a single image)
  """
  hist, edges = np.histogram(feature_list, bins=num_bins, density=True)
  distribution = hist/sum(hist)
  return distribution



def get_image_contours(img_gr):
  """
  function returns 3 lists: a list of contours, respective perimeters, and areas.
  perimeters are filtered by area and mean pixel intensity on the image pixels
  inside the perimeter
  """

  height, width = img_gr.shape

  center = (width // 2, height // 2)
  radius = min(width, height) // 2

  small_mask = np.zeros((height, width), dtype=np.uint8)
  small_radius = radius - 3
  cv2.circle(small_mask, center, small_radius, 255, thickness=-1)
  blur = cv2.GaussianBlur(img_gr,(3,3),0)
  thr_img = blur > rank.otsu(blur,disk(40))

  background = np.ones_like(img_gr,dtype=np.uint8)
  thr_img = np.array(thr_img, dtype=np.uint8)

  # inside
  inside_img = cv2.bitwise_or(thr_img,thr_img,mask=small_mask)
  #outside
  outside_img = cv2.bitwise_or(background,background,mask=cv2.bitwise_not(small_mask))
  output=inside_img+outside_img
  output_inv = cv2.bitwise_not(output).astype('uint8') - 254

  #contrast
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(45,45))
  # Top Hat Transform
  topHat = cv2.morphologyEx(img_gr, cv2.MORPH_TOPHAT, kernel)
  # Black Hat Transform
  blackHat = cv2.morphologyEx(img_gr, cv2.MORPH_BLACKHAT, kernel)
  contr_img = img_gr + topHat - blackHat


  # contours
  contours, hierarchy = cv2.findContours(output_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


  # first pass: roi pixel mean
  total_mean = []
  rgs = 0
  for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    if perimeter > 1 and area > 5:
      mask = np.zeros_like(img_gr)
      cv2.drawContours(mask, [contour], 0, 255, thickness=cv2.FILLED)

      roi = cv2.bitwise_and(contr_img, contr_img, mask=mask)
      roi_pixels = contr_img[mask != 0]

      total_mean.append(np.mean(roi_pixels))
      rgs += 1
  total_mean = np.asarray(total_mean)
  avg_mean = total_mean.mean()
  mean_std = total_mean.std()

  # second pass

  final_contours = []
  final_perimeters = []
  final_areas = []

  for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)

    if perimeter > 1 and area > 5:
      mask = np.zeros_like(img_gr)
      cv2.drawContours(mask, [contour], 0, 255, thickness=cv2.FILLED)

      roi = cv2.bitwise_and(contr_img, contr_img, mask=mask)
      roi_pixels = contr_img[mask != 0]

      if np.mean(roi_pixels) < avg_mean + mean_std/3:
        final_contours.append(contour)
        final_perimeters.append(perimeter)
        final_areas.append(area)


  return (final_contours,final_perimeters,final_areas)


def get_glcmprops(orig_img,img_contours,distance,angle):
  """
  function returns 5 glcm features from the object with the biggest perimeter;
  some of the zeroed background is also included in the calculation, as skimage
  function only takes 2d arrays :(
  """
  c_img = np.copy(orig_img)

  sorted_contours = sorted(img_contours,key=lambda x:x.shape[0],reverse=True)
  largest_contour = sorted_contours[0]

  mask = np.zeros_like(c_img)
  cv2.drawContours(mask, [largest_contour], 0, 255, thickness=cv2.FILLED)

  # Extract region of interest using the mask
  roi = cv2.bitwise_and(c_img, c_img, mask=mask)

  row_indices, col_indices = np.nonzero(roi)
  nonzero_coords = np.column_stack((row_indices, col_indices))
  r_max = np.amax(row_indices)
  r_min = np.amin(row_indices)
  c_max = np.amax(col_indices)
  c_min = np.amin(col_indices)

  cutout_roi = roi[r_min:r_max+1,c_min:c_max+1]


  glcm = graycomatrix(cutout_roi, distances=[distance], angles=[angle], symmetric=True, normed=True)

  # GLCM features for the current object
  contrast = graycoprops(glcm, 'contrast')
  dissimilarity = graycoprops(glcm,'dissimilarity')
  homogeneity = graycoprops(glcm, 'homogeneity')
  energy = graycoprops(glcm, 'energy')
  correlation = graycoprops(glcm, 'correlation')
  return (contrast,dissimilarity,homogeneity,energy,correlation)

def get_shape_features(img_contours):
  aspects = []
  extents = []
  solidities = []
  eq_diameters = []

  for cnt in img_contours:
    rect = cv2.minAreaRect(cnt)

    #aspect ratio
    width,height = rect[1]
    aspect_ratio = float(width)/height

    #extent: obj_area/bound_rect_area
    obj_area = cv2.contourArea(cnt)
    extent = float(obj_area)/(width*height)

    #solidity: obj_area/conv_hull_Area
    solidity = float(obj_area)/cv2.contourArea(cv2.convexHull(cnt))

    #equivalent diameter: diameter ofthe circle whose are is the same as contour area
    eq_diameter = np.sqrt(4*float(obj_area)/np.pi)

    aspects.append(aspect_ratio)
    extents.append(extent)
    solidities.append(solidity)
    eq_diameters.append(eq_diameter)

  return (aspects,extents,solidities,eq_diameters)

def get_lbp(orig_img,img_contours):

  c_img = np.copy(orig_img)

  sorted_contours = sorted(img_contours,key=lambda x:x.shape[0],reverse=True)
  largest_contour = sorted_contours[0]

  mask = np.zeros_like(c_img)
  cv2.drawContours(mask, [largest_contour], 0, 255, thickness=cv2.FILLED)

  # Extract region of interest using the mask
  roi = cv2.bitwise_and(c_img, c_img, mask=mask)

  row_indices, col_indices = np.nonzero(roi)
  nonzero_coords = np.column_stack((row_indices, col_indices))
  r_max = np.amax(row_indices)
  r_min = np.amin(row_indices)
  c_max = np.amax(col_indices)
  c_min = np.amin(col_indices)

  cutout_roi = roi[r_min:r_max+1,c_min:c_max+1]

  radius = 2
  lbp = local_binary_pattern(cutout_roi,8*radius,radius,method='uniform')

  lbp_dist = get_distribution(lbp.ravel(),np.arange(0,8*radius+3))

  return lbp_dist