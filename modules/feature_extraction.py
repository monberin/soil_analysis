"""
This module contains functions for feature extration from images.
"""

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.morphology import disk
from skimage.filters import rank
from skimage.feature import local_binary_pattern


def get_distribution(feature_list, num_bins):
    """
    The function returns a distribution from values in the input list;
    feature_list: list
    num_bins: range or int
    """
    hist, edges = np.histogram(feature_list, bins=num_bins, density=True)
    distribution = hist/sum(hist)
    return distribution


def get_image_contours(img_gr):
    """
    The function returns 3 lists: a list of contours, respective perimeters, and areas.
    perimeters are filtered by area and mean pixel intensity on the image pixels
    inside the perimeter;
    img_gr : a grayscale image (2-D numpy array)
    """

    # creating a circle mask
    height, width = img_gr.shape
    center = (width // 2, height // 2)
    radius = min(width, height) // 2
    small_mask = np.zeros((height, width), dtype=np.uint8)
    small_radius = radius - 3
    cv2.circle(small_mask, center, small_radius, 255, thickness=-1)

    # blurring + local Otsu thresholding over the original image
    blur = cv2.GaussianBlur(img_gr, (3, 3), 0)
    thr_img = blur > rank.otsu(blur, disk(40))

    background = np.ones_like(img_gr, dtype=np.uint8)
    thr_img = np.array(thr_img, dtype=np.uint8)

    # masking out pixels that are not in the central circle
    inside_img = cv2.bitwise_or(thr_img, thr_img, mask=small_mask)
    outside_img = cv2.bitwise_or(
        background, background, mask=cv2.bitwise_not(small_mask))
    output = inside_img+outside_img
    output_inv = cv2.bitwise_not(output).astype('uint8') - 254

    # enhancing contrast on the image
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))
    # Top Hat transform
    topHat = cv2.morphologyEx(img_gr, cv2.MORPH_TOPHAT, kernel)
    # Black Hat transform
    blackHat = cv2.morphologyEx(img_gr, cv2.MORPH_BLACKHAT, kernel)
    contr_img = img_gr + topHat - blackHat

    # retrieving contours
    contours, hierarchy = cv2.findContours(
        output_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # first pass over the contoured objects: measuring the roi pixel mean
    total_mean = []
    rgs = 0
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        if perimeter > 1 and area > 5:  # extremely small objects are discarded
            mask = np.zeros_like(img_gr)
            cv2.drawContours(mask, [contour], 0, 255, thickness=cv2.FILLED)

            roi = cv2.bitwise_and(contr_img, contr_img, mask=mask)
            roi_pixels = contr_img[mask != 0]

            total_mean.append(np.mean(roi_pixels))
            rgs += 1
    total_mean = np.asarray(total_mean)
    avg_mean = total_mean.mean()
    mean_std = total_mean.std()

    # second pass over contoured objects: saving final contours/areas/perimeters

    final_contours = []
    final_perimeters = []
    final_areas = []

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)

        if perimeter > 1 and area > 5:  # extremely small objects are discarded
            mask = np.zeros_like(img_gr)
            cv2.drawContours(mask, [contour], 0, 255, thickness=cv2.FILLED)

            roi = cv2.bitwise_and(contr_img, contr_img, mask=mask)
            roi_pixels = contr_img[mask != 0]
            # objects that have a lower mean pixel intencity than the total average are discarded
            if np.mean(roi_pixels) < avg_mean + mean_std/3:
                final_contours.append(contour)
                final_perimeters.append(perimeter)
                final_areas.append(area)

    return (final_contours, final_perimeters, final_areas)


def features_glcm(image, distance, angle):
    """
    The function returns 5 Haralick features extracted from the image
    image: 2-D numpy array
    distance: int
    angle: float
    """
    glcm = graycomatrix(image, distances=[distance], angles=[
        angle], symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')
    dissimilarity = graycoprops(glcm, 'dissimilarity')
    homogeneity = graycoprops(glcm, 'homogeneity')
    energy = graycoprops(glcm, 'energy')
    correlation = graycoprops(glcm, 'correlation')
    return contrast, dissimilarity, homogeneity, energy, correlation


def get_glcmprops(orig_img, img_contours, distance, angle):
    """
    The function returns Haralick texture features extracted from the largest contoured object
    orig_img: 2-D numpy array
    img_contours: list
    distance: int
    angle: float
    """
    c_img = np.copy(orig_img)
    # finding the largest object on the image
    sorted_contours = sorted(
        img_contours, key=lambda x: x.shape[0], reverse=True)
    largest_contour = sorted_contours[0]

    # masking out the object from the original image
    mask = np.zeros_like(c_img)
    cv2.drawContours(mask, [largest_contour], 0, 255, thickness=cv2.FILLED)

    # extracting a rectangular region of interest around the object
    roi = cv2.bitwise_and(c_img, c_img, mask=mask)

    row_indices, col_indices = np.nonzero(roi)
    nonzero_coords = np.column_stack((row_indices, col_indices))
    r_max = np.amax(row_indices)
    r_min = np.amin(row_indices)
    c_max = np.amax(col_indices)
    c_min = np.amin(col_indices)

    cutout_roi = roi[r_min:r_max+1, c_min:c_max+1]

    return features_glcm(cutout_roi, distance, angle)


def get_shape_features(img_contours):
    """
    The function returns 4 lists of shape features from all contoured objects
    img_contours: list
    """
    aspects = []
    extents = []
    solidities = []
    eq_diameters = []

    for cnt in img_contours:
        rect = cv2.minAreaRect(cnt)

        # aspect ratio
        width, height = rect[1]
        aspect_ratio = float(width)/height

        # extent: obj_area/bound_rect_area
        obj_area = cv2.contourArea(cnt)
        extent = float(obj_area)/(width*height)

        # solidity: obj_area/conv_hull_Area
        solidity = float(obj_area)/cv2.contourArea(cv2.convexHull(cnt))

        # equivalent diameter: diameter of the circle whose are is the same as contour area
        eq_diameter = np.sqrt(4*float(obj_area)/np.pi)

        aspects.append(aspect_ratio)
        extents.append(extent)
        solidities.append(solidity)
        eq_diameters.append(eq_diameter)

    return (aspects, extents, solidities, eq_diameters)


def lbp_distribution(orig_img, radius=2):
    """
    The function returns a distribution of local binary patterns
    oring_img: 2-D numpy array
    radius: int
    """
    # checking if the provided image is in grayscale, transforming to grayscale if not
    if orig_img.ndim == 2:
        lbp = local_binary_pattern(
            orig_img, 8*radius, radius, method='uniform')
    else:
        lbp = local_binary_pattern(cv2.cvtColor(
            orig_img, cv2.COLOR_BGR2GRAY), 8*radius, radius, method='uniform')
    num_bins = int(lbp.max() + 1)
    lbp_dist = get_distribution(lbp.ravel(), num_bins)
    return lbp_dist


def get_lbp(orig_img, img_contours):
    """
    The function returns a distribution of local binary patterns from the largest contoured oblect on the image
    oring_img: 2-D numpy array
    img_contours: list
    """

    c_img = np.copy(orig_img)
    # finding the biggest contoured image
    sorted_contours = sorted(
        img_contours, key=lambda x: x.shape[0], reverse=True)
    largest_contour = sorted_contours[0]
    # masking out the image from the original image
    mask = np.zeros_like(c_img)
    cv2.drawContours(mask, [largest_contour], 0, 255, thickness=cv2.FILLED)

    # extracting a rectangular region of interest around the object
    roi = cv2.bitwise_and(c_img, c_img, mask=mask)

    row_indices, col_indices = np.nonzero(roi)
    nonzero_coords = np.column_stack((row_indices, col_indices))
    r_max = np.amax(row_indices)
    r_min = np.amin(row_indices)
    c_max = np.amax(col_indices)
    c_min = np.amin(col_indices)

    cutout_roi = roi[r_min:r_max+1, c_min:c_max+1]
    radius = 2

    return lbp_distribution(cutout_roi, radius)


def cut_square_img(img):
    """
    A helper function to cut out a rectangle fit into the inner circle of the original image
    img: 2-D numpy array
    """
    height, width, _ = img.shape
    center = (width // 2, height // 2)
    half_side = int(min(width, height) // 2 // np.sqrt(2))
    return img[center[1]-half_side:center[1]+half_side, center[0]-half_side:center[0]+half_side, :]


def flatlist(el_tuple):
    """
    A helper function for flattening nested arrays
    """
    flatted = []
    for el in el_tuple:
        flatted += el.flatten().tolist()
    return flatted


def get_feature_vectors(all_images):
    """
    This function iterates over a list of images, returns a list of vectors that consist of all extracted features
    all_images: list
    """
    feature_vectors = []

    for curr_img in all_images:
        # extracting local binary patterns
        lbp_dist = lbp_distribution(curr_img)

        im = cut_square_img(curr_img)
        # separating the image into 3 channels, to extract Haralick features from them separately
        blue, green, red = cv2.split(im)

        vec = lbp_dist.tolist()

        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        # extracting Haralick features from gl comatrices with different angles/distances from three image channels
        for d in range(1, 10):
            for angle in angles:
                blue_tuple = features_glcm(blue, d, angle)
                green_tuple = features_glcm(green, d, angle)
                red_tuple = features_glcm(blue, d, angle)
                vec += flatlist(blue_tuple) + \
                    flatlist(green_tuple) + flatlist(red_tuple)

        feature_vectors.append(vec)

    return feature_vectors


def get_feature_vectors_pd(all_images):
    """
    Function for the Petri Dishes dataset (includes shape features)
    This function iterates over a list of images, returns a list of vectors that consist of all extracted features
    all_images: list
    """
    feature_vectors = []

    for curr_img in all_images:
        # extracting contours, their perimeters and areas
        curr_contours,curr_perimeters,curr_areas = get_image_contours(curr_img)
        curr_aspects,curr_extents,curr_solidities,curr_diameters = get_shape_features(curr_contours)

        # extracting local binary patterns
        lbp_dist = get_lbp(curr_img,curr_contours)

        # transforming the features into probability distributions
        perimeter_dist = get_distribution(curr_perimeters,50)
        area_dist = get_distribution(curr_areas,50)
        aspect_dist = get_distribution(curr_aspects,50)
        extent_dist = get_distribution(curr_extents,50)
        solidity_dist = get_distribution(curr_solidities,50)

        vec = lbp_dist.tolist() + perimeter_dist.tolist() + area_dist.tolist() + aspect_dist.tolist() + extent_dist.tolist() + \
            solidity_dist.tolist()
        
        # extracting Haralick features from gl comatrices with different angles/distances
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]
        for i in range(1,6):
            for angle in angles:
            contrast,dissimiliarity,homogeneity,energy,correlation = get_glcmprops(curr_img,curr_contours,i,angle)

            vec += contrast.flatten().tolist() + dissimiliarity.flatten().tolist() + \
            homogeneity.flatten().tolist() + energy.flatten().tolist() + correlation.flatten().tolist()

        feature_vectors.append(vec)
    
    return feature_vectors