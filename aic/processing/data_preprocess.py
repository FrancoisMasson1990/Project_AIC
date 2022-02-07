#!/usr/bin/env python3.9
# *-* coding: utf-8*-*
"""
Copyright (C) 2022 Project AIC - All Rights Reserved.

Unauthorized copy of this file, via any medium is strictly
prohibited. Proprietary and confidential.

Note
----
Library for processing medical data inputs
and data for training/infering.
"""

import os
import numpy as np
from tqdm import tqdm
import glob
import pydicom
from skimage.transform import resize
from natsort import natsorted
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from vtk.util.numpy_support import vtk_to_numpy
from skimage import measure
import scipy.ndimage
from sklearn.cluster import DBSCAN
from collections import Counter
import open3d as o3d
from scipy import optimize
import vtk
from vedo import *
from vedo import volume, mesh

LABEL_CHANNELS = {"labels": {
    "background": 0,
    "other": 1,
    "Magna_valve": 2,
}}


def normalize_img(img):
    """Normalize the pixel values.

    This is one of the most important preprocessing steps.
    We need to make sure that the pixel values have a mean of 0
    and a standard deviation of 1 to help the model to train
    faster and more accurately.
    """
    for channel in range(img.shape[3]):
        img[:, :, :, channel] = (
            img[:, :, :, channel] - np.mean(img[:, :, :, channel])) \
            / np.std(img[:, :, :, channel])

    return img


def crop_center(img, cropx, cropy, cropz):
    """Take a center crop of the images.

    If we are using a 2D model, then we'll just stack the
    z dimension.
    """
    z, x, y, c = img.shape

    # Make sure starting index is >= 0
    startx = max(x // 2 - (cropx // 2), 0)
    starty = max(y // 2 - (cropy // 2), 0)

    # Make sure ending index is <= size
    endx = min(startx + cropx, x)
    endy = min(starty + cropy, y)

    return img[:, startx:endx, starty:endy, :]


def resampling(img, size):
    """Resampling images."""
    new_image = np.empty((img.shape[0], size, size, 1))
    for i in range(img.shape[0]):
        new_image[i] = resize(img[i], (size, size))
    return new_image


def preprocess_inputs(img, resize=-1):
    """Process the input images."""
    if len(img.shape) != 4:  # Make sure 4D
        img = np.expand_dims(img, -1)

    if (resize != -1):
        img = crop_center(img, resize, resize, -1)

    img = normalize_img(img)

    return img


def preprocess_label_v1(msk, intel_model=False, resize=-1):
    """Process the ground truth labels."""
    # Stack the loaded npy files
    msk = [np.load(msk[i]) for i in range(len(msk))]
    msk = np.stack(msk, axis=0)

    if intel_model:
        if len(msk.shape) != 4:  # Make sure 4D
            msk = np.expand_dims(msk, -1)
    else:
        # extract certain classes from mask
        msks = [(msk == v) for v in LABEL_CHANNELS["labels"].values()]
        msk = np.stack(msks, axis=-1).astype('float')

    # Cropping
    if (resize != -1):
        msk = crop_center(msk, resize, resize, -1)

    # WIP : Trying to find labels with no data imbalanced
    # Remove one label
    # msk = np.delete(msk,1,3) #Removed Others

    index = []
    for index in range(msk.shape[0]):
        is_value = np.all((msk[index, :, :, 1] == 0))
        if not is_value:
            index.append(index)

    return msk, np.array(index)


def preprocess_label(label):
    """Set label attribution.

    Please refer LABEL_CHANNEL for the mask attribution.
    """
    # Stack the loaded npy files
    label = [np.load(label[i]) for i in range(len(label))]
    label = np.stack(label, axis=0)
    # Took the decision to set to 0 other labels and to 1 magna valve
    label[label == 1] = 0.0
    label[label == 2] = 1.0

    return label


def preprocess_img(img):
    """Preprocess images.

    Preprocessing for the image
    z-score normalize
    """
    # Based on vtk algorithm :
    # scrange -> img.GetScalarRange() [min,max values]
    # threshold = (2 * scrange[0] + scrange[1]) / 3.0
    # 500 or 1000 is a good threshold based on observation for Magna valve
    img[img < 0] = 0
    # Read Intensity normalization in medical images from
    # https://theaisummer.com/medical-image-processing/
    # Scale applied plays a crucial role in training
    img[img > 1000] = 1000
    return (img - img.mean()) / img.std()


def crop_dim(img, crop_dim):
    """Crop images.

    Crop around the center of the images based on size provided
    If we are using a 2D model, then we'll just stack the
    z dimension.
    """
    if img.ndim == 3:
        z, x, y = img.shape
    elif img.ndim == 4:
        z, x, y, c = img.shape

    # Make sure starting index is >= 0
    startx = max(x // 2 - (crop_dim // 2), 0)
    starty = max(y // 2 - (crop_dim // 2), 0)

    # Make sure ending index is <= size
    endx = min(startx + crop_dim, x)
    endy = min(starty + crop_dim, y)

    if img.ndim == 3:
        return img[:, startx:endx, starty:endy]
    elif img.ndim == 4:
        return img[:, startx:endx, starty:endy, :]


def expand_list(data_path, format=None):
    """Expand the list."""
    sub_folders = os.listdir(data_path)
    data = []
    for sub_folder in sub_folders:
        root = os.path.join(data_path, sub_folder)
        sub_ = os.listdir(root)
        for i, sub in enumerate(sub_):
            if format is not None:
                data.append(glob.glob(os.path.join(root, sub) + format))
            else:
                data.append(os.path.join(root, sub))
    data = natsorted(data)
    return data


def load_scan(path):
    """Load scan."""
    files = os.listdir(path)
    file_dcm = []
    for f in files:
        if f.endswith('.dcm'):
            file_dcm.append(f)
    slices = [pydicom.read_file(path + '/' + s) for s in file_dcm]
    slices.sort(key=lambda x: int(x.InstanceNumber))

    try:
        slice_thickness = np.abs(
            slices[0].ImagePositionPatient[2] -
            slices[1].ImagePositionPatient[2])
    except Exception as e:
        slice_thickness = np.abs(
            slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices


def load_mask(path):
    """Load mask."""
    mask = glob.glob(path + '/*.npy')
    mask = natsorted(mask)

    return mask


def get_pixels_hu(scans):
    """Get Pixel HU."""
    image = np.stack([s.pixel_array for s in scans])
    image = image.astype(np.int16)
    image[image == -2000] = 0
    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
    image += np.int16(intercept)
    return np.array(image, dtype=np.int16)


def imbalanced_data_counter(image, msks):
    """Deal with imbalanced data.

    Get a repartition of the ratio of the different classes.
    Go through the dataset.json file.
    This done image wise and pixel wise
    """
    # Pixel Wise
    total_pixel = image.shape[0] * image.shape[1] * \
        image.shape[2] * image.shape[3]

    print("\n")
    for key, value in LABEL_CHANNELS["labels"].items():
        count = (msks[:, :, :, 0] == value).sum()
        ratio = 100*count/total_pixel
        print("pixel wise ratio (%) of {} is {}".format(key, str(ratio)))

    # Image Wise
    for key, value in LABEL_CHANNELS["labels"].items():
        count = 0
        for index in range(msks.shape[0]):
            if value == 0:
                is_value = np.all((msks[index, :, :, 0] == value))
            else:
                is_value = np.any((msks[index, :, :, 0] == value))
            if is_value:
                count += 1
        print("image wise ratio (%) of {} is {}".format(
            key, str(count/msks.shape[0])))
    print("\n")


def imbalanced_data_augmentation(imgs, msks, total=20, seed=42):
    """Augment data.

    Construct the image generator for data augmentation then
    initialize the total number of images generated thus far.
    """
    aug = ImageDataGenerator(
        rotation_range=30,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="nearest")

    msks_stack = []
    for i in tqdm(range(msks.shape[0])):
        msks_ = msks[i, :, :, :]
        extra_channel = np.zeros((msks.shape[1], msks.shape[2], 1))
        msks_ = np.concatenate((msks_, extra_channel), axis=2)
        msks_ = np.expand_dims(msks_, 0)
        # prepare iterator
        it = aug.flow(msks_, batch_size=1, seed=seed)
        # generate samples
        for i in range(total):
            batch = it.next()
            msks_stack.append(batch)

    imgs_stack = []
    for i in tqdm(range(imgs.shape[0])):
        imgs_ = np.expand_dims(imgs[i, :, :, :], 0)
        # prepare iterator
        it = aug.flow(imgs_, batch_size=1, seed=seed)
        # generate samples
        for i in range(total):
            batch = it.next()
            imgs_stack.append(batch)

    imgs_augmented = np.vstack(imgs_stack)
    msks_augmented = np.vstack(msks_stack)
    msks_augmented = msks_augmented[:, :, :, :2]

    return imgs_augmented, msks_augmented


def make_mesh(image, threshold=-300, step_size=1):
    """Make mesh."""
    p = image.transpose(1, 2, 0)
    verts, faces, norm, val = measure.marching_cubes(
        p, threshold, step_size=step_size, allow_degenerate=True)
    return verts, faces


def resample(image, pixelspacing, slicethickness, new_spacing=[1, 1, 1]):
    """Resample image."""
    # Determine current pixel spacing
    spacing = map(float, ([slicethickness] + pixelspacing))
    spacing = np.array(list(spacing))

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)

    return image, new_spacing


def clustering(data, model_version, center_volume, gt_data, ratio,
               threshold=3800, eps=2.5, min_samples=2, max_=None,
               spacings=None, dimensions=None, crop_values=None):
    """Use clustering techniques."""
    index = []
    if max_ is None:
        if model_version == 0:
            # Crop border of images
            x_min = float(center_volume[0]-0.8*center_volume[0])
            x_max = float(center_volume[0]+0.8*center_volume[0])
            y_min = float(center_volume[1]-ratio*center_volume[1])
            y_max = float(center_volume[1]+ratio*center_volume[1])
            z_min = float(center_volume[2]-ratio*center_volume[2])
            z_max = float(center_volume[2]+ratio*center_volume[2])

            index = np.where((data[:, 0] > x_min) & (data[:, 0] < x_max)
                             & (data[:, 1] > y_min) & (data[:, 1] < y_max)
                             & (data[:, 2] > z_min) & (data[:, 2] < z_max))
            if data[index].shape[0] < 1500:
                # Bad prediction : Border detection most of the time
                data = gt_data
                z_min = float(center_volume[2]-0.3*center_volume[2])
                z_max = float(center_volume[2]+0.3*center_volume[2])
                y_min = float(center_volume[1]-0.3*center_volume[1])
                y_max = float(center_volume[1]+0.3*center_volume[1])
                index = np.where((data[:, 0] > x_min) & (data[:, 0] < x_max)
                                 & (data[:, 1] > y_min) & (data[:, 1] < y_max)
                                 & (data[:, 2] > z_min) & (data[:, 2] < z_max))
        elif model_version == 1:
            # Crop border of images
            if crop_values is not None:
                x_min = crop_values[0]
                x_max = crop_values[1]
                y_min = crop_values[2]
                y_max = crop_values[3]
            else:
                x_min = float(spacings[0])
                x_max = float(0.95*spacings[0]*dimensions[0])
                y_min = float(spacings[1])
                y_max = float(0.95*spacings[1]*dimensions[1])
            z_min = float(spacings[2])
            z_max = float(0.95*spacings[2]*dimensions[2])
            index = np.where((data[:, 0] > x_min) & (data[:, 0] < x_max)
                             & (data[:, 1] > y_min) & (data[:, 1] < y_max)
                             & (data[:, 2] > z_min) & (data[:, 2] < z_max))
    else:
        index = \
            np.where((data[:, 0] > max_[0]-25) & (data[:, 0] < max_[0]+25)
                     & (data[:, 1] > max_[1]-25) & (data[:, 1] < max_[1]+25)
                     & (data[:, 2] > max_[2]-25) & (data[:, 2] < max_[2]+25))
    if (index) and (data[index].shape[0] != 0):
        data = data[index]

    model = DBSCAN(eps=eps, min_samples=min_samples)
    model.fit_predict(data)

    print("number of cluster found: {}".format(len(set(model.labels_))))
    index = Counter(model.labels_).most_common()

    j = 0
    while index[j][1] > threshold:  # Arbitrary values
        j += 1
    i = np.isin(model.labels_, np.array([index[j][0]]))
    return data[i, :]


def boxe_3d(volume_array, predict, max_=None):
    """Generate boxes 3d coordinates."""
    if max_ is None:
        x_min = np.min(predict[:, 0])
        x_max = np.max(predict[:, 0])
        y_min = np.min(predict[:, 1])
        y_max = np.max(predict[:, 1])
        z_min = np.min(predict[:, 2])
        z_max = np.max(predict[:, 2])
    else:
        x_min = max_[0]-25
        x_max = max_[0]+25
        y_min = max_[1]-25
        y_max = max_[1]+25
        z_min = max_[2]-25
        z_max = max_[2]+25

    if isinstance(volume_array, volume.Volume):
        dimensions = volume_array.dimensions()
        spacing = volume_array.spacing()
        volume_array = \
            volume_array.crop(
                # z_max
                top=1-(z_max/(dimensions[2]*spacing[2])),
                # z_min
                bottom=(z_min/(dimensions[2]*spacing[2])),
                # y_max
                front=1 - (y_max/(dimensions[1]*spacing[1])),
                # y_min
                back=(y_min/(dimensions[1]*spacing[1])),
                # x_max
                right=1 - (x_max/(dimensions[0]*spacing[0])),
                # x_min
                left=(x_min/(dimensions[1]*spacing[1])),)
    if isinstance(volume_array, mesh.Mesh):
        volume_array = volume_array.crop(
            bounds=[x_min, x_max, y_min, y_max, z_min, z_max])
    return volume_array


def normalize(v):
    """Normalize a vector based on its 2 norm."""
    if 0 == np.linalg.norm(v):
        return v
    return v / np.linalg.norm(v)


def point_line_distance(p, l_p, l_v):
    """Calculate line distance.

    Calculate the distance between a point and a line defined
    by a point and a direction vector.
    """
    l_v = normalize(l_v)
    u = p - l_p
    return np.linalg.norm(u - np.dot(u, l_v) * l_v)


def to_points(data, threshold=None, template=False):
    """Extract point from Volume/Mesh polydata."""
    if isinstance(data, volume.Volume):
        points = vtk_to_numpy(
            data.topoints().GetMapper().GetInput().GetPoints().GetData())
        # Pixel value intensity
        scalar = np.expand_dims(vtk_to_numpy(
            data.imagedata().GetPointData().GetScalars()), axis=1)
        points = np.concatenate((points, scalar), axis=1)
        # Minimal value of interest for Agatston score
        points = points[points[:, 3] > 130]
        if template:
            if threshold is not None:
                points = points[points[:, 3] < threshold]
    if isinstance(data, mesh.Mesh):
        points = vtk_to_numpy(
            data.GetMapper().GetInput().GetPoints().GetData())

    return points


def z_projection(points, w_fit, save=False, name='projection_array.npy'):
    """Project along z.

    Projection of the points along z_axis using
    vector of direction of the cylinder axis
    """
    matrix_array = matrix_transformation(w_fit)
    rot_x = np.array([[1, 0, 0, 0],
                      [0, 0, -1, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 1]])

    intensity = None
    if points.shape[1] == 4:
        intensity = points[:, 3]

    matrix = rot_x@matrix_array
    points = (matrix[:3, :3]@(points[:, :3].T)).T

    if intensity is not None:
        xyz = points
        points = np.zeros((xyz.shape[0], 4))
        points[:, :3] = xyz
        points[:, 3] = intensity

    if save:
        if not name.endswith('.npy'):
            name += '.npy'
        with open(name, 'wb') as f:
            np.save(f, points)
    return points


def matrix_transformation(w_fit):
    """Calculate matrix transformation.

    Calculate matrix of transformation
    from direction of the cylinder axis
    """
    theta = np.arccos(w_fit[2])
    phi = np.arctan2(w_fit[1], w_fit[0])
    t = vtk.vtkTransform()
    t.PostMultiply()
    t.RotateX(90)  # put it along Z
    t.RotateY(np.rad2deg(theta))
    t.RotateZ(np.rad2deg(phi))
    matrix_array = np.array([
        [t.GetInverse().GetMatrix().GetElement(
            r, c) for c in range(4)] for r in range(4)])
    return matrix_array


def isInHull(P, hull):
    """Determine if the list of points P lies inside the hull.

    List of boolean where true means that the point is inside the convex hull
    """
    A = hull.equations[:, 0:-1]
    b = np.transpose(np.array([hull.equations[:, -1]]))
    isInHullBool = np.all((A @ np.transpose(P)) <=
                          np.tile(-b, (1, len(P))), axis=0)
    return isInHullBool


def closest_element(value, upper=True):
    """Get closest element.

    How to find the NumPy array element closest
    uppto a given value in Python
    if upper : select upper bound
    """
    array = np.asarray([19, 21, 23, 25, 27, 29])  # Magna size
    if upper:
        eps = 0.001
        value = np.ceil(value) + eps
    absolute_val_array = np.abs(array - value)
    smallest_difference_index = absolute_val_array.argmin()
    closest_element = array[smallest_difference_index]
    return closest_element


def icp(predictions,
        template,
        threshold,
        w_fit,
        cylinder,
        verbose=False,
        show=False):
    """Apply ICP.

    Apply Iterative Closest Point to match point cloud valve with template
    """
    source = template.copy()
    source_threshold = source[source[:, 3] > threshold]
    # Pass source to Open3D.o3d.geometry.PointCloud and visualize
    pcd_source_threshold = o3d.geometry.PointCloud()
    pcd_source_threshold.points = o3d.utility.Vector3dVector(
        source_threshold[:, :3])

    target_raw = predictions.copy()
    target = target_raw[target_raw[:, 3] > threshold]
    # Pass source to Open3D.o3d.geometry.PointCloud and visualize
    pcd_target = o3d.geometry.PointCloud()
    pcd_target.points = o3d.utility.Vector3dVector(target[:, :3])

    transformation = matrix_transformation(w_fit)
    translation = np.array([[cylinder.getTransform().GetMatrix().GetElement(
        r, c) for c in range(4)] for r in range(4)])[:, 3]
    transformation[:, 3] = translation
    rot_x = np.array([[1, 0, 0, 0],
                      [0, 0, -1, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 1]])

    transformation = transformation@rot_x
    threshold = 10.0
    print("Apply point-to-point ICP")
    reg_p2p = \
        o3d.pipelines.registration.registration_icp(
            pcd_source_threshold, pcd_target,
            threshold, transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=2000))

    if verbose:
        print(reg_p2p)
        print("Transformation is:")
        print(reg_p2p.transformation)
        print("")
        print(np.asarray(reg_p2p.correspondence_set))

    if show:
        pcd_source_threshold.paint_uniform_color([1, 0, 0])
        pcd_target.paint_uniform_color([0, 0, 1])
        pcd_source_threshold.transform(reg_p2p.transformation)
        o3d.visualization.draw_geometries([pcd_source_threshold, pcd_target])

    pcd_source = o3d.geometry.PointCloud()
    pcd_source.points = o3d.utility.Vector3dVector(source[:, :3])

    return np.asarray(pcd_source.transform(reg_p2p.transformation).points)


def calc_R(x, y, xc, yc):
    """Get Radius.

    Calculate the distance of each 2D points
    from the center (xc, yc)
    """
    return np.sqrt((x-xc)**2 + (y-yc)**2)


def f(c, x, y):
    """Get algebraic mean.

    Calculate the algebraic distance between the data
    points and the mean circle centered at c=(xc, yc)
    """
    Ri = calc_R(x, y, *c)
    return Ri - Ri.mean()


def leastsq_circle(x, y):
    """Get the coordinates of the barycenter."""
    x_m = np.mean(x)
    y_m = np.mean(y)
    center_estimate = x_m, y_m
    center, ier = optimize.leastsq(f, center_estimate, args=(x, y))
    xc, yc = center
    Ri = calc_R(x, y, *center)
    R = Ri.mean()
    residu = np.sum((Ri - R)**2)
    return xc, yc, R, residu


def euclidean(point1, point2):
    """Get Euclidean values.

    Calculating Euclidean distance using linalg.norm()
    """
    dist = np.linalg.norm(point1 - point2)
    return dist
