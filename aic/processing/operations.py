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

import numpy as np
from tqdm import tqdm
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
import aic.processing.fitting as ft
import aic.misc.files as fs


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
    elif isinstance(volume_array, mesh.Mesh):
        volume_array = volume_array.crop(
            bounds=[x_min, x_max, y_min, y_max, z_min, z_max])
    elif isinstance(volume_array, np.ndarray):
        print('here')
        index = \
            np.where((volume_array[:, 0] > x_min)
                     & (volume_array[:, 0] < x_max)
                     & (volume_array[:, 1] > y_min)
                     & (volume_array[:, 1] < y_max)
                     & (volume_array[:, 2] > z_min)
                     & (volume_array[:, 2] < z_max))
        volume_array = volume_array[index]
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
    return points, matrix


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


def icp(source,
        target,
        transformation=np.eye(4),
        verbose=False,
        show=False):
    """Apply ICP.

    Apply Iterative Closest Point to match point cloud valve with template
    """
    # Pass source to Open3D.o3d.geometry.PointCloud and visualize
    pcd_source_threshold = o3d.geometry.PointCloud()
    pcd_source_threshold.points = o3d.utility.Vector3dVector(
        source[:, :3])

    # Pass source to Open3D.o3d.geometry.PointCloud and visualize
    pcd_target = o3d.geometry.PointCloud()
    pcd_target.points = o3d.utility.Vector3dVector(target[:, :3])

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


def get_mask_2D(data,
                dimensions,
                spacing):
    """Generate mask."""
    print("Generating mask...")
    mask_agatston = np.zeros([dimensions[2],
                              dimensions[0],
                              dimensions[1]],
                             dtype=np.uint8)

    # all voxels have value zero except ones predicted:
    for d in tqdm(data):
        x = int(d[0]/spacing[0])
        y = int(d[1]/spacing[1])
        z = int(d[2]/spacing[2])
        mask_agatston[z, x, y] = 1

    for k in range(dimensions[2]):
        mask_agatston[k, ::] = np.rot90(mask_agatston[k, ::])
    return mask_agatston


def centered_point(points):
    """Put points centered in (0,0,0)."""
    x_mean = \
        (np.min(points[:, 0]) + np.max(points[:, 0]))/2
    y_mean = \
        (np.min(points[:, 1]) + np.max(points[:, 1]))/2
    z_mean = \
        (np.min(points[:, 2]) + np.max(points[:, 2]))/2

    if points.shape[1] == 4:
        # 4 dimension due to intensity value
        points -= np.array([x_mean,
                            y_mean,
                            z_mean,
                            0])
        translation = np.array([x_mean,
                                y_mean,
                                z_mean,
                                0])
    else:
        points -= np.array([x_mean,
                            y_mean,
                            z_mean])
        translation = np.array([x_mean,
                                y_mean,
                                z_mean])
    return points, translation


def get_native_valve(radius):
    """Load native valve based on radius."""
    path = fs.get_native_root / str(radius) / '.npy'
    with open(str(path), 'rb') as f:
        native = np.load(f)
    return native


def get_candidates(predictions_agatston_points,
                   w_fit,
                   r_fit,
                   threshold,
                   ratio_spacing,
                   spacing,
                   dimensions):
    """Get Agatston candidates."""
    # Projection along z axis
    predictions_agatston_points_final = \
        predictions_agatston_points.copy()
    points, matrix = z_projection(predictions_agatston_points,
                                  w_fit)
    # Center in (0,0,0)
    points, translation = centered_point(points)
    # Determine the radius of the valve
    taille = closest_element(2*r_fit)
    # Load associated native valve
    native = get_native_valve(taille)
    # ICP Registration
    native = icp(source=native, target=points)
    # Hull method
    hull = ft.convex_hull(native[:, :3])
    mask_hull = isInHull(points[:, :3],
                         hull)
    points = points[mask_hull]
    # Affine transformation in original location
    points += translation 
    if points.shape[1] == 4:
        points = \
            (np.linalg.inv(matrix)@(points.T)).T
    else:
        points = \
            (np.linalg.inv(matrix[:3, :3])@(points[:, :3].T)).T          
    with open('test.npy','wb') as f:
        np.save(f, points)
    # Algo a developer
    # Chargement de la valve native correspondante et icp fit
    # Estimation des points à l'intérieur de la valve (A vérifier)
    ##############################################################
    # Rounding of layer due to floating error
    points[:, 2] = np.round(points[:, 2])

    predictions_agatston_points = \
        points[points[:, 3] < threshold]
    predictions_final_points_threshold = \
        points[points[:, 3] >= threshold]

    # Loop to remove outside points
    inner_points = []
    # Hack to avoid first and last layer where few points lead to wrong
    # circle estimation
    first_slices = \
        int(np.percentile(
            np.arange(
                len(np.unique(
                    predictions_agatston_points[:, 2])
                    )
                ), 20))
    last_slices = \
        int(np.percentile(
            np.arange(
                len(np.unique(
                    predictions_agatston_points[:, 2])
                    )
                ), 80))
    for i, z in enumerate(
            np.unique(predictions_agatston_points[:, 2])):
        predictions_final_tmp = \
            predictions_final_points_threshold.copy()
        predictions_final_tmp = \
            predictions_final_tmp[predictions_final_tmp[:, 2] == z]
        predictions_agatston = \
            predictions_agatston_points.copy()
        predictions_agatston = \
            predictions_agatston[predictions_agatston[:, 2] == z]
        if predictions_final_tmp.shape[0] > 2:
            xc, yc, _, _ = \
                leastsq_circle(predictions_final_tmp[:, 0],
                               predictions_final_tmp[:, 1])
            circle_center = np.array([xc, yc, z])
            # Estimate the min value by slices
            r_fit = []
            for point in predictions_final_tmp[:, :3]:
                # Exclude intensity points
                r_fit.append(euclidean(point, circle_center))
            if len(r_fit) > 0:
                r_fit = np.array(r_fit)
                # Based on experimental analysis on template valve,
                # residual space along stent
                if ratio_spacing is not None:
                    r_fit = np.min(r_fit) - \
                        ratio_spacing*spacing[0]
                else:
                    r_fit = np.min(r_fit)
                # Estimate the distance of each point for the agatston
                d = []
                for point in predictions_agatston[:, :3]:
                    # Exclude intensity points
                    d.append(euclidean(point, circle_center))
                d = np.array(d)
                if i < first_slices or i > last_slices:
                    if predictions_final_tmp.shape[0] > 30:
                        p = \
                            predictions_agatston[np.where(d < r_fit)]
                    else:
                        p = np.empty((0, 4))
                else:
                    p = \
                        predictions_agatston[np.where(d < r_fit)]
            else:
                p = np.empty((0, 4))
        else:
            p = np.empty((0, 4))
        inner_points.append(p)
    inner_points = np.concatenate(inner_points)
    mask = \
        np.where(np.all(np.isin(points, inner_points),
                        axis=1))
    mask_agatston = \
        get_mask_2D(
            predictions_agatston_points_final[mask],
            dimensions,
            spacing)
    return mask_agatston
