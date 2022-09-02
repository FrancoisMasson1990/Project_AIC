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

from collections import Counter

import numpy as np
import open3d as o3d
import scipy.ndimage
import vtk
from scipy import optimize
from skimage import measure
from sklearn.cluster import DBSCAN
from tqdm import tqdm
from vedo import mesh
from vedo import volume
from vtk.util.numpy_support import vtk_to_numpy

import aic.misc.files as fs
import aic.misc.sql as sql
import aic.processing.fitting as ft


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
        p, threshold, step_size=step_size, allow_degenerate=True
    )
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


def clustering(
    data,
    model_version,
    center_volume,
    gt_data,
    ratio,
    threshold=3800,
    eps=2.5,
    min_samples=2,
    max_=None,
    spacings=None,
    dimensions=None,
    crop_values=None,
):
    """Use clustering techniques."""
    index = []
    if max_ is None:
        if model_version == 0:
            # Crop border of images
            x_min = float(center_volume[0] - 0.8 * center_volume[0])
            x_max = float(center_volume[0] + 0.8 * center_volume[0])
            y_min = float(center_volume[1] - ratio * center_volume[1])
            y_max = float(center_volume[1] + ratio * center_volume[1])
            z_min = float(center_volume[2] - ratio * center_volume[2])
            z_max = float(center_volume[2] + ratio * center_volume[2])

            index = np.where(
                (data[:, 0] > x_min)
                & (data[:, 0] < x_max)
                & (data[:, 1] > y_min)
                & (data[:, 1] < y_max)
                & (data[:, 2] > z_min)
                & (data[:, 2] < z_max)
            )
            if data[index].shape[0] < 1500:
                # Bad prediction : Border detection most of the time
                data = gt_data
                z_min = float(center_volume[2] - 0.3 * center_volume[2])
                z_max = float(center_volume[2] + 0.3 * center_volume[2])
                y_min = float(center_volume[1] - 0.3 * center_volume[1])
                y_max = float(center_volume[1] + 0.3 * center_volume[1])
                index = np.where(
                    (data[:, 0] > x_min)
                    & (data[:, 0] < x_max)
                    & (data[:, 1] > y_min)
                    & (data[:, 1] < y_max)
                    & (data[:, 2] > z_min)
                    & (data[:, 2] < z_max)
                )
        elif model_version == 1:
            # Crop border of images
            if crop_values is not None:
                x_min = crop_values[0]
                x_max = crop_values[1]
                y_min = crop_values[2]
                y_max = crop_values[3]
            else:
                x_min = float(spacings[0])
                x_max = float(0.95 * spacings[0] * dimensions[0])
                y_min = float(spacings[1])
                y_max = float(0.95 * spacings[1] * dimensions[1])
            z_min = float(spacings[2])
            z_max = float(0.95 * spacings[2] * dimensions[2])
            index = np.where(
                (data[:, 0] > x_min)
                & (data[:, 0] < x_max)
                & (data[:, 1] > y_min)
                & (data[:, 1] < y_max)
                & (data[:, 2] > z_min)
                & (data[:, 2] < z_max)
            )
    else:
        index = np.where(
            (data[:, 0] > max_[0] - 25)
            & (data[:, 0] < max_[0] + 25)
            & (data[:, 1] > max_[1] - 25)
            & (data[:, 1] < max_[1] + 25)
            & (data[:, 2] > max_[2] - 25)
            & (data[:, 2] < max_[2] + 25)
        )
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
        x_min = max_[0] - 25
        x_max = max_[0] + 25
        y_min = max_[1] - 25
        y_max = max_[1] + 25
        z_min = max_[2] - 25
        z_max = max_[2] + 25

    if isinstance(volume_array, volume.Volume):
        dimensions = volume_array.dimensions()
        spacing = volume_array.spacing()
        volume_array = volume_array.crop(
            # z_max
            top=1 - (z_max / (dimensions[2] * spacing[2])),
            # z_min
            bottom=(z_min / (dimensions[2] * spacing[2])),
            # y_max
            front=1 - (y_max / (dimensions[1] * spacing[1])),
            # y_min
            back=(y_min / (dimensions[1] * spacing[1])),
            # x_max
            right=1 - (x_max / (dimensions[0] * spacing[0])),
            # x_min
            left=(x_min / (dimensions[1] * spacing[1])),
        )
    elif isinstance(volume_array, mesh.Mesh):
        volume_array = volume_array.crop(
            bounds=[x_min, x_max, y_min, y_max, z_min, z_max]
        )
    elif isinstance(volume_array, np.ndarray):
        print("here")
        index = np.where(
            (volume_array[:, 0] > x_min)
            & (volume_array[:, 0] < x_max)
            & (volume_array[:, 1] > y_min)
            & (volume_array[:, 1] < y_max)
            & (volume_array[:, 2] > z_min)
            & (volume_array[:, 2] < z_max)
        )
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
            data.topoints().GetMapper().GetInput().GetPoints().GetData()
        )
        # Pixel value intensity
        scalar = np.expand_dims(
            vtk_to_numpy(data.imagedata().GetPointData().GetScalars()), axis=1
        )
        points = np.concatenate((points, scalar), axis=1)
        # Minimal value of interest for Agatston score
        points = points[points[:, 3] > 130]
        if template:
            if threshold is not None:
                points = points[points[:, 3] < threshold]
    if isinstance(data, mesh.Mesh):
        points = vtk_to_numpy(
            data.GetMapper().GetInput().GetPoints().GetData()
        )

    return points


def z_projection(points, w_fit, save=False, name="projection_array.npy"):
    """Project along z.

    Projection of the points along z_axis using
    vector of direction of the cylinder axis
    """
    matrix_array = matrix_transformation(w_fit)
    rot_x = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

    intensity = None
    if points.shape[1] == 4:
        intensity = points[:, 3]

    matrix = rot_x @ matrix_array
    points = (matrix[:3, :3] @ (points[:, :3].T)).T

    if intensity is not None:
        xyz = points
        points = np.zeros((xyz.shape[0], 4))
        points[:, :3] = xyz
        points[:, 3] = intensity

    if save:
        if not name.endswith(".npy"):
            name += ".npy"
        with open(name, "wb") as f:
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
    matrix_array = np.array(
        [
            [t.GetInverse().GetMatrix().GetElement(r, c) for c in range(4)]
            for r in range(4)
        ]
    )
    return matrix_array


def affine_projection(points, affine):
    """Apply Affine transformation."""
    if points.shape[1] == 4:
        points = (np.linalg.inv(affine) @ (points.T)).T
    else:
        points = (np.linalg.inv(affine[:3, :3]) @ (points[:, :3].T)).T
    return points


def isInHull(P, hull):
    """Determine if the list of points P lies inside the hull.

    List of boolean where true means that the point is inside the convex hull
    """
    A = hull.equations[:, 0:-1]
    b = np.transpose(np.array([hull.equations[:, -1]]))
    isInHullBool = np.all(
        (A @ np.transpose(P)) <= np.tile(-b, (1, len(P))), axis=0
    )
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
    # Path for 27 because missing size
    if closest_element == 27:
        return 25
    return closest_element


def icp(source, target, transformation=np.eye(4), verbose=False, show=False):
    """Apply ICP.

    Apply Iterative Closest Point to match point cloud valve with template
    """
    # Pass source to Open3D.o3d.geometry.PointCloud and visualize
    pcd_source = o3d.geometry.PointCloud()
    pcd_source.points = o3d.utility.Vector3dVector(source[:, :3])

    # Pass source to Open3D.o3d.geometry.PointCloud and visualize
    pcd_target = o3d.geometry.PointCloud()
    pcd_target.points = o3d.utility.Vector3dVector(target[:, :3])

    threshold = 10.0
    print("Apply point-to-point ICP")
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd_source,
        pcd_target,
        threshold,
        transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000),
    )

    if verbose:
        print(reg_p2p)
        print("Transformation is:")
        print(reg_p2p.transformation)
        print("")
        print(np.asarray(reg_p2p.correspondence_set))

    if show:
        pcd_source.paint_uniform_color([1, 0, 0])
        pcd_target.paint_uniform_color([0, 0, 1])
        pcd_source.transform(reg_p2p.transformation)
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.6, origin=[0, 0, 0]
        )
        o3d.visualization.draw_geometries([mesh_frame, pcd_source, pcd_target])

    pcd_source = o3d.geometry.PointCloud()
    pcd_source.points = o3d.utility.Vector3dVector(source[:, :3])

    return (
        np.asarray(pcd_source.transform(reg_p2p.transformation).points),
        reg_p2p.transformation,
    )


def affine_icp(points, matrix, normalize_val=10 * 4):
    """Apply Affine transformation for ICP."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    colors = np.zeros((points.shape[0], 3))
    colors[:, 0] = points[:, -1] / normalize_val
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd = pcd.transform(np.linalg.inv(matrix))
    points[:, :3] = np.asarray(pcd.points)
    points[:, -1] = np.asarray(pcd.colors)[:, 0] * normalize_val
    return points


def calc_R(x, y, xc, yc):
    """Get Radius.

    Calculate the distance of each 2D points
    from the center (xc, yc)
    """
    return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)


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
    residu = np.sum((Ri - R) ** 2)
    return xc, yc, R, residu


def euclidean(point1, point2):
    """Get Euclidean values.

    Calculating Euclidean distance using linalg.norm()
    """
    dist = np.linalg.norm(point1 - point2)
    return dist


def get_mask_2D(data, dimensions, spacing):
    """Generate mask."""
    print("Generating mask...")
    mask_agatston = np.zeros(
        [dimensions[2], dimensions[0], dimensions[1]], dtype=np.uint8
    )

    # all voxels have value zero except ones predicted:
    for d in tqdm(data):
        x = int(d[0] / spacing[0])
        y = int(d[1] / spacing[1])
        z = int(d[2] / spacing[2])
        mask_agatston[z, x, y] = 1

    for k in range(dimensions[2]):
        mask_agatston[k, ::] = np.rot90(mask_agatston[k, ::])
    return mask_agatston


def centered_point(points):
    """Put points centered in (0,0,0)."""
    x_mean = (np.min(points[:, 0]) + np.max(points[:, 0])) / 2
    y_mean = (np.min(points[:, 1]) + np.max(points[:, 1])) / 2
    z_mean = (np.min(points[:, 2]) + np.max(points[:, 2])) / 2

    if points.shape[1] == 4:
        # 4 dimension due to intensity value
        points -= np.array([x_mean, y_mean, z_mean, 0])
        translation = np.array([x_mean, y_mean, z_mean, 0])
    else:
        points -= np.array([x_mean, y_mean, z_mean])
        translation = np.array([x_mean, y_mean, z_mean])
    return points, translation


def get_native_valve(radius, typ="Magna"):
    """Load native valve based on radius."""
    path = (
        fs.get_native_root()
        / typ
        / "projected"
        / ("_").join([typ, str(radius)])
        / "projected.npy"
    )
    with open(str(path), "rb") as f:
        native = np.load(f)
    return native


def get_thickness_infos(
    size,
    layer,
    database=str(fs.get_native_root() / "Magna" / "thickness_info.db"),
    columns=["normalized_thickness"],
):
    """Get thickness infos of native valve."""
    conditions = [f'size = "{size}"', f'layer = "{layer}"']
    args = {"conditions": conditions, "columns": columns}
    df = sql.load_sql(database, **args)
    if not df.empty:
        return float(df.iloc[0][columns])


def get_candidates(points, w_fit, r_fit, threshold, spacing, dimensions):
    """Get Agatston candidates."""
    # Projection along z axis
    valve = points.copy()
    valve, affine = z_projection(valve, w_fit)
    # Center in (0,0,0)
    valve, translation = centered_point(valve)
    # Determine the radius of the valve
    size = closest_element(2 * r_fit)
    # Load associated native valve
    native = get_native_valve(size)
    # Arbitrary level of threshold to find the metalic part
    native_threshold = native[native[:, 3] > threshold]
    valve_threshold = valve[valve[:, 3] > threshold]
    # Perform ICP using mainly metalic part
    rot_z = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    _, matrix = icp(
        source=valve_threshold,
        target=native_threshold,
        transformation=rot_z,
        show=False,
    )
    # Apply given affine transformation
    pcd_ = o3d.geometry.PointCloud()
    pcd_.points = o3d.utility.Vector3dVector(valve[:, :3])
    valve[:, :3] = np.asarray(pcd_.transform(matrix).points)
    # Perform Convex-Hull algo to extract outside part
    hull = ft.convex_hull(native[:, :3])
    mask_hull = isInHull(valve[:, :3], hull)
    # Apply masks
    valve = valve[mask_hull]
    # Add a column that will play the role of index
    valve = np.insert(valve, valve.shape[1], np.arange(len(valve)), axis=1)
    # Project along z axis to help for circle-points fitting
    valve_p = valve.copy()
    valve_p[:, 2] = np.round(valve_p[:, 2])
    valve_threshold = valve_p[valve_p[:, 3] > threshold]
    # For each layer, attempt to fit a circle using the component
    # of the metalic part and remove points outside of it by saving
    # its index position for the last column
    candidates = np.array([False] * valve.shape[0])
    p_fit = []
    for _, z in enumerate(np.unique(valve_p[:, 2])[2:-2]):
        valve_z = valve_threshold[valve_threshold[:, 2] == z]
        if valve_z.shape[0] > 2:
            # Fit a circle using 2 points for each layer
            xc, yc, radius, _ = leastsq_circle(valve_z[:, 0], valve_z[:, 1])
            circle_center = np.array([xc, yc, z])
            for point in valve_p[valve_p[:, 2] == z]:
                # Measure the distance from the center of the circle
                dist = euclidean(point[:3], circle_center)
                # Get thickness from a native valve
                thick = get_thickness_infos(size=size, layer=z)
                # Keep only points where distance < radius - thickness
                # Half pixel resolution
                if (thick) and (dist < (radius - (thick / spacing[0]))):
                    p_fit.append(int(point[-1]))

    find_candidates = False
    if p_fit:
        p_fit = np.array(p_fit)
        candidates[p_fit] = True
        find_candidates = True
    valve_candidates = valve[candidates][:, :4]
    # Restore in ref before icp
    valve_candidates = affine_icp(valve_candidates, matrix)
    # Restore translation
    valve_candidates += translation
    # Restore starting ref
    valve_candidates = affine_projection(valve_candidates, affine)
    valve = valve[~candidates][:, :4]
    # Restore in ref before icp
    valve = affine_icp(valve, matrix)
    # Restore translation
    valve += translation
    # Restore starting ref
    valve = affine_projection(valve, affine)
    if find_candidates:
        mask_agatston = get_mask_2D(valve_candidates, dimensions, spacing)
    else:
        mask_agatston = np.zeros(
            [dimensions[2], dimensions[0], dimensions[1]], dtype=np.uint8
        )
    return mask_agatston, valve, valve_candidates
