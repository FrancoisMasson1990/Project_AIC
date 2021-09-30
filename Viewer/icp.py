import numpy as np
import open3d as o3d
import copy
import glob 

for file_ in glob.glob("./templates/*npy"):
    source = np.load(file_)
    print(source.shape)
    print(np.max(source[:,3]))
    print(np.min(source[:,3]))

# def draw_registration_result(source, target, transformation):
#     source_temp = copy.deepcopy(source)
#     target_temp = copy.deepcopy(target)
#     source_temp.paint_uniform_color([1, 0, 0])
#     target_temp.paint_uniform_color([0, 0, 1])
#     source_temp.transform(transformation)
#     o3d.visualization.draw_geometries([source_temp, target_temp])

# #template
# source = np.load("source.npy")
# # Pass source to Open3D.o3d.geometry.PointCloud and visualize
# pcd_source = o3d.geometry.PointCloud()
# pcd_source.points = o3d.utility.Vector3dVector(source[:,:3])
# #o3d.visualization.draw_geometries([pcd_source])

# target_raw = np.load("target.npy")
# target = target_raw[target_raw[:,3] > 600]
# # Pass source to Open3D.o3d.geometry.PointCloud and visualize
# pcd_target = o3d.geometry.PointCloud()
# pcd_target.points = o3d.utility.Vector3dVector(target[:,:3])
# #o3d.visualization.draw_geometries([pcd_target])


# # #threshold = 0.02
# trans_init = np.eye(4)
# transformation = np.load("transformation.npy")
# rot_x = np.array([[1, 0, 0, 0],
#                   [0, 0,-1, 0],
#                   [0, 1, 0, 0],
#                   [0, 0, 0, 1]])

# transformation = transformation@rot_x

# o3d.visualization.draw_geometries([pcd_source, pcd_target])
# draw_registration_result(pcd_source, pcd_target, transformation)

# threshold = 10.0
# print("Apply point-to-point ICP")
# reg_p2p = o3d.pipelines.registration.registration_icp(pcd_source, pcd_target, 
#                                                       threshold, transformation,
#                                                       o3d.pipelines.registration.TransformationEstimationPointToPoint(),
#                                                       o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = 2000))


# print(reg_p2p)
# print("Transformation is:")
# print(reg_p2p.transformation)
# print("")
# print(np.asarray(reg_p2p.correspondence_set))
# draw_registration_result(pcd_source, pcd_target, reg_p2p.transformation)


# new_target = target_raw[~np.asarray(reg_p2p.correspondence_set)[:,1]]
# pcd_new_target = o3d.geometry.PointCloud()
# pcd_new_target.points = o3d.utility.Vector3dVector(new_target[:,:3])

# pcd_source.paint_uniform_color([1, 0, 0])
# pcd_new_target.paint_uniform_color([0, 0, 1])
# pcd_source.transform(reg_p2p.transformation)
# o3d.visualization.draw_geometries([pcd_source,pcd_new_target])

