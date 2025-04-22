import numpy as np
import open3d as o3d
import copy

def register_point_cloud(pcd_source, pcd_target, voxel_size=0.025):    
    # Estimate normals
    pcd_source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    pcd_target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    
    # Apply FPFH feature matching
    pcd_source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_source,
                                                                      o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5, max_nn=100))
    pcd_target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_target,
                                                                      o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5, max_nn=100))

    # Apply RANSAC for initial alignment
    distance_threshold = voxel_size # * 1.5
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        pcd_source, pcd_target, pcd_source_fpfh, pcd_target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        10, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal(0.9)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    # print("RANSAC Result:", result_ransac.transformation)
    
    # Refine the registration using ICP
    distance_threshold = voxel_size #* 0.4
    result_icp = o3d.pipelines.registration.registration_icp(
        pcd_source, pcd_target, distance_threshold,
        result_ransac.transformation,
        # np.identity(4),
        # o3d.pipelines.registration.TransformationEstimationPointToPoint()
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )

    return result_icp.transformation

def multiscale_registration(pcd_source, pcd_target, scales=np.arange(0.8,1.25,0.025), voxel_size=0.01):
    
    # draw([pcd_source, pcd_target], width=800, height=600)
    
    best_transformation = None
    best_cd = float('inf')
    for scale in scales:
        # Downsample the point clouds
        pcd_source_copy = copy.deepcopy(pcd_source)
        pcd_source_copy = pcd_source_copy.voxel_down_sample(voxel_size=voxel_size)
        pcd_target_copy = copy.deepcopy(pcd_target)
        pcd_target_copy = pcd_target_copy.voxel_down_sample(voxel_size=voxel_size)
        
        pcd_source_copy = pcd_source_copy.scale(scale, center=pcd_source_copy.get_center())
        transformation = register_point_cloud(pcd_source_copy, pcd_target_copy)
        pcd_source_copy = pcd_source_copy.transform(transformation)
        # Compute the point cloud distance
        cd = np.mean(pcd_target_copy.compute_point_cloud_distance(pcd_source_copy)) # + \
            # np.mean(pcd_source_copy.compute_point_cloud_distance(pcd_target))
        if cd < best_cd:
            best_cd = cd
            best_transformation = transformation
    return best_transformation