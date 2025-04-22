import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from tqdm import tqdm

def generate_camera_positions(n, center=np.asarray([0,0,0]), radius = 2.0):
    """
    Generate n evenly distributed points on the unit sphere using the Fibonacci sphere method.
    """
    indices = np.arange(0, n, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n)
    theta = np.pi * (1 + 5**0.5) * indices
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.vstack((x, y, z)).T * radius + center

def compute_orthonormal_basis(normal):
    """
    Given a unit normal vector, compute two perpendicular unit vectors
    that span the plane orthogonal to it.
    """
    # If the normal's x component is not the smallest in magnitude,
    # we can choose an arbitrary vector not parallel to the normal.
    if np.abs(normal[0]) > np.abs(normal[2]):
        u = np.array([-normal[1], normal[0], 0.0])
    else:
        u = np.array([0.0, -normal[2], normal[1]])
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    return u, v

def generate_camera_positions_disk(n, center=np.asarray([0,0,0]), radius = 1.0, normal = np.array([0,0,1])):
    """
    Generate n evenly distributed points on the unit disk using the Fibonacci disk method.
    """
    indices = np.arange(0, n, dtype=float) + 0.5
    r = np.sqrt(indices / n) * radius
    theta = 2 * np.pi * (1 - 1/ ( (1 + np.sqrt(5)) / 2 )) * indices
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    u, v = compute_orthonormal_basis(normal)
    
    camera_positions = center + x[:, None] * u + y[:, None] * v
    
    return camera_positions

def spherical_inversion(points, R):
    """
    Apply spherical inversion (flipping) to the set of points.
    Points that are at the origin are left unchanged.
    """
    transformed = []
    for p in points:
        norm_p = np.linalg.norm(p)
        if norm_p == 0:
            # If the point is exactly at the camera (origin), keep it.
            transformed.append(p)
        else:
            p_flipped = p + 2 * (R - norm_p) * (p / norm_p)
            transformed.append(p_flipped)
    return np.array(transformed)

def camera_hull_visibility(point_cloud, camera, margin=1.0):
    """
    For a given camera position, compute the number of points in the
    point cloud that are visible using the HPR operator.
    """
    # Translate the point cloud so that the camera is at the origin.
    translated = point_cloud - camera
    
    # Choose R such that all points are inside the sphere.
    norms = np.linalg.norm(translated, axis=1)
    R = norms.max() + margin
    
    # Apply spherical inversion.
    # transformed = spherical_inversion(translated, R)
    transformed = translated + 2 * (R - norms)[:, None] * (translated / norms[:, None])  # faster
    
    # It is often a good idea to include the origin in the convex hull.
    
    # Compute the convex hull.
    all_points = np.vstack((transformed, np.zeros((1, 3))))
    
    # hull = ConvexHull(all_points)
    # all_points = np.vstack((np.zeros((1, 3)), transformed))
    # hull = SciConvexHull(all_points)
    
    o3d_pcl = o3d.geometry.PointCloud()
    o3d_pcl.points = o3d.utility.Vector3dVector(all_points)
    hull, _ = o3d_pcl.compute_convex_hull(joggle_inputs=True)
    
    # # Extract indices of the points on the hull.
    visible_idx = np.asarray(hull.vertices)
    # visible_idx_sci = hull_sci.vertices
    
    # The first point in all_points is the origin. Remove it.
    # Visible indices that are greater than 0 correspond to original points.
    visible_original = visible_idx[visible_idx > 0] - 1  # subtract 1 because we added origin at index 0
    
    return len(visible_original), visible_original, hull, all_points

def build_camera_matrix(eye, look_at, up=np.array([0,1,0])):
    """
    Create an extrinsic camera matrix that transforms points from world space
    to camera space. Here we assume a pinhole camera where points in front
    have positive z.
    """
    # Compute the forward (view) direction
    forward = look_at - eye
    forward = forward / np.linalg.norm(forward)
    
    # If up is nearly parallel to forward, choose a different up.
    if np.abs(np.dot(forward, up)) > 0.99:
        up = np.array([0,1,0])
    
    # Right vector is cross(forward, up)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    
    # Recompute true up as cross(right, forward)
    true_up = np.cross(right, forward)
    
    # Build rotation matrix: columns are the right, true_up, forward vectors.
    # We want to transform a world point X to camera coordinates: X_cam = R^T (X - eye)
    R = np.column_stack((right, true_up, forward))
    return R, eye  # eye is the camera center

def get_intrinsics(width, height, fov_deg=60):
    """
    Compute the camera intrinsic matrix for a given image size and FOV.
    We assume square pixels and the principal point at the center.
    """
    fov = np.deg2rad(fov_deg)
    f = (width/2) / np.tan(fov/2)
    cx, cy = width/2, height/2
    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1]])
    return K

def project_points(point_cloud, R, cam_center, K):
    """
    Project the world point cloud into the camera image.
    Returns:
      - image_points: 2D homogeneous coordinates (Nx2)
      - depths: corresponding z (depth) values in camera space.
    """
    # Translate and rotate points: X_cam = R^T (X - cam_center)
    translated = point_cloud - cam_center  # (N,3)
    X_cam = translated.dot(R)  # because R's columns are the camera axes
    
    # Only keep points with z > 0 (in front of the camera)
    valid = X_cam[:,2] > 0
    X_cam = X_cam[valid]
    
    # Perspective projection: (u, v, 1)^T = K * (x/z, y/z, 1)^T
    points_norm = X_cam[:, :2] / X_cam[:, 2][:, np.newaxis]
    homogeneous = np.hstack((points_norm, np.ones((points_norm.shape[0], 1))))
    proj = (K @ homogeneous.T).T  # (N,3)
    
    # Convert homogeneous coordinates to pixel coordinates (u, v)
    image_points = proj[:, :2] / proj[:, 2][:, np.newaxis]
    # print(image_points.shape)
    depths = X_cam[:, 2]
    depths[valid] = 1 / depths[valid]  # depth = 1/z
    
    return image_points, depths

def build_depth_map(image_points, depths, width, height):
    """
    Build a depth map (2D array) of size (height, width) from projected points.
    For each pixel, we keep the minimum depth value.
    """
    depth_map = np.full((height, width), 0.0)  # initialize with zeros
    
    # Round the floating-point image points to nearest pixel indices.
    # Make sure the indices are within the image bounds.
    u = np.round(image_points[:,0]).astype(int)
    v = np.round(image_points[:,1]).astype(int)
    # Reverse the order of u
    u = width - 1 - u
    
    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, d = u[valid], v[valid], depths[valid]
    
    # For each valid point, update the depth map with the minimum depth.
    for i, j, depth in zip(v, u, d):  # note: row index = v, column index = u
        if depth > depth_map[i, j]:
            depth_map[i, j] = depth
            
    # Replace inf with 0 (or you might choose to leave as inf)
    depth_map[depth_map == np.inf] = 0
    return depth_map

def camera_depth_visibility(points, 
                               cam,
                               look_at=np.array([0,0,0]),
                               up=np.array([0,1,0]),
                               width=128,
                               height=128,
                               fov_deg=80):
    """
    Evaluate the visibility of a point cloud from a camera position.
    Returns:
      - visible_count: number of visible pixels
      - depth_map: depth map of the visible points
    """
    R_cam, cam_center = build_camera_matrix(cam, look_at, up)
    K = get_intrinsics(width, height, fov_deg)
    img_pts, depths = project_points(points, R_cam, cam_center, K)
    depth_map = build_depth_map(img_pts, depths, width, height)
    visible_count = np.count_nonzero(depth_map)
    return visible_count, depth_map

def find_best_camera_hull(point_cloud, camera_positions):
    """
    Given a point cloud and an array of camera positions,
    determine the camera that sees the maximum number of points.
    """
    
    best_camera = None
    best_visible_count = -1
    
    for camera in tqdm(camera_positions, desc='Computing visibility with convex hull'):
        count, vis_indices, _, _ = camera_hull_visibility(point_cloud, camera)
        if count > best_visible_count:
            best_visible_count = count
            best_camera = camera
            
    return best_camera, best_visible_count

def find_best_camera_depth(points, 
                            camera_positions,
                            look_at=np.array([0,0,0]),
                            up=np.array([0,1,0]),
                            width=512,
                            height=512,
                            fov_deg=60):
    """
    Find the camera position with the most visible points.
    """
    max_visible_count = 0
    best_camera = None
    best_depth_map = None
    
    for cam in tqdm(camera_positions, desc='Computing visibility with depth map'):
        visible_count, depth_map = camera_depth_visibility(points, cam, look_at, up, width, height, fov_deg)
        if visible_count > max_visible_count:
            max_visible_count = visible_count
            best_camera = cam
            best_depth_map = depth_map
            
    return best_camera, max_visible_count, best_depth_map



def find_best_camera_iter(point_cloud, n_cam_hull=100, n_cam_depth_iter=100, radius=2.0, width=512, height=512, fov_deg=60):
    """
    Given a point cloud and an array of camera positions,
    determine the camera that sees the maximum number of points.
    """
    best_camera = np.zeros(3)
    best_visible_count = -1
    
    camera_positions_hull = generate_camera_positions(n_cam_hull, radius=radius)
    best_camera, best_visible_count = find_best_camera_hull(point_cloud, camera_positions_hull)
    
    print("[INFO] Best camera position:", best_camera)
    print("[INFO] Best visible count:", best_visible_count)
    
    best_camera_prev = best_camera
    delta_best = 1.0
    n_iters = 0
    
    camera_positions_all = camera_positions_hull
    
    while delta_best > 1e-3 or n_iters < 5:
        r_curr = radius / (2**(0.25*(n_iters+1)))
        print("[INFO] Current radius:", r_curr)
        camera_positions_depth = generate_camera_positions_disk(n_cam_depth_iter, 
                                                                center=best_camera, 
                                                                radius=r_curr,
                                                                normal=best_camera / radius)
        camera_positions_depth = camera_positions_depth / np.linalg.norm(camera_positions_depth, axis=1)[:, None] * radius
        cam, count, best_depth_map = find_best_camera_depth(point_cloud, 
                                                            camera_positions_depth,
                                                            width=width,
                                                            height=height,
                                                            fov_deg=fov_deg)
        if count > best_visible_count:
            best_visible_count = count
            best_camera_prev = best_camera
            best_camera = cam
            print("[INFO] Best camera position:", best_camera)
            print("[INFO] Best visible count:", best_visible_count)
            delta_best = np.linalg.norm(best_camera_prev - best_camera)
        else:
            delta_best = 0.0
            
        n_iters += 1
        print("Iteration:", n_iters, "Best visible count:", best_visible_count)
        
        camera_positions_all = np.vstack((camera_positions_all, camera_positions_depth))
        
    print("[INFO] Best camera position:", best_camera)
    print("[INFO] Best visible count:", best_visible_count)
    
    return best_camera, best_visible_count, best_depth_map, camera_positions_all
        
    # # create a set of camera positions around the best camera as a disk
    # camera_positions_depth = generate_camera_positions(n_cam_depth_iter, center=best_camera, radius=radius / 2)
    # camera_positions_depth = camera_positions_depth / np.linalg.norm(camera_positions_depth, axis=1)[:, None] * radius
    
    # best_camera, best_visible_count, best_depth_map = find_best_camera_depth(point_cloud, camera_positions_depth)
    # print("[INFO] Best camera position:", best_camera)
    # print("[INFO] Best visible count:", best_visible_count)
    
    # return best_depth_map, np.vstack((camera_positions_hull, camera_positions_depth)), best_camera, best_visible_count
    
    # camera_positions_all = camera_positions
    
    # while delta_best > 0 or n_iters < 10:
        
    #     best_camera_now = best_camera
    #     for camera in camera_positions:
    #         count, vis_indices, _, _ = camera_hull_visibility(point_cloud, camera)
    #         if count > best_visible_count:
    #             best_visible_count = count
    #             best_camera_now = camera
    #             best_visible_indices = vis_indices
        
    #     # compute the maximum distance between the best camera and the camera positions
    #     radius_local = radius / (2**(0.5*(n_iters+1)))
    #     # radius_local = radius / 2
    #     camera_positions = generate_camera_positions(n_cam_iter, center=best_camera_now, radius=radius_local)
    #     camera_positions = camera_positions / np.linalg.norm(camera_positions, axis=1)[:, None] * radius
    #     camera_positions_all = np.vstack((camera_positions_all, camera_positions))
        
    #     delta_best = np.linalg.norm(best_camera_now - best_camera)
    #     best_camera = best_camera_now
        
    #     n_iters += 1
    #     print("Iteration:", n_iters, "Best visible count:", best_visible_count)
            
    # return best_camera, best_visible_count, best_visible_indices, camera_positions_all