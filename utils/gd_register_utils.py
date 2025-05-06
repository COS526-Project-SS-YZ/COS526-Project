import numpy as np
import torch
import torch.optim as optim
import fpsample
from utils.chamfer_python import distChamfer

import open3d as o3d
import copy

def euler_angles_to_rotation_matrix(theta):
    """
    Convert a tensor of 3 Euler angles (in radians) to a 3x3 rotation matrix.
    theta = [alpha, beta, gamma] corresponds to rotations about x, y, and z axes.
    """
    one = torch.tensor(1., device=theta.device, dtype=theta.dtype)
    zero = torch.tensor(0., device=theta.device, dtype=theta.dtype)
    
    cos_a = torch.cos(theta[0])
    sin_a = torch.sin(theta[0])
    cos_b = torch.cos(theta[1])
    sin_b = torch.sin(theta[1])
    cos_c = torch.cos(theta[2])
    sin_c = torch.sin(theta[2])
    
    # Rotation about x-axis
    R_x = torch.stack([
        torch.stack([one,      zero,   zero]),
        torch.stack([zero, cos_a, -sin_a]),
        torch.stack([zero, sin_a,  cos_a])
    ])
    
    # Rotation about y-axis
    R_y = torch.stack([
        torch.stack([cos_b, zero, sin_b]),
        torch.stack([zero,  one,  zero]),
        torch.stack([-sin_b, zero, cos_b])
    ])
    
    # Rotation about z-axis
    R_z = torch.stack([
        torch.stack([cos_c, -sin_c, zero]),
        torch.stack([sin_c,  cos_c, zero]),
        torch.stack([zero,   zero,  one])
    ])
    
    # Combined rotation matrix: using Z-Y-X order
    R = torch.mm(R_z, torch.mm(R_y, R_x))
    return R

def transform_points(points, angles, translation, scale):
    """
    Apply a rigid transformation with scale to the points.
    points: Tensor of shape (N, 3)
    angles: Tensor of shape (3,)
    translation: Tensor of shape (3,)
    scale: Tensor of shape (1,) representing uniform scale.
    Returns the transformed point cloud.
    """
    R = euler_angles_to_rotation_matrix(angles)
    zero = torch.tensor(0., device=points.device, dtype=points.dtype)
    one = torch.tensor(1., device=points.device, dtype=points.dtype)
    # Using matrix multiplication: note that we use the transpose of R so that each point is rotated properly.
    transformation = torch.stack([torch.stack([R[0, 0], R[0, 1], R[0, 2], translation[0]]),
                                  torch.stack([R[1, 0], R[1, 1], R[1, 2], translation[1]]),
                                  torch.stack([R[2, 0], R[2, 1], R[2, 2], translation[2]]),
                                  torch.stack([zero, zero, zero, one])])
    points_homogeneous = torch.cat([points, torch.ones(points.shape[0], 1, device=points.device)], dim=1)  # shape: (N, 4)
    transformed = torch.matmul(points_homogeneous, transformation.T)[:, :3]  # shape: (N, 3)
    transformed = transformed * scale  # Apply uniform scaling
    return transformed

def register_points_gd(source_pc, target_pc, num_iterations=500, learning_rate=0.5, device='cuda', isotropic_scale=True):
    """
    Register the source point cloud to the target point cloud using gradient descent.
    
    Parameters:
        source_pc (torch.Tensor): Source point cloud (partial) of shape (N, 3).
        target_pc (torch.Tensor): Target point cloud (complete) of shape (M, 3).
        num_iterations (int): Number of optimization iterations.
        learning_rate (float): Learning rate for the optimizer.
    
    Returns:
        angles, translation, scale: Learned parameters.
    """
    # Initialize the learnable parameters
    angles = torch.nn.Parameter(torch.zeros(3, device=device))
    translation = torch.nn.Parameter(torch.zeros(3, device=device))
    scale = torch.nn.Parameter(torch.ones(1, device=device))
    if not isotropic_scale:
        scale = torch.nn.Parameter(torch.ones(3, device=device))
    
    # Best results
    best_loss = float('inf')
    best_angles = angles.clone()
    best_translation = translation.clone()
    best_scale = scale.clone()

    # Set up the optimizer
    # optimizer = optim.SGD([angles, translation, scale], lr=learning_rate, momentum=0.0)
    optimizer = optim.Adam([angles, translation, scale], lr=learning_rate)

    print("Stage 1: Joint optimization of rotation, translation, and scale")
    for iteration in range(num_iterations):
        optimizer.zero_grad()
        
        # Apply the transformation to the source point cloud
        transformed_source = transform_points(source_pc, angles, translation, scale)
        
        # Compute the Chamfer loss between the transformed source and target point clouds
        cd_1, cd_2, _, _ = distChamfer(transformed_source.unsqueeze(0), target_pc.unsqueeze(0))
        loss = cd_2.mean()  # Mean Chamfer distance
        
        # Backpropagation
        loss.backward()
        # Multiply the gradient of the scales by 0.1 to make it less sensitive
        scale.grad #*= 0.1
             
        optimizer.step()
        
        if iteration % 100 == 0:
            print(f"Iteration {iteration:4d}: Loss = {loss.item():.6f}")
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_angles = angles.clone()
            best_translation = translation.clone()
            best_scale = scale.clone()

    print("Stage 2: Only optimize translation and scale")
    # Freeze angles
    angles.requires_grad = False
    for iteration in range(num_iterations//4):
        optimizer.zero_grad()
        
        # Apply the transformation to the source point cloud
        transformed_source = transform_points(source_pc, angles, translation, scale)
        
        # Compute the Chamfer loss between the transformed source and target point clouds
        cd_1, cd_2, _, _ = distChamfer(transformed_source.unsqueeze(0), target_pc.unsqueeze(0))
        loss = cd_2.mean()
        
        # Backpropagation
        loss.backward()
        # Multiply the gradient of the scales by 0.1 to make it less sensitive
        scale.grad *= 0.1
        
        optimizer.step()
        if iteration % 100 == 0:
            print(f"Iteration {iteration:4d}: Loss = {loss.item():.6f}")
            
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_angles = angles.clone()
            best_translation = translation.clone()
            best_scale = scale.clone()

    return best_angles, best_translation, best_scale

def gd_registration(pcd_source, pcd_target, fps_sample=2048, device='cuda', isotropic_scale=True):
    torch.manual_seed(42)
    
    # Downsample the point clouds
    target_pc_idx = fpsample.fps_sampling(np.asarray(pcd_target.points), fps_sample)
    source_pc_idx = fpsample.fps_sampling(np.asarray(pcd_source.points), fps_sample)
    target_pc = torch.tensor(np.asarray(pcd_target.points)[target_pc_idx], dtype=torch.float32, device=device)
    source_pc = torch.tensor(np.asarray(pcd_source.points)[source_pc_idx], dtype=torch.float32, device=device)
    print(f"Target point cloud shape: {target_pc.shape}")
    print(f"Source point cloud shape: {source_pc.shape}")

    # Register the point clouds
    angles, translation, scale = register_points_gd(source_pc, 
                                                    target_pc, 
                                                    num_iterations=1000, 
                                                    learning_rate=0.1,
                                                    device=device,
                                                    isotropic_scale=isotropic_scale)

    print("Optimization complete.")
    print("Learned rotation angles (radians):", angles.data.cpu().numpy())
    print("Learned translation:", translation.data.cpu().numpy())
    print("Learned scale:", scale.data.cpu().numpy())

    # Plot the results
    R = euler_angles_to_rotation_matrix(angles)
    zero = torch.tensor(0., device=angles.device, dtype=angles.dtype)
    one = torch.tensor(1., device=angles.device, dtype=angles.dtype)
    # one = scale.data.squeeze()
    # Using matrix multiplication: note that we use the transpose of R so that each point is rotated properly.
    transformation = torch.stack([torch.stack([R[0, 0], R[0, 1], R[0, 2], translation[0]]),
                                    torch.stack([R[1, 0], R[1, 1], R[1, 2], translation[1]]),
                                    torch.stack([R[2, 0], R[2, 1], R[2, 2], translation[2]]),
                                    torch.stack([zero, zero, zero, one])]).detach().cpu().numpy()
    scale = scale.detach().cpu().numpy()

    # Visualize the partial point cloud and the mesh
    pcd_source_registered = copy.deepcopy(pcd_source)
    pcd_source_registered = pcd_source_registered.transform(transformation)
    pcd_source_registered.points = o3d.utility.Vector3dVector(
        np.asarray(pcd_source_registered.points) * scale
        )
    
    return pcd_source_registered
    
    # return transformation
