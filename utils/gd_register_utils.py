import numpy as np
import torch
import torch.optim as optim
import fpsample
from utils.chamfer_python import distChamfer

import open3d as o3d
import copy

def euler_angles_to_rotation_matrix(alpha, beta, gamma):
    """
    Convert 3 Euler angles (in radians) to a 3x3 rotation matrix.
    """
    one = torch.tensor(1., device=alpha.device, dtype=alpha.dtype)
    zero = torch.tensor(0., device=alpha.device, dtype=alpha.dtype)
    
    cos_a = torch.cos(alpha)
    sin_a = torch.sin(alpha)
    cos_b = torch.cos(beta)
    sin_b = torch.sin(beta)
    cos_c = torch.cos(gamma)
    sin_c = torch.sin(gamma)
    
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

def transform_points(points, alpha, beta, gamma, translation, scale):
    """
    Apply a rigid transformation with scale to the points.
    points: Tensor of shape (N, 3)
    alpha, beta, gamma: Tensors of shape (1,) representing rotation angles in radians.
    translation: Tensor of shape (3,)
    scale: Tensor of shape (1,) representing uniform scale.
    Returns the transformed point cloud.
    """
    R = euler_angles_to_rotation_matrix(alpha, beta, gamma)
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

def register_points_gd(source_pc, target_pc, 
                       num_iterations=500, learning_rate=0.5, device='cuda', 
                       learn_alpha=True, learn_beta=True, learn_gamma=True,
                       learn_translation=True, learn_scale=True,
                       isotropic_scale=True, stage2=False):
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
    trainable_params = []
    alpha = torch.tensor(0., device=device, requires_grad=learn_alpha)
    if learn_alpha: 
        trainable_params.append(alpha)
        
    beta = torch.tensor(0., device=device, requires_grad=learn_beta)
    if learn_beta: 
        trainable_params.append(beta)
        
    gamma = torch.tensor(0., device=device, requires_grad=learn_gamma)
    if learn_gamma: 
        trainable_params.append(gamma)
        
    angles = [alpha, beta, gamma]
    translation = torch.nn.Parameter(torch.zeros(3, device=device), requires_grad=learn_translation)
    if learn_translation: trainable_params.append(translation)
    scale = torch.nn.Parameter(torch.ones(1, device=device), requires_grad=learn_scale)
    if not isotropic_scale:
        scale = torch.nn.Parameter(torch.ones(3, device=device), requires_grad=learn_scale)
    if learn_scale: trainable_params.append(scale)
    
    print("Trainable parameters:")
    print
    
    # Best results
    best_loss = float('inf')
    best_angles = [alpha.clone(), beta.clone(), gamma.clone()]
    best_translation = translation.clone()
    best_scale = scale.clone()

    # Set up the optimizer
    optimizer = optim.SGD(trainable_params, lr=learning_rate, momentum=0.0)
    # optimizer = optim.Adam(trainable_params, lr=learning_rate)

    print("Stage 1: Joint optimization of rotation, translation, and scale")
    for iteration in range(num_iterations):
        optimizer.zero_grad()
        
        # Apply the transformation to the source point cloud
        transformed_source = transform_points(source_pc, alpha, beta, gamma, translation, scale)
        
        # Compute the Chamfer loss between the transformed source and target point clouds
        cd_1, cd_2, _, _ = distChamfer(transformed_source.unsqueeze(0), target_pc.unsqueeze(0))
        loss = cd_2.mean()  # Mean Chamfer distance
        
        # Backpropagation
        loss.backward()
        # Multiply the gradient of the scales by 0.1 to make it less sensitive
        if learn_scale:
            scale.grad *= 0.1
             
        optimizer.step()
        
        if iteration % 100 == 0:
            print(f"Iteration {iteration:4d}: Loss = {loss.item():.6f}")
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_angles = [alpha.clone(), beta.clone(), gamma.clone()]
            best_translation = translation.clone()
            best_scale = scale.clone()

    if stage2:
        print("Stage 2: Only optimize translation and scale")
        # Freeze angles
        alpha.requires_grad = False
        beta.requires_grad = False
        gamma.requires_grad = False
        for iteration in range(num_iterations//4):
            optimizer.zero_grad()
            
            # Apply the transformation to the source point cloud
            transformed_source = transform_points(source_pc, alpha, beta, gamma, translation, scale)
            
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
                best_angles = [alpha.clone(), beta.clone(), gamma.clone()]
                best_translation = translation.clone()
                best_scale = scale.clone()

    return best_angles, best_translation, best_scale

def gd_registration(pcd_source, pcd_target, fps_sample=2048, 
                    device='cuda', num_iterations=1000, learning_rate=0.1, 
                    learn_alpha=True, learn_beta=True, learn_gamma=True,
                    learn_translation=True, learn_scale=True,
                    isotropic_scale=True, stage2=False):
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
                                                    num_iterations=num_iterations, 
                                                    learning_rate=learning_rate,
                                                    device=device,
                                                    isotropic_scale=isotropic_scale,
                                                    stage2=stage2,
                                                    learn_alpha=learn_alpha,
                                                    learn_beta=learn_beta,
                                                    learn_gamma=learn_gamma,
                                                    learn_translation=learn_translation,
                                                    learn_scale=learn_scale)

    print("Optimization complete.")
    print("Learned rotation angles (radians):", [angle.data.cpu().numpy() for angle in angles])
    print("Learned translation:", translation.data.cpu().numpy())
    print("Learned scale:", scale.data.cpu().numpy())

    # Plot the results
    R = euler_angles_to_rotation_matrix(angles[0], angles[1], angles[2])
    zero = torch.tensor(0., device=angles[0].device, dtype=angles[0].dtype)
    one = torch.tensor(1., device=angles[0].device, dtype=angles[0].dtype)
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

def register_points_gd_shapenet(source_pc, target_pc, 
                       num_iterations=500, learning_rate=0.5, device='cuda', 
                       learn_alpha=True, learn_beta=True, learn_gamma=True,
                       learn_translation=True, learn_scale=True,
                       isotropic_scale=True, stage2=False):
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
    trainable_params = []
    alpha = torch.tensor(0., device=device, requires_grad=learn_alpha)
    if learn_alpha: 
        alpha.data = torch.rand_like(alpha.data) * 2 * np.pi
        trainable_params.append(alpha)
        
    beta = torch.tensor(0., device=device, requires_grad=learn_beta)
    if learn_beta: 
        beta.data = torch.rand_like(beta.data) * 2 * np.pi
        trainable_params.append(beta)
        
    gamma = torch.tensor(0., device=device, requires_grad=learn_gamma)
    if learn_gamma: 
        gamma.data = torch.rand_like(gamma.data) * 2 * np.pi
        trainable_params.append(gamma)
        
    angles = [alpha, beta, gamma]
    translation = torch.nn.Parameter(torch.zeros(3, device=device), requires_grad=learn_translation)
    if learn_translation: trainable_params.append(translation)
    scale = torch.nn.Parameter(torch.ones(1, device=device), requires_grad=learn_scale)
    if not isotropic_scale:
        scale = torch.nn.Parameter(torch.ones(3, device=device), requires_grad=learn_scale)
    if learn_scale: trainable_params.append(scale)
    
    print("Trainable parameters:")
    print
    
    # Best results
    best_loss = float('inf')
    best_angles = [alpha.clone(), beta.clone(), gamma.clone()]
    best_translation = translation.clone()
    best_scale = scale.clone()

    # Set up the optimizer
    optimizer = optim.SGD(trainable_params, lr=learning_rate, momentum=0.0)
    # optimizer = optim.Adam(trainable_params, lr=learning_rate)

    print("Stage 1: Joint optimization of rotation, translation, and scale")
    for iteration in range(num_iterations):
        optimizer.zero_grad()
        
        # Apply the transformation to the source point cloud
        transformed_source = transform_points(source_pc, alpha, beta, gamma, translation, scale)
        
        # Compute the Chamfer loss between the transformed source and target point clouds
        cd_1, cd_2, _, _ = distChamfer(transformed_source.unsqueeze(0), target_pc.unsqueeze(0))
        loss = cd_2.mean()  # Mean Chamfer distance
        
        # Backpropagation
        loss.backward()
        # Multiply the gradient of the scales by 0.1 to make it less sensitive
        if learn_scale:
            scale.grad *= 0.1
             
        optimizer.step()
        
        if iteration % 100 == 0:
            print(f"Iteration {iteration:4d}: Loss = {loss.item():.6f}")
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_angles = [alpha.clone(), beta.clone(), gamma.clone()]
            best_translation = translation.clone()
            best_scale = scale.clone()

    if stage2:
        print("Stage 2: Only optimize translation and scale")
        # Freeze angles
        alpha.requires_grad = False
        beta.requires_grad = False
        gamma.requires_grad = False
        for iteration in range(num_iterations//4):
            optimizer.zero_grad()
            
            # Apply the transformation to the source point cloud
            transformed_source = transform_points(source_pc, alpha, beta, gamma, translation, scale)
            
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
                best_angles = [alpha.clone(), beta.clone(), gamma.clone()]
                best_translation = translation.clone()
                best_scale = scale.clone()

    return best_angles, best_translation, best_scale

def gd_registration_shapenet(pcd_source, pcd_target, fps_sample=2048, 
                    device='cuda', num_iterations=1000, learning_rate=0.1, 
                    learn_alpha=True, learn_beta=True, learn_gamma=True,
                    learn_translation=True, learn_scale=True,
                    isotropic_scale=True, stage2=False):
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
                                                    num_iterations=num_iterations, 
                                                    learning_rate=learning_rate,
                                                    device=device,
                                                    isotropic_scale=isotropic_scale,
                                                    stage2=stage2,
                                                    learn_alpha=learn_alpha,
                                                    learn_beta=learn_beta,
                                                    learn_gamma=learn_gamma,
                                                    learn_translation=learn_translation,
                                                    learn_scale=learn_scale)

    print("Optimization complete.")
    print("Learned rotation angles (radians):", [angle.data.cpu().numpy() for angle in angles])
    print("Learned translation:", translation.data.cpu().numpy())
    print("Learned scale:", scale.data.cpu().numpy())

    # Plot the results
    R = euler_angles_to_rotation_matrix(angles[0], angles[1], angles[2])
    zero = torch.tensor(0., device=angles[0].device, dtype=angles[0].dtype)
    one = torch.tensor(1., device=angles[0].device, dtype=angles[0].dtype)
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