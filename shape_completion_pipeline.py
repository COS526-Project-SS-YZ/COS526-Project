import os    
import cv2  
import argparse
import numpy as np

from utils.depth_utils import *
from utils.register_utils import *

from transformers import pipeline
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers.utils import load_image

from rembg import remove

from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

hunyuan3D_pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2')

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16
)

d2i_pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
)

d2i_pipe.scheduler = UniPCMultistepScheduler.from_config(d2i_pipe.scheduler.config)

# Remove if you do not have xformers installed
# see https://huggingface.co/docs/diffusers/v0.13.0/en/optimization/xformers#installing-xformers
# for installation instructions
d2i_pipe.enable_xformers_memory_efficient_attention()

d2i_pipe.enable_model_cpu_offload()


data_map_txt={"01184":"An outdoor trash can with wheels",
              "06127":"A plant in a large vase",
              "06830":"Children's tricycle with adult's handle" ,
              "07306":"An office trash can",
              "05452":"An a outside chair",
              "06145":"A one leg square table",
              "05117":"An old chair",
              "06188":"A motorcycle",
              "07136":"A couch",
              "09639":"A black executive chair"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shape Completion Pipeline")
    parser.add_argument(
        "--obj_id",
        type=str,
        default="09639",
        choices=data_map_txt.keys(),
        help="Object ID to process",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory to save output images",
    )
    parser.add_argument(
        "--gen_rgb",
        action="store_true",
        help="Generate RGB image using ControlNet",
    )
    parser.add_argument(
        "--inference_steps",
        type=int,
        default=50,
        help="Number of inference steps for image generation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()
    
    # Set the random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    object_id = args.obj_id
    verbose = args.verbose
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    pcl_path = f'./redwood_dataset/point_clouds/{object_id}.ply'
    partial_pcl = o3d.io.read_point_cloud(pcl_path)
    partial_pcl_translate = -partial_pcl.get_center()
    partial_pcl = partial_pcl.translate(partial_pcl_translate)
    partial_pcl_scale = 1.0 / np.max(np.linalg.norm(np.asarray(partial_pcl.points), axis=1))
    partial_pcl_scale_center = np.asarray((0, 0, 0))
    partial_pcl = partial_pcl.scale(partial_pcl_scale, center=partial_pcl_scale_center)
    partial_pcl = np.asarray(partial_pcl.points)
    
    best_camera, best_visible_count, best_depth_map, camera_positions_all = find_best_camera_iter(partial_pcl, 
                                                                                          n_cam_hull=1000, 
                                                                                          n_cam_depth_iter=100, 
                                                                                          radius=2.0,
                                                                                          width=384,
                                                                                          height=384,
                                                                                          fov_deg=60)
    
    # downsample the depth map to 1/4 of the original size
    depth_map = cv2.resize(best_depth_map, (best_depth_map.shape[1] // 4, best_depth_map.shape[0] // 4), interpolation=cv2.INTER_NEAREST)
    depth_map = cv2.medianBlur(best_depth_map.astype(np.float32), 5)
    depth_map = cv2.bilateralFilter(best_depth_map.astype(np.float32), d=9, sigmaColor=75, sigmaSpace=75)
    depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_map = cv2.equalizeHist(depth_map)
    depth_map = best_depth_map.astype(np.float32)
    depth_map = (depth_map * 255).astype(np.uint8)
    if verbose: cv2.imwrite(os.path.join(output_dir, f"{object_id}_depth_map.png"), depth_map)
    
    if args.gen_rgb:
        print("[INFO] Depth map generated, now generating image using ControlNet...")
        depth_map = Image.fromarray(depth_map).convert("RGB")
        image = d2i_pipe(data_map_txt[object_id], 
                        depth_map, 
                        num_inference_steps=args.inference_steps,
                        controlnet_conditioning_scale = 3.0,
                        ).images[0]

        if verbose: image.save(os.path.join(output_dir, f"{object_id}_controlnet_gen.png"))
    else:
        image = Image.fromarray(depth_map).convert("RGB")
    
    print("[INFO] Image generated, now removing background...")
    image = remove(image)
    
    if verbose: image.save(os.path.join(output_dir, f"{object_id}_controlnet_gen_no_bg.png"))
    
    print("[INFO] Background removed, now generating mesh using Hunyuan3D...")
    mesh = hunyuan3D_pipe(image=image)[0]

    # Save the mesh to a file
    print("[INFO] Saving mesh...")
    mesh.export(os.path.join(output_dir, f"{object_id}_mesh.ply"))
    print(f"Mesh saved to {os.path.join(output_dir, f'{object_id}_mesh.ply')}")
    
    # Register the mesh to the original point cloud
    print("[INFO] Registering mesh to original point cloud...")
    
    # Load the ground truth mesh
    complete_pcd = o3d.io.read_triangle_mesh(f"./redwood_dataset/GT/{object_id}.ply")
    complete_pcd = complete_pcd.sample_points_uniformly(number_of_points=16384)
    complete_pcd.points = o3d.utility.Vector3dVector(-np.asarray(complete_pcd.points))
    translate = -complete_pcd.get_center()
    complete_pcd = complete_pcd.translate(translate)
    scale = 0.5 / np.max(np.linalg.norm(np.asarray(complete_pcd.points), axis=1))
    complete_pcd = complete_pcd.scale(scale, center=complete_pcd.get_center())
    
    # Load the partial point cloud
    partial_pcd = o3d.io.read_point_cloud(f"./redwood_dataset/point_clouds/{object_id}.ply")
    partial_pcd.points = o3d.utility.Vector3dVector(-np.asarray(partial_pcd.points))
    partial_pcd = partial_pcd.translate(translate)
    partial_pcd = partial_pcd.scale(scale, center=complete_pcd.get_center())
    
    # Load the generated mesh
    mesh = o3d.io.read_triangle_mesh(os.path.join(output_dir, f"{object_id}_mesh.ply"))
    # Normalize the mesh
    mesh = mesh.translate(-mesh.get_center())
    mesh = mesh.scale(0.5 / np.max(np.linalg.norm(np.asarray(mesh.vertices), axis=1)), center=mesh.get_center())
    mesh_pcd = mesh.sample_points_uniformly(number_of_points=16384)
    
    # Register the mesh to the original point cloud
    transformation = multiscale_registration(mesh_pcd, partial_pcd)
    
    mesh_pcd.transform(transformation)
    
    # Compute the chamfer distance
    cd = np.mean(complete_pcd.compute_point_cloud_distance(mesh_pcd)) + \
        np.mean(mesh_pcd.compute_point_cloud_distance(complete_pcd))
    print(f"Chamfer Distance: {cd}")