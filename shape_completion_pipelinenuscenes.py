# this is the start of the code
import os, argparse, random

import cv2, numpy as np, open3d as o3d, torch, matplotlib.pyplot as plt

import sys

from PIL import Image

from utils.depth_utils import *

from utils.register_utils import *

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

from rembg import remove

from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline



# CHANGED folders for nuScenes

DATA_ROOT   = "nuscenes_data_output"

PARTIAL_DIR = os.path.join(DATA_ROOT, "point_clouds")

GT_DIR      = os.path.join(DATA_ROOT, "GT")

CATEGORY_PROMPT = "a photo of a roadside object"

# models
hunyuan3D_pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2')
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth",
                                             torch_dtype=torch.float16)
d2i_pipe    = StableDiffusionControlNetPipeline.from_pretrained(
                 "runwayml/stable-diffusion-v1-5",
                 controlnet=controlnet,
                 safety_checker=None,
                 torch_dtype=torch.float16)
d2i_pipe.scheduler = UniPCMultistepScheduler.from_config(d2i_pipe.scheduler.config)
d2i_pipe.enable_xformers_memory_efficient_attention()
d2i_pipe.enable_model_cpu_offload()

#
if __name__ == "__main__":

    valid_tokens = [os.path.splitext(f)[0] for f in os.listdir(PARTIAL_DIR)]

    parser = argparse.ArgumentParser("Shape completion on nuScenes crops")

    parser.add_argument("--token", type=str,
                        default=random.choice(valid_tokens),
                        choices=valid_tokens,
                        help="nuScenes annotation token (filename without .ply)")
    parser.add_argument("--output_dir", type=str, default="./output")

    parser.add_argument("--gen_rgb", action="store_true")

    parser.add_argument("--inference_steps", type=int, default=30)

    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    torch.manual_seed(args.seed);  np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    #load point clouds 
    partial_pcd = o3d.io.read_point_cloud(os.path.join(PARTIAL_DIR, f"{args.token}.ply"))

    complete_pcd= o3d.io.read_point_cloud(os.path.join(GT_DIR,      f"{args.token}.ply"))
    
    min_pts = 30          # threshold; tweak if you like
    if len(partial_pcd.points) < min_pts or len(complete_pcd.points) < min_pts:
        print(f"[{args.token}] skipped – "
              f"{len(partial_pcd.points)} partial pts / {len(complete_pcd.points)} GT pts")
        sys.exit(0)

    # centre & normalise to unit sphere (shared for all clouds)
    center = complete_pcd.get_center()

    radius = np.max(np.linalg.norm(np.asarray(complete_pcd.points) - center, axis=1))

    if radius == 0:                          # all points identical so convex hull fails
        print(f"[{args.token}] skipped – GT radius is zero")
        sys.exit(0)
    scale = 1.0 / radius

    for cloud in [partial_pcd, complete_pcd]:
        cloud.translate(-center)
        cloud.scale(scale, center=(0,0,0))

    # numpy array for depth‑camera search
    partial_np = np.asarray(partial_pcd.points)

    # find best camera & depth map 
    best_cam, _, best_depth, _ = find_best_camera_iter(
        partial_np, n_cam_hull=1000, n_cam_depth_iter=100,
        radius=2.0, width=384, height=384, fov_deg=60)

    depth_map = cv2.normalize(best_depth, None, 0, 255,
                              cv2.NORM_MINMAX).astype(np.uint8)

    # optional text‑conditioned RGB 
    if args.gen_rgb:
        rgb_cond = Image.fromarray(depth_map).convert("RGB")
        image = d2i_pipe(CATEGORY_PROMPT, rgb_cond,
                         num_inference_steps=args.inference_steps,
                         controlnet_conditioning_scale=3.0).images[0]
    else:
        image = Image.fromarray(depth_map).convert("RGB")

    image = remove(image)  # strip background

    #  mesh generation
    mesh = hunyuan3D_pipe(image=image)[0]

    mesh_path = os.path.join(args.output_dir, f"{args.token}_mesh.ply")

    mesh.export(mesh_path)

    # align mesh & compute Chamfer
    mesh = o3d.io.read_triangle_mesh(mesh_path)

    mesh.translate(-mesh.get_center())

    mesh.scale(scale, center=(0,0,0))

    mesh_pcd = mesh.sample_points_uniformly(number_of_points=16384)

    transformation = multiscale_registration(mesh_pcd, partial_pcd)

    mesh_pcd.transform(transformation)

    cd = (np.mean(complete_pcd.compute_point_cloud_distance(mesh_pcd)) +
          np.mean(mesh_pcd.compute_point_cloud_distance(complete_pcd)))
    print(f"[{args.token}] Chamfer distance: {cd:.3f} mm")
# this is the end of the code
