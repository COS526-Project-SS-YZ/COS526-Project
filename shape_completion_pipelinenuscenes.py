# shape_completion_pipelinenuscenes.py

import os, sys, cv2, random
import numpy as np, open3d as o3d, torch
from PIL import Image

from utils.depth_utils import find_best_camera_iter
from utils.register_utils import multiscale_registration
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from rembg import remove
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

# Paths
DATA_ROOT     = "nuscenes_data_output"
PARTIAL_DIR   = os.path.join(DATA_ROOT, "point_clouds")
GT_DIR        = os.path.join(DATA_ROOT, "GT")
CATEGORY_PROMPT = "a photo of a roadside object"

# Models (load once)
hunyuan3D_pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    'tencent/Hunyuan3D-2', device="cuda"  # on Colab GPU; change to "mps" or remove for CPU
)
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16
)
d2i_pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    safety_checker=None,
    torch_dtype=torch.float16
)
d2i_pipe.scheduler = UniPCMultistepScheduler.from_config(d2i_pipe.scheduler.config)

try:
    d2i_pipe.enable_xformers_memory_efficient_attention()
    print("✅ xFormers efficient attention enabled")
except Exception:
    d2i_pipe.enable_attention_slicing()
    print("⚠️ xFormers not available — using attention slicing instead")

try:
    d2i_pipe.enable_model_cpu_offload()
    print("✅ model CPU-offload enabled")
except Exception:
    print("⚠️ CPU-offload unavailable — proceeding on GPU/RAM only")



def run_completion(
    token: str,
    output_dir: str = "output",
    gen_rgb: bool = False,
    inpaint_depth: bool = False,
    inference_steps: int = 30,
    seed: int = 0,
) -> float:
    """
    Run shape-completion for one nuScenes token.
    Returns the Chamfer distance (in mm).
    """
    # reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # 1) Load partial & GT point clouds
    partial_pcd  = o3d.io.read_point_cloud(os.path.join(PARTIAL_DIR, f"{token}.ply"))
    complete_pcd = o3d.io.read_point_cloud(os.path.join(GT_DIR,      f"{token}.ply"))

    # 2) Skip too-small crops
    if min(len(partial_pcd.points), len(complete_pcd.points)) < 30:
        raise RuntimeError(f"{token} has too few points")

    # 3) Normalize to unit sphere
    center = complete_pcd.get_center()
    radius = np.max(np.linalg.norm(np.asarray(complete_pcd.points) - center, axis=1))
    if radius == 0:
        raise RuntimeError(f"{token} radius is zero")
    scale = 1.0 / radius
    for pc in (partial_pcd, complete_pcd):
        pc.translate(-center)
        pc.scale(scale, center=(0,0,0))

    # 4) Render depth map from best viewpoint
    partial_np = np.asarray(partial_pcd.points)
    _, _, best_depth, _ = find_best_camera_iter(
        partial_np,
        n_cam_hull=1000,
        n_cam_depth_iter=100,
        radius=2.0,
        width=384,
        height=384,
        fov_deg=60
    )
    depth_map = cv2.normalize(best_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 5) Optional hole-filling
    if inpaint_depth:
        mask = (depth_map == 0).astype('uint8') * 255
        depth_map = cv2.inpaint(depth_map, mask, 3, cv2.INPAINT_NS)

    # 6) Optional ControlNet → RGB image
    if gen_rgb:
        rgb_cond = Image.fromarray(depth_map).convert("RGB")
        image = d2i_pipe(
            CATEGORY_PROMPT,
            rgb_cond,
            num_inference_steps=inference_steps,
            controlnet_conditioning_scale=3.0
        ).images[0]
    else:
        image = Image.fromarray(depth_map).convert("RGB")

    # 7) Remove background & run Hunyuan3D
    image = remove(image)
    mesh = hunyuan3D_pipe(image=image)[0]
    mesh_path = os.path.join(output_dir, f"{token}_mesh.ply")
    mesh.export(mesh_path)

    # 8) Align to partial & compute Chamfer distance
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.translate(-mesh.get_center())
    mesh.scale(scale, center=(0,0,0))
    mesh_pcd = mesh.sample_points_uniformly(number_of_points=16384)

    tfm = multiscale_registration(mesh_pcd, partial_pcd)
    mesh_pcd.transform(tfm)

    cd = (
        np.mean(complete_pcd.compute_point_cloud_distance(mesh_pcd)) +
        np.mean(mesh_pcd.compute_point_cloud_distance(complete_pcd))
    )
    return cd


if __name__ == "__main__":
    import argparse
    valid = [os.path.splitext(f)[0] for f in os.listdir(PARTIAL_DIR) if f.endswith('.ply')]

    p = argparse.ArgumentParser("Shape completion on nuScenes crops")
    p.add_argument("--token",           type=str,   choices=valid, help="annotation token")
    p.add_argument("--output_dir",      type=str,   default="output")
    p.add_argument("--gen_rgb",         action="store_true")
    p.add_argument("--inpaint_depth",   action="store_true")
    p.add_argument("--inference_steps", type=int,   default=30)
    p.add_argument("--seed",            type=int,   default=0)
    args = p.parse_args()

    cd = run_completion(
        token           = args.token,
        output_dir      = args.output_dir,
        gen_rgb         = args.gen_rgb,
        inpaint_depth   = args.inpaint_depth,
        inference_steps = args.inference_steps,
        seed            = args.seed,
    )
    print(f"[{args.token}] Chamfer distance: {cd:.3f} mm")
