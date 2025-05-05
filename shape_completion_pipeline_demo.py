import os    
import cv2  
import argparse
import numpy as np
from scipy import ndimage
from skimage.transform import resize

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
from diffusers.utils import make_image_grid

from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

hunyuan3D_mesh_pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2')
# hunyuan3D_paint_pipe = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2')

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
d2i_pipe.to("cuda")


data_map_txt={"01184":"An outdoor trash can with wheels", # Wheelie-Bin
              "06127":"A plant in a large vase", # vase
              "06830":"Children's tricycle with adult's handle" , # tricycle
              "07306":"An office trash can", # trash can
              "05452":"An a outside chair", # arm chair
              "06145":"A one leg square table", # table
              "05117":"A chair", # chair
              "06188":"A motorcycle", # vespa
              "07136":"A couch", # sofa
              "09639":"An executive chair"} # Swivel chair

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shape Completion Pipeline")
    parser = argparse.ArgumentParser()
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
        "--depth_inpainting",
        action="store_true",
        help="Enable depth inpainting",
    )
    parser.add_argument(
        "--inference_steps",
        type=int,
        default=100,
        help="Number of inference steps for image generation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args(['--obj_id', '05117', 
                            '--verbose', 
                            '--gen_rgb', 
                            #   '--depth_inpainting', 
                            '--inference_steps', '50'])
    print(args)
    
    # Set the random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    object_id = args.obj_id
    verbose = args.verbose
    depth_inpainting = args.depth_inpainting
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
    
    _, _, best_depth_map, best_depth_map_low, _ = find_best_camera_iter_w_low(partial_pcl, 
                                                                        n_cam_hull=500, 
                                                                        n_cam_depth_iter=100, 
                                                                        radius=2.0,
                                                                        width=512,
                                                                        height=512,
                                                                        fov_deg=60,
                                                                        low_res_ratio=1/8)

    depth_map_c = cv2.medianBlur(best_depth_map.astype(np.float32), 5)
    
    # =====================================================================
    # Depth Inpainting
    # =====================================================================
    if depth_inpainting:
        # depth_map_c = cv2.medianBlur(depth_map_c.astype(np.float32), 5)
        # depth_map_c = cv2.medianBlur(depth_map_c.astype(np.float32), 5)
        depth_map_binary = cv2.threshold(depth_map_c, 0, 1, cv2.THRESH_BINARY)[1].astype(np.uint8)
        depth_map_binary_low = cv2.threshold(best_depth_map_low, 0, 1, cv2.THRESH_BINARY)[1]
        depth_map_binary_low = cv2.GaussianBlur(depth_map_binary_low, (5,5), sigmaX=1.0, sigmaY=1.0)
        depth_map_binary_low_up = cv2.resize(depth_map_binary_low, (depth_map_c.shape[1], depth_map_c.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        depth_map_binary_low_up = cv2.threshold(depth_map_binary_low_up, 0.5, 1, cv2.THRESH_BINARY)[1].astype(np.uint8)
        depth_map_xor = cv2.bitwise_xor(depth_map_binary, depth_map_binary_low_up)
        depth_map_xor[depth_map_binary != 0] = 0
        from diffusers import StableDiffusionInpaintPipeline
        from PIL import Image
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=torch.float16,
        )

        pipe.to("cuda")
        prompt = data_map_txt[object_id]
        negative_prompt = "bad anatomy, deformed, ugly, disfigured, intricate details, blurry, out of focus, bad art, bad anatomy, disfig"
        #image and mask_image should be PIL images.
        # convert the depth map to a PIL image
        depth_map_image = cv2.normalize(depth_map_c, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_map_image = cv2.cvtColor(depth_map_image, cv2.COLOR_GRAY2RGB)
        depth_map_image = cv2.resize(depth_map_image, (512, 512), interpolation=cv2.INTER_NEAREST)
        # convert the binary mask to a PIL image
        mask_image = cv2.normalize(depth_map_xor, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        mask_image = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2RGB)
        mask_image = cv2.resize(mask_image, (512, 512), interpolation=cv2.INTER_NEAREST)
        #The mask structure is white for inpainting and black for keeping as is
        depth_map = pipe(prompt=prompt, 
                            negative_prompt=negative_prompt, 
                            image=Image.fromarray(depth_map_image), 
                            mask_image=Image.fromarray(mask_image),
                    num_inference_steps=100,
                    guidance_scale=2.0
                    ).images[0]
        depth_map = np.array(depth_map)
        make_image_grid([Image.fromarray(depth_map_image), 
                        Image.fromarray(mask_image), 
                        Image.fromarray(depth_map)], rows=1, cols=3).save(os.path.join(output_dir, f"{object_id}_depth_inpaint.png"))
    else:
        # depth_map_c = cv2.medianBlur(depth_map_c.astype(np.float32), 5)
        depth_map = cv2.normalize(depth_map_c, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # increase a little bit contrast
        depth_map = cv2.convertScaleAbs(depth_map, alpha=0.9, beta=0)
        depth_map = cv2.cvtColor(depth_map, cv2.COLOR_GRAY2RGB)
        depth_map = cv2.resize(depth_map, (512, 512), interpolation=cv2.INTER_NEAREST)
        make_image_grid([Image.fromarray(depth_map)], rows=1, cols=1).save(os.path.join(output_dir, f"{object_id}_depth.png"))

    # ======================================================================
    # Generate RGB image using ControlNet
    # ======================================================================
    if args.gen_rgb:
        print("[INFO] Depth map generated, now generating image using ControlNet...")
        auxiliary_prompt = ", clean background, no people, no animals"
        # auxiliary_prompt = ""
        image = d2i_pipe(data_map_txt[object_id] + auxiliary_prompt,
                        Image.fromarray(depth_map), 
                        num_inference_steps=args.inference_steps,
                        # num_inference_steps=75,
                        guidance_scale = 6.0,
                        # controlnet_conditioning_scale=0.9,
                        negative_prompt="bad anatomy, deformed, ugly, disfigured, \
                        intricate details, blurry, out of focus, bad art, bad anatomy",
                        # negative_prompt = "bad anatomy, deformed, ugly, disfigured, intricate details, blurry, out of focus, bad art, bad anatomy, disfig, intricate background",
                        # generator=torch.manual_seed(0),
                        generator=torch.manual_seed(42),
                        ).images[0]
        image = remove(image)
    else:
        depth_map_gray = cv2.cvtColor(depth_map, cv2.COLOR_RGB2GRAY)
        depth_map_gray = cv2.normalize(depth_map_gray, None, 0, 1, cv2.NORM_MINMAX)
        depth_map_binary = cv2.threshold(depth_map_gray, 0, 1, cv2.THRESH_BINARY)[1].astype(np.uint8)
        depth_map_binary = cv2.medianBlur(depth_map_binary.astype(np.float32), 5)
        depth_map_rgba = cv2.cvtColor(depth_map, cv2.COLOR_RGB2RGBA)
        depth_map_rgba[depth_map_binary == 0] = (0, 0, 0, 0)
        depth_map_rgba = Image.fromarray(depth_map_rgba.astype(np.uint8))
        image = depth_map_rgba.copy()
    
    print("[INFO] Background removed, now generating mesh using Hunyuan3D...")
    mesh = hunyuan3D_mesh_pipe(image=image)[0]

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