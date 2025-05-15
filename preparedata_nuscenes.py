from pathlib import Path
import os, numpy as np, open3d as o3d
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix, points_in_box

# Config
DATAROOT = '/Users/ssrotriyo/Desktop/COS526-Project-main/nuScenes' # download the dataset from https://www.nuscenes.org/download
SPLIT    = 'v1.0-mini'
NSWEEPS_PARTIAL = 5     # increase for denser input
NSWEEPS_GT      = 10

OUT_ROOT = Path('nuscenes_data_output')
(OUT_ROOT / 'point_clouds').mkdir(parents=True, exist_ok=True)
(OUT_ROOT / 'GT').mkdir(exist_ok=True)


print("Loading nuScenes …")
nusc = NuScenes(version=SPLIT, dataroot=DATAROOT, verbose=True)

sample_tokens = [rec['token'] for rec in nusc.sample]
print(f"Total samples in split: {len(sample_tokens)}")

processed, skipped = 0, 0

for s_idx, sample_token in enumerate(sample_tokens):
    sample = nusc.get('sample', sample_token)
    lidar_token = sample['data']['LIDAR_TOP']

    # transforms
    sd_rec  = nusc.get('sample_data', lidar_token)
    cs_rec  = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_rec= nusc.get('ego_pose',        sd_rec['ego_pose_token'])

    T_lidar  = transform_matrix(cs_rec['translation'],   Quaternion(cs_rec['rotation']))
    T_global = transform_matrix(pose_rec['translation'], Quaternion(pose_rec['rotation']))

    #  partial cloud (multi‑sweep)
    pc, _ = LidarPointCloud.from_file_multisweep(
    nusc, sample, 'LIDAR_TOP', 'LIDAR_TOP', nsweeps=NSWEEPS_PARTIAL)
    pc.transform(T_global @ T_lidar)

    # iterate over annotations in THIS sample
    for ann_token in sample['anns']:
        ann  = nusc.get('sample_annotation', ann_token)
        box  = nusc.get_box(ann_token)

        mask = points_in_box(box, pc.points[:3, :])        # remove .T
        part = pc.points[:, mask][:3, :].T

        if part.shape[0] == 0:          # skip empty crops
            skipped += 1
            continue

        processed += 1
        o3d.io.write_point_cloud(
            str(OUT_ROOT/'point_clouds'/f'{ann_token}.ply'),
            o3d.geometry.PointCloud(o3d.utility.Vector3dVector(part * 1000))
        )

        #GT cloud (denser multi‑sweep)
        pc_full, _ = LidarPointCloud.from_file_multisweep(
        nusc, sample, 'LIDAR_TOP', 'LIDAR_TOP', nsweeps=NSWEEPS_GT)
        pc_full.transform(T_global @ T_lidar)

        mask_full = points_in_box(box, pc_full.points[:3, :])   # remove .T
        compl = pc_full.points[:, mask_full][:3, :].T
        if compl.shape[0] > 16_384:
            compl = compl[np.random.choice(compl.shape[0], 16_384, replace=False)]

        o3d.io.write_point_cloud(
            str(OUT_ROOT/'GT'/f'{ann_token}.ply'),
            o3d.geometry.PointCloud(o3d.utility.Vector3dVector(compl * 1000))
        )

    # progress every 20 samples
    if (s_idx + 1) % 20 == 0:
        print(f'Processed {s_idx+1}/{len(sample_tokens)} samples '
              f'— {processed} PCs saved, {skipped} skipped')

print("\nDone.")
print(f"Point clouds saved : {processed}")
print(f"Annotations skipped: {skipped}")
print(f"Output folder      : {OUT_ROOT.resolve()}")