# this code does a sanity check on the density of the point clouds and their ground truth
import random, matplotlib.pyplot as plt
import os, random, glob
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

root = "nuscenes_data_output"

# build the file list  ⟵  add this line
partials = glob.glob(os.path.join(root, "point_clouds", "*.ply"))


token = random.choice([os.path.basename(f) for f in partials])
pc  = o3d.io.read_point_cloud(os.path.join(root,"point_clouds",token))
gt  = o3d.io.read_point_cloud(os.path.join(root,"GT",token))

pc_z = np.asarray(pc.points)[:,2]
gt_z = np.asarray(gt.points)[:,2]

plt.hist([pc_z, gt_z], bins=40, label=["partial","GT"])
plt.legend(); plt.title(token); plt.xlabel("height (mm)")
plt.show()
