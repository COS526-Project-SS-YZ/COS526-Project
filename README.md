# Zero Shot Point Cloud Completion with 3D Diffusion Priors

## Prerequisites
<!-- ```
conda create -n cos526 python=3.9
conda activate cos526
conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
pip install scikit-image matplotlib imageio plotly opencv-python
pip install black usort flake8 flake8-bugbear flake8-comprehensions
conda install pytorch3d -c pytorch3d
conda install pyg -c pyg
pip install point_cloud_utils
pip install open3d
pip install transformers
pip install diffusers
pip install rembg
pip install tqdm
``` -->
```
torch
transformers
diffusers
rembg
tqdm
open3d
```
Install Hunyuan3D-v2 following https://github.com/Tencent/Hunyuan3D-2

## Usage

#### Redwood dataset
Download data in frames_data_full following README.md

#### Prepare data
```
python prepare_data.py
```

#### Run Zero-shot completion on Redwood
See [shape_completion_pipeline_demo_final.ipynb](shape_completion_pipeline_demo_final.ipynb)

#### Run Zero-shot completion on ShapeNet
See [shape_completion_pipeline_demo_shapenet.ipynb](shape_completion_pipeline_demo_shapenet.ipynb)
