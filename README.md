# Zero Shot Point Cloud Completion with 3D Diffusion Priors

## Prerequisites
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

#### Run Zero-shot completion
```
python shape_completion_pipeline.py
```
