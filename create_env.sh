conda create -n cos526 python=3.10 -y
conda activate cos526
# conda install pip -y
# pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
# pip install pytorch3d
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
conda install -c bottler nvidiacub -y
pip install scikit-image matplotlib imageio plotly opencv-python
pip install black usort flake8 flake8-bugbear flake8-comprehensions
conda install pytorch3d -c pytorch3d -y
# conda install pyg -c pyg -y
# pip install point_cloud_utils
pip install open3d
pip install transformers
pip install diffusers
pip install tqdm
pip install ninja
pip install pybind11
pip install einops
pip install opencv-python
pip install omegaconf
pip install trimesh pymeshlab pygltflib xatlas
pip install accelerate
pip install gradio fastapi uvicorn rembg onnxruntime

cd Hunyuan3D-2
# pip install -r requirements.txt
pip install -e .
# # for texture
# cd hy3dgen/texgen/custom_rasterizer
# python3 setup.py install
# cd ../../..
# cd hy3dgen/texgen/differentiable_renderer
# python3 setup.py install
# cd ../../..
# cd ..