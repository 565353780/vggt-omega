cd ..
git clone git@github.com:565353780/camera-control.git
git clone git@github.com:565353780/colmap-manage.git

cd camera-control
./dev_setup.sh

cd ../colmap-manage
./dev_setup.sh

cd ../vggt-omega

pip install tqdm hydra-core omegaconf opencv-python \
  scipy onnxruntime requests matplotlib pillow \
  huggingface_hub einops safetensors plyfile

pip install numpy==1.26.4
pip install pycolmap==3.10.0
#pip install gradio==5.50.0
#pip install gradio-client==1.14.0
