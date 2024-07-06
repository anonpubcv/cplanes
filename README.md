


# Environment Setup
```
    # create conda environment
    conda create --name cplanes python=3.8
    
    # activate env
    conda activate cplanes
    conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1  cudatoolkit=11.6 -c pytorch -c conda-forge

    # pip install 
    pip install -r requirements.txt
    python setup.py develop
    pip install tensorboard
    pip install imageio[ffmpeg]

```
[D-NeRF dataset](https://github.com/albertpumarola/D-NeRF)
[Plenoptic Dataset](https://github.com/facebookresearch/Neural_3D_Video)

Please change the "datadir" in config based on the locations of the datasets.
Code base follows original Hexplane implementation. 
# Reconstruction
```
python main_multi.py config=dnerf_general.yaml


# Renduring
With `render_test=True`, `render_path=True`, results at test viewpoint are automatically evaluated and validation viewpoints are generated after reconstruction.  

Or

python main_multi.py config=dnerf_general.yaml systems.ckpt="checkpoint/path" render_only=True

