## Install the dependencies. 


`conda env create -f env.yaml`

`conda activate StableVideos`

## Create your hugging face account and generate a key.

` huggingface-cli login`


## Download midas weights

`curl -LJ https://github.com/isl-org/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt --create-dirs -o weights/midas_models/dpt_hybrid-midas-501f0c75.pt`

`curl -LJ https://github.com/isl-org/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt --create-dirs -o weights/midas_models/dpt_large-midas-2f21e586.pt`

`wget https://huggingface.co/stabilityai/stable-diffusion-2-depth/blob/main/512-depth-ema.ckpt`

`mv 512-depth-ema.ckpt weights/512-depth-ema.ckpt`

`export PYTHONPATH="${PYTHONPATH}:/home/jathu/Diffusion/Stable_Videos/stablediffusion"`


point to correct midas path
<!-- /home/jathu/Diffusion/Stable_Videos/stablediffusion/ldm/modules/midas/api.py -->