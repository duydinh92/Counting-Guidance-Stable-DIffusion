# Counting Guidance Stable Diffusion
Use diffusion to generate images that contain many objects from a specified category.

## Requirements
I follow the previously published implementation of the paper [Universal Guidance for Diffusion Models](https://arxiv.org/abs/2302.07121). The setup for Anaconda environment is as follows:
1. Create new python environment with `conda env create -f environment.yaml` and activate it
2. Run `conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia`
3. Run `pip install -r requirements.txt` in this directory to install the remaining packages.
4. Clone the repo `taming-transformer` in [link](https://github.com/CompVis/taming-transformers) to [src](src) and run `pip install -e .` inside this repo.
5. Clone the repo `CLIP` in [link](https://github.com/CompVis/taming-transformers) to [src](src) and run `pip install -e .` inside this repo.
6. Run `pip install -e .` in the Stable_Diffusion repo.
7. Down load the `sd-v1-4.ckpt` checkpoint of stable diffusion model in [link](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/blob/main/sd-v1-4.ckpt).


## Demo
- To generate an image with given text prompt (for examples: "50 apples on the tables"), run:
```
bash scripts/apples.sh

```
 we can control the image generation by changing the following arguments: \
`--optim_num_steps` sets the number of recurrent steps `k` \
`--optim_forward_guidance_wt` sets the constant in `s(t)` \
`--scale` represents unconditional guidance scale 
