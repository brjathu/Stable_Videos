import requests
from PIL import Image
from io import BytesIO
import torch
import os
from diffusers import DiffusionPipeline, DDIMScheduler


seed = 0
has_cuda = torch.cuda.is_available()
device = torch.device('cpu' if not has_cuda else 'cuda')
generator = torch.Generator("cuda").manual_seed(0)

pipe = DiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    safety_checker=None,
    use_auth_token=True,
    custom_pipeline="imagic_stable_diffusion",
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
).to(device)

# url = 'https://www.dropbox.com/s/6tlwzr73jd1r9yk/obama.png?dl=1'
# response = requests.get(url)
# init_image = Image.open(BytesIO(response.content)).convert("RGB")
init_image = Image.open("assets/test1.jpg").convert("RGB")
init_image = init_image.resize((512, 512))


prompt = "A photo of Barack Obama smiling with a big grin"

res = pipe.train(
    prompt,
    image=init_image,
    generator=generator)

res = pipe(alpha=1, guidance_scale=7.5, num_inference_steps=50)
image = res.images[0]
image.save('./out/imagic/imagic_image_alpha_1.png')
