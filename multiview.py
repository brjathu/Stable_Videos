# import sys
# import torch
# import numpy as np
# # import gradio as gr
# import cv2
# from PIL import Image
# from omegaconf import OmegaConf
# from einops import repeat, rearrange
# from pytorch_lightning import seed_everything
# import os
# from imwatermark import WatermarkEncoder
# import imgviz
# from stablediffusion.scripts.txt2img import put_watermark
# from stablediffusion.ldm.util import instantiate_from_config
# from stablediffusion.ldm.models.diffusion.ddim import DDIMSampler
# from stablediffusion.ldm.data.util import AddMiDaS

# torch.set_grad_enabled(False)


# def make_batch_sd(
#         image,
#         txt,
#         device,
#         num_samples=1,
#         model_type="dpt_hybrid"
# ):
#     image = np.array(image.convert("RGB"))
#     image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
#     # sample['jpg'] is tensor hwc in [-1, 1] at this point
#     midas_trafo = AddMiDaS(model_type=model_type)
#     batch = {
#         "jpg": image,
#         "txt": num_samples * [txt],
#     }
#     batch = midas_trafo(batch)
#     batch["jpg"] = rearrange(batch["jpg"], 'h w c -> 1 c h w')
#     batch["jpg"] = repeat(batch["jpg"].to(device=device), "1 ... -> n ...", n=num_samples)
#     batch["midas_in"] = repeat(torch.from_numpy(batch["midas_in"][None, ...]).to(device=device), "1 ... -> n ...", n=num_samples)
#     return batch


# def paint(sampler, image, prompt, t_enc, seed, scale, num_samples=1, callback=None,
#           do_full_sample=False):
#     device = torch.device(
#         "cuda") if torch.cuda.is_available() else torch.device("cpu")
#     model = sampler.model
#     seed_everything(seed)

#     print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
#     wm = "SDV2"
#     wm_encoder = WatermarkEncoder()
#     wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

#     with torch.no_grad(),torch.autocast("cuda"):
#         batch = make_batch_sd(image, txt=prompt, device=device, num_samples=num_samples)
#         z = model.get_first_stage_encoding(model.encode_first_stage(batch[model.first_stage_key]))  # move to latent space
#         c = model.cond_stage_model.encode(batch["txt"])
#         c_cat = list()
#         for ck in model.concat_keys:
#             cc = batch[ck]
#             cc = model.depth_model(cc)
#             depth_min, depth_max = torch.amin(cc, dim=[1, 2, 3], keepdim=True), torch.amax(cc, dim=[1, 2, 3], keepdim=True)
#             display_depth = (cc - depth_min) / (depth_max - depth_min)
#             depth_image = Image.fromarray((display_depth[0, 0, ...].cpu().numpy() * 255.).astype(np.uint8))
#             cc = torch.nn.functional.interpolate(
#                 cc,
#                 size=z.shape[2:],
#                 mode="bicubic",
#                 align_corners=False,
#             )
#             depth_min, depth_max = torch.amin(cc, dim=[1, 2, 3], keepdim=True), torch.amax(cc, dim=[1, 2, 3], keepdim=True)
#             cc = 2. * (cc - depth_min) / (depth_max - depth_min) - 1.
#             c_cat.append(cc)
#         c_cat = torch.cat(c_cat, dim=1)
#         # cond
#         cond = {"c_concat": [c_cat], "c_crossattn": [c]}

#         # uncond cond
#         uc_cross = model.get_unconditional_conditioning(num_samples, "")
#         uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}
#         if not do_full_sample:
#             # encode (scaled latent)
#             z_enc = sampler.stochastic_encode(z, torch.tensor([t_enc] * num_samples).to(model.device))
#         else:
#             z_enc = torch.randn_like(z)
#         # decode it
#         samples = sampler.decode(z_enc, cond, t_enc, unconditional_guidance_scale=scale, unconditional_conditioning=uc_full, callback=callback)
#         x_samples_ddim = model.decode_first_stage(samples)
#         result = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
#         result = result.cpu().numpy().transpose(0, 2, 3, 1) * 255
#     return [depth_image] + [put_watermark(Image.fromarray(img.astype(np.uint8)), wm_encoder) for img in result]


# def pad_image(input_image):
#     pad_w, pad_h = np.max(((2, 2), np.ceil(
#         np.array(input_image.size) / 64).astype(int)), axis=0) * 64 - input_image.size
#     im_padded = Image.fromarray(
#         np.pad(np.array(input_image), ((0, pad_h), (0, pad_w), (0, 0)), mode='edge'))
#     w, h = im_padded.size
#     if w == h:
#         return im_padded
#     elif w > h:
#         new_image = Image.new(im_padded.mode, (w, w), (0, 0, 0))
#         new_image.paste(im_padded, (0, (w - h) // 2))
#         return new_image
#     else:
#         new_image = Image.new(im_padded.mode, (h, h), (0, 0, 0))
#         new_image.paste(im_padded, ((h - w) // 2, 0))
#         return new_image


# def predict(input_image, prompt, steps, num_samples, scale, seed, eta, strength):
#     num_samples = 1
#     init_image = input_image.convert("RGB")
#     image = pad_image(init_image)  # resize to integer multiple of 32
#     image = image.resize((512, 512))
#     sampler.make_schedule(steps, ddim_eta=eta, verbose=True)
#     assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
#     do_full_sample = strength == 1.
#     t_enc = min(int(strength * steps), steps-1)
#     result = paint(
#         sampler=sampler,
#         image=image,
#         prompt=prompt,
#         t_enc=t_enc,
#         seed=seed,
#         scale=scale,
#         num_samples=num_samples,
#         callback=None,
#         do_full_sample=do_full_sample
#     )
#     return result




# def initialize_model(config, ckpt):
#     config = OmegaConf.load(config)
#     model = instantiate_from_config(config.model)
#     model.load_state_dict(torch.load(ckpt)["state_dict"], strict=False)

#     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#     model = model.to(device)
#     sampler = DDIMSampler(model)
#     return sampler

# os.makedirs(".out", exist_ok=True)

# sampler = initialize_model(sys.argv[1], sys.argv[2])


# prompt = "A high resolution photo of a cat in front view."
# ddim_steps = 50
# num_samples = 1
# scale = 5.0
# seed = 42
# eta = 0.0
# strength = 0.9
# input_image = Image.open("assets/cat.jpg")

# result = predict(input_image, prompt, ddim_steps, num_samples, scale, seed, eta, strength)
# result[0].save("out/depth.jpg")
# result[1].save("out/img.jpg")












import open3d as o3d
import numpy as np
import glob

intrinsic = o3d.io.read_pinhole_camera_intrinsic("assets/intrinsics.json")

c_imgs = "out/img.jpg"
d_imgs = "out/depth.jpg"

import ipdb; ipdb.set_trace()


color = o3d.io.read_image(c_imgs)
depth = o3d.io.read_image(d_imgs)
idepth = depth - np.amin(depth)
idepth /= np.amax(idepth)

focal = intrinsic.intrinsic_matrix[0, 0]

# for idx in range(len(c_imgs)):
#     color = o3d.read_image(c_imgs[idx])
#     idepth = read_pfm(d_imgs[idx])[0]

#     idepth = idepth - np.amin(idepth)
#     idepth /= np.amax(idepth)

#     focal = intrinsic.intrinsic_matrix[0, 0]
#     depth = focal / (idepth)
#     depth[depth >= 5 * focal] = np.inf

#     depth = o3d.Image(depth)

#     rgbdi = o3d.create_rgbd_image_from_color_and_depth(color, depth)
#     pcd = o3d.create_point_cloud_from_rgbd_image(rgbdi, intrinsic)
#     o3d.draw_geometries([pcd])