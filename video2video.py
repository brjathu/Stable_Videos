import os
import sys

# import gradio as gr
import cv2
import imgviz
import joblib
import numpy as np
import torch
from einops import rearrange, repeat
from imwatermark import WatermarkEncoder
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from stablediffusion.ldm.data.util import AddMiDaS
from stablediffusion.ldm.models.diffusion.ddim import DDIMSampler
from stablediffusion.ldm.util import instantiate_from_config
from stablediffusion.scripts.txt2img import put_watermark

torch.set_grad_enabled(False)

def task_divider(data, batch_id, num_task):

    batch_length = len(data)//num_task
    start_       = batch_id*(batch_length+1)
    end_         = (batch_id+1)*(batch_length+1)
    if(start_>len(data)): exit()
    if(end_  >len(data)): end_ = len(data)
    data    = data[start_:end_] if batch_id>=0 else data

    return data   

def blockPrinting(func):
    def func_wrapper(*args, **kwargs):
        # block all printing to the console
        sys.stdout = open(os.devnull, 'w')
        # call the method in question
        value = func(*args, **kwargs)
        # enable all printing to the console
        sys.stdout = sys.__stdout__
        # pass the return value of the method back
        return value

    return func_wrapper

@blockPrinting
def make_batch_sd(
        image,
        txt,
        device,
        num_samples=1,
        model_type="dpt_hybrid"
):
    image = np.array(image.convert("RGB"))
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
    # sample['jpg'] is tensor hwc in [-1, 1] at this point
    midas_trafo = AddMiDaS(model_type=model_type)
    batch = {
        "jpg": image,
        "txt": num_samples * [txt],
    }
    batch = midas_trafo(batch)
    batch["jpg"] = rearrange(batch["jpg"], 'h w c -> 1 c h w')
    batch["jpg"] = repeat(batch["jpg"].to(device=device),
                          "1 ... -> n ...", n=num_samples)
    batch["midas_in"] = repeat(torch.from_numpy(batch["midas_in"][None, ...]).to(
        device=device), "1 ... -> n ...", n=num_samples)
    return batch

@blockPrinting
def paint(sampler, image, prompt, t_enc, seed, scale, num_samples=1, callback=None,
          do_full_sample=False):
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = sampler.model
    seed_everything(seed)

    print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    wm = "SDV2"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    with torch.no_grad(),\
            torch.autocast("cuda"):
        batch = make_batch_sd(
            image, txt=prompt, device=device, num_samples=num_samples)
        z = model.get_first_stage_encoding(model.encode_first_stage(
            batch[model.first_stage_key]))  # move to latent space
        c = model.cond_stage_model.encode(batch["txt"])
        c_cat = list()
        for ck in model.concat_keys:
            cc = batch[ck]
            cc = model.depth_model(cc)
            depth_min, depth_max = torch.amin(cc, dim=[1, 2, 3], keepdim=True), torch.amax(cc, dim=[1, 2, 3],
                                                                                           keepdim=True)
            display_depth = (cc - depth_min) / (depth_max - depth_min)
            depth_image = Image.fromarray(
                (display_depth[0, 0, ...].cpu().numpy() * 255.).astype(np.uint8))
            cc = torch.nn.functional.interpolate(
                cc,
                size=z.shape[2:],
                mode="bicubic",
                align_corners=False,
            )
            depth_min, depth_max = torch.amin(cc, dim=[1, 2, 3], keepdim=True), torch.amax(cc, dim=[1, 2, 3],
                                                                                           keepdim=True)
            cc = 2. * (cc - depth_min) / (depth_max - depth_min) - 1.
            c_cat.append(cc)
        c_cat = torch.cat(c_cat, dim=1)
        # cond
        cond = {"c_concat": [c_cat], "c_crossattn": [c]}

        # uncond cond
        uc_cross = model.get_unconditional_conditioning(num_samples, "")
        uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}
        if not do_full_sample:
            # encode (scaled latent)
            z_enc = sampler.stochastic_encode(
                z, torch.tensor([t_enc] * num_samples).to(model.device))
        else:
            z_enc = torch.randn_like(z)
        # decode it
        samples = sampler.decode(z_enc, cond, t_enc, unconditional_guidance_scale=scale,
                                 unconditional_conditioning=uc_full, callback=callback)
        x_samples_ddim = model.decode_first_stage(samples)
        result = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
        result = result.cpu().numpy().transpose(0, 2, 3, 1) * 255
    return [depth_image] + [put_watermark(Image.fromarray(img.astype(np.uint8)), wm_encoder) for img in result]

@blockPrinting
def pad_image(input_image):
    pad_w, pad_h = np.max(((2, 2), np.ceil(
        np.array(input_image.size) / 64).astype(int)), axis=0) * 64 - input_image.size
    im_padded = Image.fromarray(
        np.pad(np.array(input_image), ((0, pad_h), (0, pad_w), (0, 0)), mode='edge'))
    w, h = im_padded.size
    if w == h:
        return im_padded
    elif w > h:
        new_image = Image.new(im_padded.mode, (w, w), (0, 0, 0))
        new_image.paste(im_padded, (0, (w - h) // 2))
        return new_image
    else:
        new_image = Image.new(im_padded.mode, (h, h), (0, 0, 0))
        new_image.paste(im_padded, ((h - w) // 2, 0))
        return new_image

@blockPrinting
def predict(input_image, prompt, steps, num_samples, scale, seed, eta, strength):
    num_samples = 1
    init_image = input_image.convert("RGB")
    image = pad_image(init_image)  # resize to integer multiple of 32
    image = image.resize((512, 512))
    sampler.make_schedule(steps, ddim_eta=eta, verbose=True)
    assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
    do_full_sample = strength == 1.
    t_enc = min(int(strength * steps), steps-1)
    result = paint(
        sampler=sampler,
        image=image,
        prompt=prompt,
        t_enc=t_enc,
        seed=seed,
        scale=scale,
        num_samples=num_samples,
        callback=None,
        do_full_sample=do_full_sample
    )
    return result




def initialize_model(config, ckpt):
    config = OmegaConf.load(config)
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(ckpt)["state_dict"], strict=False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)
    return sampler

os.makedirs(".out", exist_ok=True)

sampler = initialize_model(sys.argv[1], sys.argv[2])


prompt = "A high resolution photo of a man running in the forest."
ddim_steps = 50
num_samples = 1
scale = 9.0
seed = 42
eta = 0.0
strength = 0.9

video_path = "assets/videos/youtube_run_000/img/"
save_path = "out/v0_test3.mp4"
list_of_frames = np.sort([i for i in os.listdir(video_path) if ".jpg" in i])
for fid, fname in enumerate(list_of_frames[:10]):
    print("frame number : ", fid)
    input_image = Image.open(video_path + fname)
    result = predict(input_image, prompt, ddim_steps, num_samples, scale, seed, eta, strength)
    output_image = imgviz.tile([np.array(input_image.resize(result[1].size)), np.array(result[0].resize(result[1].size)), np.array(result[1])], shape=(1, 3), border=(255, 255, 255))
    output_image = output_image[:, :, ::-1]
    if(fid==0):
        video_file = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, frameSize=(output_image.shape[1],output_image.shape[0]))
    video_file.write(output_image)
video_file.release()



# def video2video(root_dir, list_of_frames, save_path):
#     prompt = "A high resolution photo."
#     ddim_steps = 50
#     num_samples = 1
#     scale = 9.0
#     seed = 42
#     eta = 0.0
#     strength = 0.9

#     list_of_frames = np.sort(list_of_frames)
#     for fid, fname in enumerate(list_of_frames):
#         print("frame number : ", fid, len(list_of_frames))
#         input_image = Image.open(root_dir + fname)
#         result = predict(input_image, prompt, ddim_steps, num_samples, scale, seed, eta, strength)
#         output_image = imgviz.tile([np.array(input_image.resize(result[1].size)), np.array(result[0].resize(result[1].size)), np.array(result[1])], shape=(1, 3), border=(255, 255, 255))
#         output_image = output_image[:, :, ::-1]
#         if(fid==0):
#             video_file = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, frameSize=(output_image.shape[1],output_image.shape[0]))
#         video_file.write(output_image)
#     video_file.release()


# batch_id = -1
# dataset_root_dir = '/datasets01/AVA/080720/frames/'
# list_of_videos_   = np.load("../../3D/TENET/_DATA/ava_train.npy")
# list_of_annots   = joblib.load("../../3D/TENET/_DATA/ava_detections_train.pkl")

# list_of_videos = list_of_videos_.copy()
# np.random.seed(0)
# np.random.shuffle(list_of_videos)

# list_of_videos = task_divider(list_of_videos, batch_id, 500)
        
# for video_num, video_name in enumerate(list_of_videos[::200]):
#     print("frame number : ", video_num, len(list_of_videos))
    
#     video_id        = video_name.split("/")[0]
#     video_seq       = video_name.split("/")[1].split(".")[0]
#     video_key       = video_id + "/" + video_seq + ".jpg"
#     subfolder       = "" #"/" + class_label + "/"
#     base_path       = dataset_root_dir + "/" + video_id + "/"
#     video_seq       = video_seq
#     sample          = ''

#     list_of_frames  = []
#     list_of_annots  = []
#     for i in range(-64,64):
#         frame_id   = int(video_seq.split("_")[-1])
#         if frame_id+i<=0:
#             frame_id = 1
#         elif frame_id+i>27030:
#             frame_id = 27030
#         else:
#             frame_id = frame_id+i
#         # frame_id   = 0 if frame_id+i<0 else frame_id+i
#         frame_name = video_id + "_" + '%06d.jpg'%(frame_id,)
#         frame_key  = video_id + "/" + video_id + "_" + '%06d.jpg'%(frame_id,)
#         if frame_name not in list_of_frames:
#             list_of_frames.append(frame_name)
#             try:
#                 list_of_annots.append(list_of_annots[frame_key])
#             except:
#                 list_of_annots.append([])

#     # import ipdb; ipdb.set_trace()

#     video2video(base_path, list_of_frames, "out/ava_v2v/" + video_key.replace("/","_") + ".mp4")