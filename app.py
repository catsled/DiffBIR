"""
Author: @mashun - sugon
email: mashun1@sugon.com
date: 2024/09/13
ref: https://github.com/0x3f3f3f3fun/DiffBIR-OpenXLab/blob/main/app.py
"""

from utils.common import instantiate_from_config, load_file_from_url, count_vram_usage
from omegaconf import OmegaConf
from typing import Dict

from utils.helpers import (
    Pipeline,
    BSRNetPipeline, SwinIRPipeline, SCUNetPipeline,
    bicubic_resize
)

from utils.face_restoration_helper import FaceRestoreHelper

import cv2

import torch
import numpy as np


MODELS = {
    ### stage_1 model weights
    "bsrnet": "https://github.com/cszn/KAIR/releases/download/v1.0/BSRNet.pth",
    # the following checkpoint is up-to-date, but we use the old version in our paper
    # "swinir_face": "https://github.com/zsyOAOA/DifFace/releases/download/V1.0/General_Face_ffhq512.pth",
    "swinir_face": "https://huggingface.co/lxq007/DiffBIR/resolve/main/face_swinir_v1.ckpt",
    "scunet_psnr": "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth",
    "swinir_general": "https://huggingface.co/lxq007/DiffBIR/resolve/main/general_swinir_v1.ckpt",
    ### stage_2 model weights
    "sd_v21": "https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt",
    "v1_face": "https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/v1_face.pth",
    "v1_general": "https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/v1_general.pth",
    "v2": "https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/v2.pth"
}


def to_tensor(image, device, bgr2rgb=True):
    if bgr2rgb:
        image = image[:, :, ::-1]
    image_tensor = torch.tensor(image[None] / 255.0, dtype=torch.float32, device=device).clamp_(0, 1)
    image_tensor = einops.rearrange(image_tensor, "n h w c -> n c h w").contiguous()
    return image_tensor


def to_array(image):
    image = image.clamp(0, 1)
    image_array = (einops.rearrange(image, "b c h w -> b h w c") * 255).cpu().numpy().clip(0, 255).astype(np.uint8)
    return image_array


def load_model_from_url(url: str) -> Dict[str, torch.Tensor]:
    sd_path = load_file_from_url(url, model_dir="weights")
    sd = torch.load(sd_path, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    if list(sd.keys())[0].startswith("module"):
        sd = {k[len("module."):]: v for k, v in sd.items()}
    return sd


# 加载通用模型
cldm = instantiate_from_config(OmegaConf.load("configs/inference/cldm.yaml"))
sd = load_model_from_url(MODELS["sd_v21"])
unused = cldm.load_pretrained_sd(sd)
control_sd = load_model_from_url(MODELS["v2"])
cldm.load_controlnet_from_ckpt(control_sd)
cldm.eval().cuda()
diffusion = instantiate_from_config(OmegaConf.load("configs/inference/diffusion.yaml")).cuda()


def inference_sr(image, 
                 steps, 
                 upscale,
                 tiled: bool = False, 
                 tile_size: int = 512, 
                 tile_stride: int = 256, 
                 pos_prompt="", 
                 neg_prompt = "low quality, blurry, low-resolution, noisy, unsharp, weird textures", 
                 cfg_scale: float = 4.0, 
                 better_start: bool = False
                ):
        bsrnet = instantiate_from_config(OmegaConf.load("configs/inference/bsrnet.yaml"))
        sd = load_model_from_url(MODELS["bsrnet"])
        bsrnet.load_state_dict(sd, strict=True)
        bsrnet.eval().cuda()
        pipeline = BSRNetPipeline(bsrnet, cldm, diffusion, None, "cuda", upscale)

        with torch.no_grad():
            sample = pipeline.run(image[None], steps, 1.0, tiled, tile_size, tile_stride, pos_prompt, neg_prompt, cfg_scale, better_start)[0]
    
        pipeline_.append(("sr", pipeline))
    
        return sample
        

def inference_dn(image, 
                 steps, 
                 upscale,
                 tiled: bool = False, 
                 tile_size: int = 512, 
                 tile_stride: int = 256, 
                 pos_prompt="", 
                 neg_prompt = "low quality, blurry, low-resolution, noisy, unsharp, weird textures", 
                 cfg_scale: float = 4.0, 
                 better_start: bool = False):

        scunet_psnr = instantiate_from_config(OmegaConf.load("configs/inference/scunet.yaml"))
        sd = load_model_from_url(MODELS["scunet_psnr"])
        scunet_psnr.load_state_dict(sd, strict=True)
        scunet_psnr.eval().cuda()
        pipeline = SCUNetPipeline(scunet_psnr, cldm, diffusion, None, "cuda")
    
        image = bicubic_resize(image, upscale)

        with torch.no_grad():
            sample = pipeline.run(image[None], steps, 1.0, tiled, tile_size, tile_stride, pos_prompt, neg_prompt, cfg_scale, better_start)[0]
    
        pipeline_.append(("dn", pipeline))
    
        return sample


def inference_fr(image, 
                 steps, 
                 upscale,
                 tiled: bool = False, 
                 tile_size: int = 512, 
                 tile_stride: int = 256, 
                 pos_prompt="", 
                 neg_prompt = "low quality, blurry, low-resolution, noisy, unsharp, weird textures", 
                 cfg_scale: float = 4.0, 
                 better_start: bool = False):
        swinir_face = instantiate_from_config(OmegaConf.load("configs/inference/swinir.yaml"))
        sd = load_model_from_url(MODELS["swinir_face"])
        swinir_face.load_state_dict(sd, strict=True)
        swinir_face.eval().cuda()
        pipeline = SwinIRPipeline(swinir_face, cldm, diffusion, None, "cuda")
    
        image = bicubic_resize(image, upscale)

        with torch.no_grad():
            sample = pipeline.run(image[None], steps, 1.0, tiled, tile_size, tile_stride, pos_prompt, neg_prompt, cfg_scale, better_start)[0]
    
        pipeline_.append(("fr", pipeline))
    
        return sample


def inference_fr_bg(image, 
                     steps, 
                     upscale,
                     tiled: bool = False, 
                     tile_size: int = 512, 
                     tile_stride: int = 256, 
                     pos_prompt="", 
                     neg_prompt = "low quality, blurry, low-resolution, noisy, unsharp, weird textures", 
                     cfg_scale: float = 4.0, 
                     better_start: bool = False):

    bsrnet = instantiate_from_config(OmegaConf.load("configs/inference/bsrnet.yaml"))
    sd = load_model_from_url(MODELS["bsrnet"])
    bsrnet.load_state_dict(sd, strict=True)
    bsrnet.eval().cuda()

    swinir_face = instantiate_from_config(OmegaConf.load("configs/inference/swinir.yaml"))
    sd = load_model_from_url(MODELS["swinir_face"])
    swinir_face.load_state_dict(sd, strict=True)
    swinir_face.eval().cuda()

    pipes = {
        "bg": BSRNetPipeline(bsrnet, cldm, diffusion, None, "cuda", upscale),
        "face": SwinIRPipeline(swinir_face, cldm, diffusion, None, "cuda")
    }

    face_helper = FaceRestoreHelper(
        device="cuda",
        upscale_factor=1,
        face_size=512,
        use_parse=True,
        det_model="retinaface_resnet50"
    )
    
    pipeline = pipes["face"]
    loop_ctx = {}

    def _process():
        face_helper.clean_all()
        upscaled_bg = bicubic_resize(image, upscale)
        face_helper.read_image(upscaled_bg)
        ### get face landmarks for each face
        face_helper.get_face_landmarks_5(resize=640, eye_dist_threshold=5)
        face_helper.align_warp_face()
        print(f"detect {len(face_helper.cropped_faces)} faces")
        ### restore each face (has been upscaeled)
        for i, lq_face in enumerate(face_helper.cropped_faces):
            loop_ctx["is_face"] = True
            loop_ctx["face_idx"] = i
            loop_ctx["cropped_face"] = lq_face
            yield lq_face
        ### restore background (hasn't been upscaled)
        loop_ctx["is_face"] = False
        yield image

    def after_load_lq():
        if loop_ctx["is_face"]:
            pipeline = pipes["face"]
        else:
            pipeline = pipes["bg"]
        return pipeline

    for lq in _process():
        pipeline = after_load_lq()
        sample = pipeline.run(
            lq[None], steps, 1.0, tiled,
            tile_size, tile_stride,
            pos_prompt, neg_prompt, cfg_scale,
            better_start
        )[0]
        if loop_ctx["is_face"]:
            face_helper.add_restored_face(sample)
        else:
            face_helper.get_inverse_affine()
            sample = face_helper.paste_faces_to_input_image(
                upsample_img=sample
            )

            return sample


def inference(mode,
              image, 
              steps, 
              tiled: bool = False, 
              tile_size: int = 512, 
              tile_stride: int = 256, 
              pos_prompt="", 
              neg_prompt = "low quality, blurry, low-resolution, noisy, unsharp, weird textures", 
              cfg_scale: float = 4.0, 
              better_start: bool = False):
    image = np.array(image)
    if mode == "sr-超分":
        sample = inference_sr(image, 
                             steps, 
                             tiled,
                             tile_size,
                             tile_stride,
                             pos_prompt,
                             neg_prompt,
                             cfg_scale, 
                             better_start)
    elif mode == "dn-降噪":
        sample = inference_dn(image, 
                             steps, 
                             tiled,
                             tile_size,
                             tile_stride,
                             pos_prompt,
                             neg_prompt,
                             cfg_scale, 
                             better_start)
    elif mode == "fr-对齐脸部修复":
        sample = inference_fr(image, 
                             steps, 
                             tiled,
                             tile_size,
                             tile_stride,
                             pos_prompt,
                             neg_prompt,
                             cfg_scale, 
                             better_start)
    elif mode == "fr_bg-未对齐脸部修复":
        sample = inference_fr_bg(image, 
                                 steps, 
                                 tiled,
                                 tile_size,
                                 tile_stride,
                                 pos_prompt,
                                 neg_prompt,
                                 cfg_scale, 
                                 better_start)
    else:
        pass

    return sample


import gradio as gr

MARKDOWN = \
"""
## DiffBIR: Towards Blind Image Restoration with Generative Diffusion Prior

[GitHub](https://github.com/XPixelGroup/DiffBIR) | [Paper](https://arxiv.org/abs/2308.15070) | [Project Page](https://0x3f3f3f3fun.github.io/projects/diffbir/)

If DiffBIR is helpful for you, please help star the GitHub Repo. Thanks!

## NOTE

1. This app processes user-uploaded images in sequence, so it may take some time before your image begins to be processed.
2. This is a publicly-used app, so please don't upload large images (>= 1024) to avoid taking up too much time.

## 注意：如果出现错误，大概率是因为传入的图像过大，可以调整放大倍数或者使用Tiled。
"""

with gr.Blocks() as block:
    with gr.Row():
        gr.Markdown(MARKDOWN)
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(sources=['upload'], type="pil")
            mode = gr.Radio(['sr-超分', 'dn-降噪', 'fr-对齐脸部修复', 'fr_bg-未对齐脸部修复'])
            run_button = gr.Button()
            with gr.Accordion("Options", open=True):
                upscale = gr.Slider(label="放大倍数", minimum=1, maximum=4, value=1, step=1)
                tiled = gr.Checkbox(label="Tiled", value=False)
                tile_size = gr.Slider(label="Tile Size", minimum=512, maximum=1024, value=512, step=256)
                tile_stride = gr.Slider(label="Tile Stride", minimum=256, maximum=512, value=256, step=128)
                pos_prompt = gr.Textbox(label="Positive Prompt", value="")
                neg_prompt = gr.Textbox(
                    label="Negative Prompt",
                    value="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
                )
                cfg_scale = gr.Slider(label="Classifier Free Guidance Scale (Set to a value larger than 1 to enable it!)", minimum=0.1, maximum=30.0, value=1.0, step=0.1)
                steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=50, step=1)
        with gr.Column():
            output_image = gr.Image(type="pil")

    run_button.click(fn=inference, inputs=[mode, input_image, steps, upscale, tiled, tile_size, tile_stride, pos_prompt, neg_prompt, cfg_scale], outputs=[output_image])

block.launch(server_name="0.0.0.0")
