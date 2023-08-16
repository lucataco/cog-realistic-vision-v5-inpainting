# Prediction interface for Cog ⚙️
from cog import BasePredictor, Input, Path
import os
import math
import torch
from PIL import Image
from diffusers import AutoencoderKL, StableDiffusionInpaintPipeline

MODEL_NAME = "SG161222/Realistic_Vision_V5.0_noVAE"
MODEL_CACHE = "cache"
VAE_CACHE = "vae-cache"

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        vae = AutoencoderKL.from_single_file(
            "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors",
            cache_dir=VAE_CACHE
        )
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            MODEL_CACHE,
            vae=vae,
        )
        self.pipe = pipe.to("cuda")

    def scale_down_image(self, image_path, max_size):
        #Open the Image
        image = Image.open(image_path)
        #Get the Original width and height
        width, height = image.size
        # Calculate the scaling factor to fit the image within the max_size
        scaling_factor = min(max_size/width, max_size/height)
        # Calaculate the new width and height
        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)
        #resize the image
        resized_image = image.resize((new_width, new_height))
        cropped_image = self.crop_center(resized_image)
        return cropped_image

    def crop_center(self, pil_img):
        img_width, img_height = pil_img.size
        crop_width = self.base(img_width)
        crop_height = self.base(img_height)
        return pil_img.crop(
                (
                    (img_width - crop_width) // 2,
                    (img_height - crop_height) // 2,
                    (img_width + crop_width) // 2,
                    (img_height + crop_height) // 2)
                )

    def base(self, x):
        return int(8 * math.floor(int(x)/8))

    def predict(
        self,
        image: Path = Input(description="Input image"),
        prompt: str = "a tabby cat, high resolution, sitting on a park bench",
        mask: Path = Input(description="Mask image"),
        negative_prompt: str = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck",
        strength: float = Input(description="strength/weight", ge=0, le=1, default=0.8),
        steps: int = Input(description=" num_inference_steps", ge=0, le=100, default=20),
        seed: int = Input(description="Leave blank to randomize",  default=None),
    ) -> Path:
        """Run a single prediction on the model"""
        if (seed == 0) or (seed == None):
            seed = int.from_bytes(os.urandom(2), byteorder='big')
        generator = torch.Generator('cuda').manual_seed(seed)
        print("Using seed:", seed)

        r_image = self.scale_down_image(image,1280)
        r_mask = self.scale_down_image(mask, 1280)
        width, height = r_image.size
        image = self.pipe(
            prompt=prompt,
            image=r_image,
            mask_image=r_mask,
            strength=strength,
            width=width,
            height=height,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            generator=generator,
        ).images[0]

        out_path = Path(f"/tmp/output.png")
        image.save(out_path)
        return  out_path
