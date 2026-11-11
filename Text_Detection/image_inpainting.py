from PIL import Image
import torch
# StableDiffusionInpaintPipeline is used to edit or replace a specific part of an existing image using AI
from diffusers import StableDiffusionInpaintPipeline

# StableDiffusionInpaintPipeline run pytorch

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float32
)

pipe = pipe.to("cpu")

image = Image.open("./images/CityCar.jpg").convert("RGB").resize((512,512))
mask = Image.open("mask_car.png").convert("L").resize((512,512))

prompt = "empty road with green grass and trees along the roadside, no car, no vehicles, no traffic"

result = pipe(
    prompt=prompt,
    image=image,
    mask_image=mask
).images[0]

result.save("inpatient_output.png")