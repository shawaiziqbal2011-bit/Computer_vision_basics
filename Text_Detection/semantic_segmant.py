from transformers import pipeline
from PIL import Image

model = pipeline("image-segmentation",model="facebook/mask2former-swin-large-ade-semantic")

image = Image.open("./images/CityCar.jpg").convert("RGB")

result = model(image)

for segment in result:
    print(f"Label: {segment['label']}, Score: {segment['score']:.2f}")
    segment['mask'].save(f"mask_{segment['label']}.png")