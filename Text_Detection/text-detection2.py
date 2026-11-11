# Python Image library is used to open, edit, manipulate, and save images in Python.
from PIL import Image
# AutoModelForCausalLLM Automatically loads a pretrained language model for generating text
# AutoProcessor Automatically loads the required processor to prepare
#  inputs (images, text, audio, etc.) for the model.
from transformers import AutoModelForCausalLM, AutoProcessor

img = Image.open("./images/eng_bw.png").convert("RGB")

model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base",
                                             trust_remote_code=True, attn_implementation="eager")
# Loads the processor associated with the Florence-2 model.
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base",
                                          trust_remote_code=True)
# Prepares the image and task instruction for the model.
# OCR_WITH_REGION means: Extract text and identify its location in the image.
# return_tensors="pt" converts inputs into PyTorch tensors.
inputs = processor(text="<OCR_WITH_REGION>", images=img, return_tensors="pt")

# Runs the model to generate predictions.
# max_new_tokens=1024 sets the maximum number of tokens to generate.
outputs = model.generate(**inputs, max_new_tokens=1024,use_cache=False)

# Decodes the model's output tokens into readable text.
# skip_special_tokens=True removes special model tokens.
text = processor.batch_decode(outputs, skip_special_tokens=False)[0]

# Converts the model's raw output into structured OCR results.
# image_size tells the processor the original image dimensions.
result = processor.post_process_generation(text, task="<OCR_WITH_REGION>", image_size=(img.width, img.height))

print("This is the result")
for label, box in zip(result["<OCR_WITH_REGION>"]["labels"], result["<OCR_WITH_REGION>"]["quad_boxes"]):
    print(label.replace("</s>", "").strip(), box)
