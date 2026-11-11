# Transformers comes from Hugging Face (An AI platform which provide pretrained AI models and Datasets)

# Transformers Provides access to pretrained AI models for tasks such as
# OCR, image classification, and text processing.

# pipeline is an function of transformers that allows you to use pretrained AI
# models easily without manually writing all the preprocessing and inference code.


from transformers import pipeline

pipe = pipeline('image-to-text', model='microsoft/trocr-base-printed')

result = pipe("./images/single.png")
print("This is the result")
print(result[0]['generated_text'])
