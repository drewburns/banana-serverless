# In this file, we define download_model
# It runs during container build time to get model weights built into the container

# In this example: A Huggingface BERT model

from diffusers import StableDiffusionPipeline
import torch
from transformers import pipeline

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    StableDiffusionPipeline.from_pretrained('andrewburns/emoji_v2', torch_dtype=torch.float16, revision="fp16")

if __name__ == "__main__":
    download_model()