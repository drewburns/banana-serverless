# from transformers import pipeline
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
import torch
import requests
from PIL import Image
from io import BytesIO


# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    global imgmodel
    
    model = StableDiffusionPipeline.from_pretrained('andrewburns/emoji_ema', torch_dtype=torch.float16)
    model = model.to('cuda')

    imgmodel = StableDiffusionImg2ImgPipeline.from_pretrained('andrewburns/emoji_ema', torch_dtype=torch.float16)

    def null_safety(images, **kwargs):
        return images, False
    model.safety_checker = null_safety
    imgmodel.safety_checker = null_safety
    # model.to("cuda")


    # model = pipeline('fill-mask', model='andrewburns/emoji_v2', device=device)

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    if prompt == None:
        return {'message': "No prompt provided"}
    
    # Run the model
    result = model(prompt)

    # Return the results as a dictionary
    return result

def img2img(prompt, path):
    # pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    #     "runwayml/stable-diffusion-v1-5", revision="fp16", torch_dtype=torch.float16
    # ).to(device)

    # let's download an initial image
    url = path

    response = requests.get(url)
    init_image = Image.open(BytesIO(response.content)).convert("RGB")
    init_image.thumbnail((125, 125))


    images = imgmodel(prompt=prompt, image=init_image, strength=0.75, guidance_scale=4).images
    return images[0]