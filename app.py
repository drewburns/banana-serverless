# from transformers import pipeline
from diffusers import StableDiffusionPipeline

import torch

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    
    model = StableDiffusionPipeline.from_pretrained('andrewburns/emoji_ema', torch_dtype=torch.float16)
    model = model.to('cuda')

    def null_safety(images, **kwargs):
        return images, False
    model.safety_checker = null_safety
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
