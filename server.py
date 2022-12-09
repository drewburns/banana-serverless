# Do not edit if deploying to Banana Serverless
# This file is boilerplate for the http server, and follows a strict interface.

# Instead, edit the init() and inference() functions in app.py

from sanic import Sanic, response
import subprocess
import base64
from io import BytesIO
import app as user_src

# We do the model load-to-GPU step on server startup
# so the model object is available globally for reuse
user_src.init()

# Create the http server app
server = Sanic("my_app")

# Healthchecks verify that the environment is correct on Banana Serverless
@server.route('/healthcheck', methods=["GET"])
def healthcheck(request):
    # dependency free way to check if GPU is visible
    gpu = False
    out = subprocess.run("nvidia-smi", shell=True)
    if out.returncode == 0: # success state on shell command
        gpu = True

    return response.json({"state": "healthy", "gpu": gpu})

# Inference POST handler at '/' is called for every http call from Banana
@server.route('/', methods=["POST"]) 
def inference(request):
    try:
        model_inputs = response.json.loads(request.json)
    except:
        model_inputs = request.json

    if "url" in model_inputs:
        user_src.img2img(model_inputs["prompt"], model_inputs["path"])
        return response.json("ok")
    else:
        images = []
        for i in range(4):
            output = user_src.inference(model_inputs)
            buffered = BytesIO()
            output.images[0].save(buffered,format='JPEG')
            image_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            images.append({'image_b64': image_b64})
        return response.json(images)



if __name__ == '__main__':
    server.run(host='0.0.0.0', port="8000", workers=1)