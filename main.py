from diffusers import StableDiffusionPipeline
from PIL import Image


with open("huggingface_token", "r") as infile:
    use_auth_token = infile.read().strip()


# get your token at https://huggingface.co/settings/tokens
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=use_auth_token)
prompt = "a photograph of an astronaut riding a horse"

image = pipe(prompt)["sample"][0]

save_name = "output.png"

image.save(save_name)
image = Image.open(save_name)
image.show()

# you can save the image with

