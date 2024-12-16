import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import make_image_grid, load_image

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "riffusion/riffusion-model-v1", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

# path to input image
path = "data/processed/1000dataset_5/specs/0C80GCp0mMuBzLf3EAXqxv_segment_2.png"
pathn = path[:-4]

init_image = load_image(path)

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

# pass prompt and image to pipeline
image = pipeline(prompt, image=init_image).images[0]
make_image_grid([init_image, image], rows=1, cols=2)
# Save the image with the name of the image and the prompt
imgname = pathn.split("/")
img = imgname[4]
img_lst = img.split("_")
img_name = img_lst[0]+"_"+"s"+img_lst[2]
# Save the image with name: img_name + prompt
image.save(f"generated_images/{img_name}_{prompt}.png")






### ControlNet
from diffusers.utils import load_image, make_image_grid

# prepare image
path = "data/processed/1000dataset_5/specs/0C80GCp0mMuBzLf3EAXqxv_segment_2.png"
pathn = path[:-4]

init_image = load_image(path)
init_image = init_image.resize((958, 960)) # resize to depth image dimensions
depth_image = load_image("https://huggingface.co/lllyasviel/control_v11f1p_sd15_depth/resolve/main/images/control.png")
make_image_grid([init_image, depth_image], rows=1, cols=2)


