import subprocess
#
#MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
#
#command = [
#    "accelerate", "launch", "diffusers/examples/text_to_image/train_text_to_image.py",
#    f"--pretrained_model_name_or_path={MODEL_NAME}",
#    "--train_data_dir=data/raw/trial",
#    "--caption_column=additional_feature", 
#    "--output_dir=results/text_to_image_trial",
#    "--mixed_precision=fp16"
#]                                               
#
##subprocess.run(command, check=True)
#
#
## Run the command and capture stdout and stderr
#result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#
## Print standard output and error
#print("STDOUT:\n", result.stdout)
#print("STDERR:\n", result.stderr)

from diffusers import StableDiffusionPipeline
import torch

# Initialize the pipeline
pipeline = StableDiffusionPipeline.from_pretrained("riffusion/riffusion-model-v1", torch_dtype=torch.float16)
pipeline.to("cuda")

# List of prompts for generating images
prompts = [
    "a jazz song with guitar and drums",
    "a classical piano solo in a grand hall",
    "an energetic rock concert with electric guitars",
    "a relaxing hip-hop rap beat",
    "a vibrant electronic dance beat with colorful visuals",
    "a soulful saxophone performance in a smoky bar",
    "a folk song with banjo and violin in the countryside",
    "a heavy metal track with loud drums and guitar solos",
    "a calm meditation track with flutes and soft tones",
    "an electronic techno beat with strong bass"
]

# Directory to save the generated images
output_dir = "generated_images"
import os
os.makedirs(output_dir, exist_ok=True)

# Generate and save images
for prompt in prompts:
    # Generate the image
    image = pipeline(prompt).images[0]
    
    # Clean the prompt to use as filename
    filename = os.path.join(output_dir, prompt.replace(" ", "_").replace("/", "_") + ".png")
    
    # Save the image
    image.save(filename)
    print(f"Saved: {filename}")
