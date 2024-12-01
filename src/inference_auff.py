import IPython, torch
import soundfile as sf
from huggingface_hub import split_torch_state_dict_into_shards, cashed_download
print(split_torch_state_dict_into_shards, cashed_download)
from auffusion_pipeline import AuffusionPipeline

pipeline = AuffusionPipeline.from_pretrained("auffusion/auffusion")
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
output_dir = "generated_output/auffusion"
import os
os.makedirs(output_dir, exist_ok=True) 

# Generate and save images
for prompt in prompts:
    # Generate the audio
    output = pipeline(prompt=prompt)
    audio = output.audios[0]
    
    # Save the audio in output_dir with the prompt as filename
    filename = os.path.join(output_dir, prompt.replace(" ", "_").replace("/", "_") + ".wav")
    sf.write(filename, audio.cpu().numpy(), 16000)