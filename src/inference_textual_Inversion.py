from diffusers import StableDiffusionPipeline
import torch


## Define prompts
base_prompt = "A heavy metal guitar riff"
inversion_prompt = "A heavy metal guitar riff in <Metallica-song> style"
#base_prompt = "A fast-paced rock melody"
#inversion_prompt = "A fast-paced rock melody in <Metallica-song> style"

## Load the base Riffusion model
pipeline = StableDiffusionPipeline.from_pretrained("riffusion/riffusion-model-v1", torch_dtype=torch.float16).to("cuda")

## (1) Generate a spectrogram with Riffusion using the base prompt
base_spectrogram = pipeline(base_prompt, num_inference_steps=50).images[0]
base_name_path = base_prompt.replace(" ", "_") + '_' + 'riffusion.png'
base_spectrogram.save("/work3/s222948/results/txt_inv/+" + base_name_path)




## (2) Generate a spectrogram with Riffusion using the base prompt
# Disable the safety checker
pipeline.safety_checker = lambda images, clip_input: (images, [False] * len(images))

# Load the trained textual inversion embedding
embedding_path = "/work3/s222948/models/Textual_Inversion/metallica_music_mixdata/Models/learned_embeds.bin"

pipeline.load_textual_inversion(embedding_path)
#inversion_pipeline.load_textual_inversion(embedding_path, token="<Metallica-song>")

print("Loaded the textual inversion embedding")

# Generate a spectrogram with the textual inversion prompt
inverted_spectrogram = pipeline(inversion_prompt, num_inference_steps=50).images[0]
# save it to a file with the name of the prompt+'text_inv'
name_path=inversion_prompt.replace(" ", "_")+ '_' + 'text_inv.png'
inverted_spectrogram.save("/work3/s222948/results/txt_inv/"+name_path)


print("Spectrograms saved ")
