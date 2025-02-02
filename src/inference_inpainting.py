import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid
import os

#initial_img_path = "/work3/s222948/data/processed/inpaintingtest/1PhLYngBKbeDtdmDzCg3Pb_segment_6.png"
#initial_img_path = "/work3/s222948/data/processed/inpaintingtest/6UB9mShVLbMm0W4e6vud4C_segment_5.png"
initial_img_path = "/work3/s222948/data/processed/inpaintingtest/6UB9mShVLbMm0W4e6vud4C_segment_8.png"

img_id = initial_img_path.split("/")[-1]

mid_mask_path= "/work3/s222948/data/processed/inpaintingtest/masks/masks_mid/mask_mid_"
end_mask_path= "/work3/s222948/data/processed/inpaintingtest/masks/masks_end/mask_end_"

pipeline = AutoPipelineForInpainting.from_pretrained(
    "riffusion/riffusion-model-v1", torch_dtype=torch.float16).to("cuda")

pipeline.enable_model_cpu_offload()

# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
#pipeline.enable_xformers_memory_efficient_attention()


init_image = load_image(initial_img_path)
#mask_image_path = end_mask_path+img_id
mask_image_path = mid_mask_path+img_id
mask_image = load_image(mask_image_path)

##1sec mask at the end
#init_image = load_image("/work3/s222948/data/processed/inpaintingtest/6UB9mShVLbMm0W4e6vud4C_segment_5.png")
#mask_image = load_image("/work3/s222948/data/processed/inpaintingtest/masks/masks_end/mask_end_6UB9mShVLbMm0W4e6vud4C_segment_5.png")


generator = torch.Generator("cuda").manual_seed(92)
#prompt = "drop the beat to create anticipation for the next part"
prompt = ""
image_inpaint = pipeline(prompt=prompt, image=init_image, mask_image=mask_image, generator=generator).images[0]
#make_image_grid([init_image, image], rows=1, cols=2).show()

output_p= '/work3/s222948/results/inpainting'

mask_type = mask_image_path.split("/")[-2]  # Get the type of mask from the path

# Create folder named after prompt and img_id
folder_name = f"{prompt.replace(' ', '_')}_{img_id}_{mask_type}"  # Replace spaces with underscores for folder name
output_folder = os.path.join(output_p, folder_name)

os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist

# Save the inpainted image
inpainted_image_path = os.path.join(output_folder, f"inpainted_{img_id}")
image_inpaint.save(inpainted_image_path)

# Save the mask image
mask_image_path_out = os.path.join(output_folder, f"mask_{img_id}")
mask_image.save(mask_image_path_out)

print(f"Files saved to {output_folder}")