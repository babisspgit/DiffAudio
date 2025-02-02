import os
import glob
from PIL import Image
import numpy as np

# Directories
INPUT_DIR = "/work3/s222948/data/processed/inpaintingtest"
OUTPUT_DIR = "/work3/s222948/data/processed/inpaintingtest/masks"
os.makedirs(OUTPUT_DIR, exist_ok=True)

#IMAGES_SUBDIR = os.path.join(OUTPUT_DIR, "images")
MASKS_END_SUBDIR = os.path.join(OUTPUT_DIR, "masks_end")
MASKS_MID_SUBDIR = os.path.join(OUTPUT_DIR, "masks_mid")

# Create subfolders
#os.makedirs(IMAGES_SUBDIR, exist_ok=True)
os.makedirs(MASKS_END_SUBDIR, exist_ok=True)
os.makedirs(MASKS_MID_SUBDIR, exist_ok=True)

def generate_end_mask_random(width, height, total_seconds=5):
    import random
    mask_seconds = random.choice([1, 2])
    fraction = mask_seconds / total_seconds
    mask_width = int(round(width * fraction))
    mask = np.zeros((height, width), dtype=np.uint8)
    left_col = width - mask_width
    mask[:, left_col:] = 1
    return mask

def generate_mid_mask_random(width, height, total_seconds=5):
    import random
    mask_seconds = random.choice([1, 2])
    fraction = mask_seconds / total_seconds
    mask_width = int(round(width * fraction))
    mask = np.zeros((height, width), dtype=np.uint8)
    left_col = random.randint(0, width - mask_width)
    mask[:, left_col:left_col + mask_width] = 1
    return mask


spectrogram_paths = glob.glob(os.path.join(INPUT_DIR, "*.png"))

for idx, path in enumerate(spectrogram_paths):
    # Load the original spectrogram
    original_img = Image.open(path).convert("RGB")
    width, height = original_img.size
    
    # Save the original image into the dataset's 'images' folder
    # (You may prefer to keep your original names, but here's a renaming approach.)
    # The file_stem2 is the first string of the path before the first . character
    file_stem = path.split("/")[-1].split(".")[0]
    #file_stem2 = f"{idx:05d}"
    #img_out_path = os.path.join(IMAGES_SUBDIR, f"image_{file_stem}.png")
    #original_img.save(img_out_path)
    
    # Generate the two masks
    mask_end = generate_end_mask_random(width, height, total_seconds=5)
    mask_mid = generate_mid_mask_random(width, height, total_seconds=5)
    
    # Convert each mask from 0/1 to a black-and-white image
    mask_end_img = Image.fromarray((mask_end * 255).astype(np.uint8))
    mask_mid_img = Image.fromarray((mask_mid * 255).astype(np.uint8))
    
    # Save them in separate subfolders
    mask_end_path = os.path.join(MASKS_END_SUBDIR, f"mask_end_{file_stem}.png")
    mask_mid_path = os.path.join(MASKS_MID_SUBDIR, f"mask_mid_{file_stem}.png")
    
    mask_end_img.save(mask_end_path)
    mask_mid_img.save(mask_mid_path)

print("Masks generated and saved")