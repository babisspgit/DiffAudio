# Importing necessary libraries
from diffusers import StableDiffusionPipeline, DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from torch.utils.data import DataLoader
from PIL import Image
import torch
import pandas as pd
from datasets import Dataset


accelerate launch train_dreambooth.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
    --instance_data_dir="C:/Users\spbsp/OneDrive - Danmarks Tekniske Universitet/Skrivebord/DiffAudio/data/raw/1000_mel" \
    --output_dir="./spectrogram_output" \
    --instance_prompt_file="C:/Users\spbsp/OneDrive - Danmarks Tekniske Universitet/Skrivebord/DiffAudio/data/raw/1000dataset_prompts" \
    --train_text_encoder \
    --resolution=512 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --learning_rate=5e-6 \
    --max_train_steps=1000
