import os
from PIL import Image
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision
from tqdm import tqdm
from torch import optim
import logging
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

from models.ddpm import Diffusion
from models.model import UNet
import random

SEED = 1
SPECTROGRAM_FOLDER = 'C:/Users/spbsp/OneDrive - Danmarks Tekniske Universitet/Skrivebord/DTU/Thesis/Project/data/raw/spectrograms_csegm'
## Folder containing your images

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def save_images(images, path, show=True, title=None, nrow=10):
    grid = torchvision.utils.make_grid(images, nrow=nrow)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    if title is not None:
        plt.title(title)
    plt.imshow(ndarr)
    plt.axis('off')
    if path is not None:
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
    if show:
        plt.show()
    plt.close()

def load_images_from_folder(folder, transform, max_images=10):
    """Load up to max_images from the given folder and apply transformations."""
    image_paths = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith(('png', 'jpg', 'jpeg'))]
    image_paths = image_paths[:max_images]  # Load up to max_images (10 in your case)
    
    images = []
    for path in image_paths:
        image = Image.open(path).convert('RGB')  # Convert to RGB if needed
        image = transform(image)
        images.append(image)
    
    return torch.stack(images)  # Stack images into a batch

def create_result_folders(experiment_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", experiment_name), exist_ok=True)
    os.makedirs(os.path.join("results", experiment_name), exist_ok=True)

def train(device='cuda', T=500, img_size=64, input_channels=1, channels=32, time_dim=256,
          batch_size=2, lr=1e-3, num_epochs=50, experiment_name="ddpm", show=False):
    """Implements algorithm 1 (Training) from the DDPM paper at page 4"""
    create_result_folders(experiment_name)

    # Define transformations (Resize, Normalize, etc.)
    import torchvision.transforms as transforms
    transform = transforms.Compose([
        #grayscale
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),  # Resize to the desired image size
        transforms.ToTensor(),                   # Convert image to tensor
        transforms.Normalize((0.5,), (0.5,))     # Normalize to range [-1, 1]
    ])

    # Load 10 images from the 'spectrograms' folder
    images = load_images_from_folder(SPECTROGRAM_FOLDER, transform, max_images=10).to(device)

    model = UNet(img_size=img_size, c_in=input_channels, c_out=input_channels, 
                 time_dim=time_dim, channels=channels, device=device).to(device)
    diffusion = Diffusion(img_size=img_size, T=T, beta_start=1e-4, beta_end=0.02, device=device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    mse = torch.nn.MSELoss()

    logger = SummaryWriter(os.path.join("runs", experiment_name))
    
    for epoch in range(1, num_epochs + 1):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm([images])  # Single batch of 10 images

        for i, batch_images in enumerate(pbar):
            batch_images = batch_images.to(device)

            # Implement the training loop
            t = diffusion.sample_timesteps(batch_images.shape[0]).to(device)  # Sample timesteps
            x_t, noise = diffusion.q_sample(batch_images, t)                  # Inject noise
            predicted_noise = model(x_t, t)                                   # Predict noise
            
            # Compute loss
            loss = mse(noise, predicted_noise)  

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * len(pbar) + i)

        # Save generated samples and model after each epoch
        num_generated_images = 1
        #num_generated_images = images.shape[0]
        
        sampled_images = diffusion.p_sample_loop(model, batch_size=num_generated_images)
        save_images(images=sampled_images, path=os.path.join("results", experiment_name, f"{epoch}.png"),
                    show=show, title=f'Epoch {epoch}')
        torch.save(model.state_dict(), os.path.join("models", experiment_name, f"weights-{epoch}.pt"))

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    print(f"Model will run on {device}")
    set_seed(seed=SEED)
    train(device=device)

if __name__ == '__main__':
    main()
    # Code to open Tensorboard to see runs
    # tensorboard --logdir=runs
