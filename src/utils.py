import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Tuple
from PIL import Image
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from scipy.io import wavfile
import librosa
import librosa.display
from pydub import AudioSegment
import traceback
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pydub import AudioSegment
import traceback

def mp3_to_melspectrogram(mp3_file, output_folder, trim_seconds=30, include_legend=False, nmels=128, 
                          segment_duration=10, save_audio_segments=False):
    """
    Converts an MP3 file to mel spectrograms and optionally saves the audio segments.

    Parameters:
        mp3_file (str): Path to the MP3 file.
        output_folder (str): Folder to save the spectrogram images and audio segments.
        trim_seconds (int): Number of seconds to trim from both ends of the audio.
        include_legend (bool): Whether to include legends and axes in the spectrogram plots.
        nmels (int): Number of mel bands to generate in the spectrogram.
        segment_duration (int): Duration of each segment in seconds. If None, processes the entire audio.
        save_audio_segments (bool): Whether to save the corresponding audio segments.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    try:
        print(f"Loading file: {mp3_file}")
        
        # Load the MP3 file using PyDub
        audio = AudioSegment.from_file(mp3_file)
        total_duration_ms = len(audio)  # Duration in milliseconds
        total_duration = total_duration_ms / 1000.0  # Convert to seconds
        print(f"File {mp3_file}, Total Duration: {total_duration:.2f} seconds")

        # Check if the file is long enough to trim
        if total_duration <= 2 * trim_seconds:
            print(f"File {mp3_file} is too short to trim {trim_seconds} seconds from both ends. Skipping.")
            return

        # Trim the first and last trim_seconds
        start_trim = trim_seconds * 1000  # Convert to milliseconds
        end_trim = total_duration_ms - trim_seconds * 1000
        audio_trimmed = audio[start_trim:end_trim]

        print(f"Trimmed audio length (ms): {len(audio_trimmed)}")
        
        # Calculate segment length in samples if segment_duration is specified
        sr = audio_trimmed.frame_rate
        samples_per_segment = int(segment_duration * sr) if segment_duration else len(audio_trimmed.get_array_of_samples())
        
        # Convert to NumPy array
        y = np.array(audio_trimmed.get_array_of_samples()).astype(np.float32)
        
        # Convert stereo to mono
        if audio_trimmed.channels == 2:
            y = y.reshape((-1, 2)).mean(axis=1)

        print(f"Sample rate: {sr}, Audio shape after conversion: {y.shape}")

        # Process each segment if segment_duration is specified
        segment_count = 0
        for start_sample in range(0, len(y) - samples_per_segment + 1, samples_per_segment):
            segment = y[start_sample:start_sample + samples_per_segment]
            segment_count += 1

            # Generate the mel spectrogram
            mel_spectrogram = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=nmels)
            D = librosa.power_to_db(mel_spectrogram, ref=np.max)

            # Plot and save the mel spectrogram
            plt.figure(figsize=(10, 6))
            librosa.display.specshow(D, sr=sr, x_axis='time' if include_legend else None, 
                                     y_axis='mel' if include_legend else None, cmap='inferno')

            if include_legend:
                plt.colorbar(format='%+2.0f dB')  # Add color bar if include_legend is True
                plt.title(f'Mel Spectrogram of {os.path.basename(mp3_file)} - Segment {segment_count}')  # Add title
            else:
                plt.axis('off')  # Turn off axes if include_legend is False
                plt.gca().set_position([0, 0, 1, 1])  # Remove any padding/margins

            # Save the spectrogram image
            output_image_path = os.path.join(
                output_folder, f"{os.path.splitext(os.path.basename(mp3_file))[0]}_segment_{segment_count}.png"
            )
            plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0 if not include_legend else 0.1)
            plt.close()

            print(f"Spectrogram saved to {output_image_path}")

            # Save the corresponding audio segment if required
            if save_audio_segments:
                audio_segment = audio_trimmed[start_sample // sr * 1000:(start_sample + samples_per_segment) // sr * 1000]
                audio_segment_path = os.path.join(
                    output_folder, f"{os.path.splitext(os.path.basename(mp3_file))[0]}_segment_{segment_count}.mp3"
                )
                audio_segment.export(audio_segment_path, format="mp3")
                print(f"Audio segment saved to {audio_segment_path}")

        if segment_duration is None:
            # Process entire audio as a single segment if segment_duration is not specified
            mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=nmels)
            D = librosa.power_to_db(mel_spectrogram, ref=np.max)

            plt.figure(figsize=(10, 6))
            librosa.display.specshow(D, sr=sr, x_axis='time' if include_legend else None, 
                                     y_axis='mel' if include_legend else None, cmap='inferno')

            if include_legend:
                plt.colorbar(format='%+2.0f dB')
                plt.title(f'Mel Spectrogram of {os.path.basename(mp3_file)}')

            else:
                plt.axis('off')
                plt.gca().set_position([0, 0, 1, 1])

            output_image_path = os.path.join(
                output_folder, f"{os.path.splitext(os.path.basename(mp3_file))[0]}_full.png"
            )
            plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0 if not include_legend else 0.1)
            plt.close()

            print(f"Full audio spectrogram saved to {output_image_path}")

    except Exception as e:
        print(f"Error processing {mp3_file}: {e}")
        traceback.print_exc()  # Print detailed error traceback


def process_all_mp3_in_folder(folder_path, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all MP3 files in the folder
    for file in os.listdir(folder_path):
        if file.endswith(".mp3"):
            mp3_file = os.path.join(folder_path, file)
            output_image_path = os.path.join(output_folder, f"{os.path.splitext(file)[0]}_trimmed.png")
            
            # Check if the spectrogram image already exists
            if os.path.exists(output_image_path):
                print(f"Skipping {mp3_file}, spectrogram already exists.")
                continue

            print(f"Processing file: {mp3_file}")
            mp3_to_melspectrogram(mp3_file, output_folder)
            
       
       
            
# Images to Spectrogram from Riffusion          
            
            
def wav_bytes_from_spectrogram_image(image: Image.Image) -> Tuple[io.BytesIO, float]:
    """
    Reconstruct a WAV audio clip from a spectrogram image. Also returns the duration in seconds.
    """
    max_volume = 50
    power_for_image = 0.25
    Sxx = spectrogram_from_image(image, max_volume=max_volume, power_for_image=power_for_image)

    sample_rate = 44100  # [Hz]
    clip_duration_ms = 5000  # [ms]

    bins_per_image = 512
    n_mels = 512

    # FFT parameters
    window_duration_ms = 100  # [ms]
    padded_duration_ms = 400  # [ms]
    step_size_ms = 10  # [ms]

    # Derived parameters
    num_samples = int(image.width / float(bins_per_image) * clip_duration_ms) * sample_rate
    n_fft = int(padded_duration_ms / 1000.0 * sample_rate)
    hop_length = int(step_size_ms / 1000.0 * sample_rate)
    win_length = int(window_duration_ms / 1000.0 * sample_rate)

    samples = waveform_from_spectrogram(
        Sxx=Sxx,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        num_samples=num_samples,
        sample_rate=sample_rate,
        mel_scale=True,
        n_mels=n_mels,
        num_griffin_lim_iters=32,
    )

    wav_bytes = io.BytesIO()
    wavfile.write(wav_bytes, sample_rate, samples.astype(np.int16))
    wav_bytes.seek(0)

    duration_s = float(len(samples)) / sample_rate

    return wav_bytes, duration_s


def spectrogram_from_image(image: Image.Image, max_volume: float = 50, power_for_image: float = 0.25) -> np.ndarray:
    """
    Compute a spectrogram magnitude array from a spectrogram image.
    """
    # Convert to a numpy array of floats
    data = np.array(image).astype(np.float32)

    # Flip Y and take a single channel
    data = data[::-1, :, 0]

    # Invert
    data = 255 - data

    # Rescale to max volume
    data = data * max_volume / 255

    # Reverse the power curve
    data = np.power(data, 1 / power_for_image)

    return data


def waveform_from_spectrogram(
    Sxx: np.ndarray,
    n_fft: int,
    hop_length: int,
    win_length: int,
    num_samples: int,
    sample_rate: int,
    mel_scale: bool = True,
    n_mels: int = 512,
    num_griffin_lim_iters: int = 32,
    device: str = "cpu",
) -> np.ndarray:
    """
    Reconstruct a waveform from a spectrogram.
    """
    Sxx_torch = torch.from_numpy(Sxx).to(device)

    if mel_scale:
        mel_inv_scaler = torchaudio.transforms.InverseMelScale(
            n_mels=n_mels,
            sample_rate=sample_rate,
            f_min=0,
            f_max=10000,
            n_stft=n_fft // 2 + 1,
            norm=None,
            mel_scale="htk",
        ).to(device)

        Sxx_torch = mel_inv_scaler(Sxx_torch)

    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        power=1.0,
        n_iter=num_griffin_lim_iters,
    ).to(device)

    waveform = griffin_lim(Sxx_torch).cpu().numpy()

    return waveform


def process_spectrogram_folder(input_dir: str):
    """
    Convert all spectrogram images in a folder to audio files.
    
    Args:
        input_dir (str): Path to the folder containing spectrogram images.
    """
    for filename in os.listdir(input_dir):
        if filename.endswith(".png"):  # Only process PNG images
            image_path = os.path.join(input_dir, filename)
            audio_path = os.path.splitext(image_path)[0] + ".wav"  # Save with .wav extension
            
            print(f"Processing: {filename}")
            
            # Load the spectrogram image
            image = Image.open(image_path)
            
            # Convert the spectrogram image to audio
            wav_bytes, duration = wav_bytes_from_spectrogram_image(image)
            
            # Save the audio file
            with open(audio_path, "wb") as f:
                f.write(wav_bytes.getbuffer())
            
            print(f"Saved audio: {audio_path} (Duration: {duration:.2f}s)")


# Example usage
#input_dir = "generated_images"
#process_spectrogram_folder(input_dir)




def map_images_to_classes(image_folder, df, association_class, target_class):
    """
    Maps image files in a folder to their respective target classes using an association between
    an identifier (association class) and a target class.

    Parameters:
        image_folder (str): Path to the folder containing image files (e.g., "../data/raw/1000_mel_spec_seg").
        association_class (pd.Series): A Pandas Series representing the association class (e.g., "track_id").
        target_class (pd.Series): A Pandas Series representing the target class/label (e.g., "track_genre").

    Returns:
        pd.DataFrame: A DataFrame mapping image paths to their target classes.
    """
    # Ensure the association class and target class have the same length
    if len(association_class) != len(target_class):
        raise ValueError("The association_class and target_class must have the same length.")

    # List all image files in the folder
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]

    # Create a list to store rows for the resulting DataFrame
    image_data = []

    # Loop through the image files
    for image_file in image_files:
        # Extract the association ID from the filename
        association_id = "_".join(image_file.split("_")[:-2])  # Adjust based on filename format
        
        # Check if the association ID exists in the DataFrame
        if association_id in df["association_id"].values:
            # Get the corresponding target class for the association ID
            row = df[df["association_id"] == association_id].iloc[0]
            target_label = row["target_label"]
            
            # Add the image path and its target class to the list
            image_data.append([os.path.join(image_folder, image_file), target_label])
        else:
            print(f"Association ID {association_id} not found in the association mapping!")
            

    # Convert the list to a DataFrame and return it
    return pd.DataFrame(image_data, columns=["image_path", "class"])
          


#
#def generate_image_association_file(image_folder, dfg, output_columns=("image_path", "class")):
#    """
#    Generates an association file between image files in a folder and their respective classes/genres from a DataFrame.
#
#    Parameters:
#        image_folder (str): Path to the folder containing image files.
#        dfg (pd.DataFrame): DataFrame containing track IDs and their corresponding genres or classes.
#        output_columns (tuple): Column names for the output DataFrame. Defaults to ("image_path", "class").
#
#    Returns:
#        pd.DataFrame: DataFrame associating image paths with their respective classes/genres.
#    """
#    # List all image files in the folder
#    image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]
#
#    # Create a list to store rows for the new DataFrame
#    image_data = []
#
#    # Loop through image files
#    for image_file in image_files:
#        # Extract the track_id from the filename (everything before "_segment")
#        track_id = "_".join(image_file.split("_")[:-2])  # Adjust splitting based on naming
#        
#        # Look up the corresponding genre or class in the dataframe
#        if track_id in dfg['track_id'].values:  # Assuming 'track_id' column exists in dfg
#            row = dfg[dfg['track_id'] == track_id].iloc[0]
#            
#            # Extract the class or genre information
#            class_info = row['track_genre']  # Replace with one-hot columns if needed
#            
#            # Add a row to the new data: [image path, class/genre]
#            image_data.append([os.path.join(image_folder, image_file), class_info])
#        else:
#            print(f"Track ID {track_id} not found in the dataframe!")  # Debugging info
#
#    # Create and return a new DataFrame for the association file
#    return pd.DataFrame(image_data, columns=output_columns)



############################################################################################################
#  DATASET-DATALOADER
############################################################################################################


from collections import defaultdict
from sklearn.model_selection import train_test_split
import pandas as pd
import random
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

# Step 1: Group images by track ID and class
def group_tracks_by_class_and_id(association_csv):
    df = pd.read_csv(association_csv)
    df['track_id'] = df['image_path'].apply(lambda x: "_".join(os.path.basename(x).split("_")[:-2]))
    class_groups = defaultdict(list)
    
    # Group track IDs by their class
    for track_id, group in df.groupby('track_id'):
        track_class = group.iloc[0]['class']
        class_groups[track_class].append(track_id)
    
    return class_groups, df

# Step 2: Split track IDs for each class
def split_tracks_by_class(class_groups, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    train_ids, val_ids, test_ids = [], [], []
    
    for track_class, track_ids in class_groups.items():
        random.shuffle(track_ids)  # Shuffle track IDs within the class
        
        # Perform splits
        train, temp = train_test_split(track_ids, test_size=(1 - train_ratio))
        val, test = train_test_split(temp, test_size=(test_ratio / (test_ratio + val_ratio)))
        
        # Append to respective splits
        train_ids.extend(train)
        val_ids.extend(val)
        test_ids.extend(test)
    
    return train_ids, val_ids, test_ids

# Step 3: Create a custom PyTorch Dataset
class SpectrogramDataset(Dataset):
    def __init__(self, df, track_ids, transform=None):
        self.data = df[df['track_id'].isin(track_ids)]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row['image_path']
        label = row['class']
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        return image, label
    

# Step 4: Create DataLoaders
def create_balanced_dataloaders(image_folder, association_csv, batch_size=32):
    # Group by class and track ID
    class_groups, df = group_tracks_by_class_and_id(association_csv)
    
    # Perform class-balanced splits
    train_ids, val_ids, test_ids = split_tracks_by_class(class_groups)
    
    # Define image transformations (e.g., resizing, normalization)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to a consistent size
        transforms.ToTensor(),         # Convert to tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
    ])
    
    # Create datasets
    train_dataset = SpectrogramDataset(df, train_ids, transform=transform)
    val_dataset = SpectrogramDataset(df, val_ids, transform=transform)
    test_dataset = SpectrogramDataset(df, test_ids, transform=transform)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader