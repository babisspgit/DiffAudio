import os
import io
from typing import Tuple

import numpy as np
from PIL import Image
from scipy.io import wavfile
import torch
import torchaudio


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
input_dir = "generated_images"
process_spectrogram_folder(input_dir)
