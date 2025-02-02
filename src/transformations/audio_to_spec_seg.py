import os
import gc
import glob
import numpy as np
import torch
import torchaudio
import librosa
import random
from PIL import Image

#####################################
# Riffusion Parameters (unchanged)
#####################################
image_width = 512
sample_rate = 44100  # [Hz]
clip_duration_ms = 5000  # [ms]

bins_per_image = 512
n_mels = 512

# FFT parameters
window_duration_ms = 100  # [ms]
padded_duration_ms = 400  # [ms]
step_size_ms = 10   # [ms]

# Derived parameters
num_samples = int(image_width / float(bins_per_image) * clip_duration_ms) * sample_rate
n_fft = int(padded_duration_ms / 1000.0 * sample_rate)
hop_length = int(step_size_ms / 1000.0 * sample_rate)
win_length = int(window_duration_ms / 1000.0 * sample_rate)

#####################################
# Existing Unchanged Functions
#####################################

def spectrogram_from_waveform(
    waveform: np.ndarray,
    sample_rate=sample_rate,
    n_fft=n_fft,
    hop_length=hop_length,
    win_length=win_length,
    mel_scale: bool = True,
    n_mels: int = 512,
) -> np.ndarray:
    """
    Convert a waveform (time-domain) signal into either a magnitude STFT
    or a magnitude Mel spectrogram (depending on mel_scale).
    """
    spectrogram_func = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        power=None,
        hop_length=hop_length,
        win_length=win_length,
    )

    waveform_tensor = torch.from_numpy(waveform.astype(np.float32)).reshape(1, -1)
    Sxx_complex = spectrogram_func(waveform_tensor).numpy()[0]
    Sxx_mag = np.abs(Sxx_complex)

    if mel_scale:
        mel_scaler = torchaudio.transforms.MelScale(
            n_mels=n_mels,
            sample_rate=sample_rate,
            f_min=0,
            f_max=10000,
            n_stft=n_fft // 2 + 1,
            norm=None,
            mel_scale="htk",
        )
        Sxx_mag = mel_scaler(torch.from_numpy(Sxx_mag)).numpy()

    return Sxx_mag


def image_from_spectrogram(spectrogram: np.ndarray, max_volume: float = 50, power_for_image: float = 0.25) -> Image.Image:
    """
    Convert a 2D spectrogram array into a PIL image, applying dynamic
    range compression, inversion, flipping, and conversion to RGB.
    """
    data = np.power(spectrogram, power_for_image)
    data = data * 255 / max_volume
    data = 255 - data
    image = Image.fromarray(data.astype(np.uint8))
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    image = image.convert("RGB")
    return image

#####################################
# Modified Function: Now with num_segments
#####################################

#### in series, so maybe better for 10seg
#def process_single_wav_file_5sec_segments(
#    input_wav_path: str,
#    out_audio_folder: str,
#    out_spectrogram_folder: str,
#    num_segments: int = 10  # <-- New parameter: how many 5-second segments to create
#):
#    """
#    1) Load .wav at 44100 Hz (global sample_rate).
#    2) Trim the first 30s and the last 30s.
#    3) Split into *5-second* segments, discarding leftover < 5s,
#       and stopping once we've created 'num_segments' segments.
#    4) Generate spectrogram (via spectrogram_from_waveform) 
#       and image (via image_from_spectrogram).
#    5) Save each segment's .wav in out_audio_folder,
#       and each spectrogram .png in out_spectrogram_folder.
#    """
#
#    # Ensure output directories exist
#    os.makedirs(out_audio_folder, exist_ok=True)
#    os.makedirs(out_spectrogram_folder, exist_ok=True)
#
#    # 1. Load the .wav at the global Riffusion sample_rate (44100)
#    audio_data, sr = librosa.load(input_wav_path, sr=sample_rate)
#
#    total_length_sec = len(audio_data) / float(sr)
#    print(f"\nProcessing file: {os.path.basename(input_wav_path)}")
#    print(f"Original audio length: {total_length_sec:.2f} seconds")
#
#    # 2. Trim 30s from start, 30s from end
#    trim_start_sec = 30.0
#    trim_end_sec = 30.0
#    start_sample = int(trim_start_sec * sr)
#    end_sample = len(audio_data) - int(trim_end_sec * sr)
#
#    if start_sample >= end_sample:
#        print("WARNING: Trimming exceeds audio length. Skipping this file.")
#        return
#
#    audio_data = audio_data[start_sample:end_sample]
#    trimmed_length_sec = len(audio_data) / float(sr)
#    print(f"Trimmed audio length: {trimmed_length_sec:.2f} seconds")
#
#    # 3. Build up to 'num_segments' chunks of exactly 5 seconds each
#    segment_length_sec = 5.0
#    segment_length_samples = int(segment_length_sec * sr)
#    segments = []
#
#    idx = 0
#    while len(segments) < num_segments and (idx + segment_length_samples <= len(audio_data)):
#        segment = audio_data[idx : idx + segment_length_samples]
#        # If for some reason the chunk is > 5s, trim it
#        if len(segment) > segment_length_samples:
#            segment = segment[:segment_length_samples]
#        segments.append(segment)
#        idx += segment_length_samples
#
#    print(f"Number of 5-second segments used: {len(segments)} (limit was {num_segments})")
#
#    # 4. For each segment: create spectrogram, save .wav, save .png
#    base_name = os.path.splitext(os.path.basename(input_wav_path))[0]
#    for i, segment in enumerate(segments, start=1):
#        seg_sec = len(segment) / float(sr)
#        print(f"  Segment {i}: {seg_sec:.2f} seconds")
#
#        # Create spectrogram with original function (unchanged)
#        spec = spectrogram_from_waveform(segment, sample_rate=sample_rate)
#
#        # Dynamically compute max_volume for image
#        current_max = np.max(spec)
#        if current_max < 1e-10:
#            current_max = 1.0
#        power_for_image = 0.25
#        max_volume = np.ceil(np.power(current_max, power_for_image))
#
#        # Create image with original function (unchanged)
#        img = image_from_spectrogram(spec, max_volume=max_volume, power_for_image=power_for_image)
#
#        # Save segment audio as .wav
#        audio_output_path = os.path.join(out_audio_folder, f"{base_name}_segment_{i}.wav")
#        segment_tensor = torch.from_numpy(segment).unsqueeze(0)  # shape (1, samples)
#        torchaudio.save(audio_output_path, segment_tensor, sr)
#
#        # Save spectrogram as .png
#        image_output_path = os.path.join(out_spectrogram_folder, f"{base_name}_segment_{i}.png")
#        img.save(image_output_path)
#
#        print(f"    -> Saved segment audio to      {audio_output_path}")
#        print(f"    -> Saved spectrogram image to  {image_output_path}")
#
#        # Optional: clear memory
#        gc.collect()
#        


#####################################
# Main Folder-Processing Function (unchanged)
#####################################

## random so better for less segments maybe
def process_single_wav_file_5sec_segments(
    input_wav_path: str,
    out_audio_folder: str,
    out_spectrogram_folder: str,
    num_segments: int = 10  # how many random 5-second segments you want at most
):
    """
    1) Load .wav at 44100 Hz (global sample_rate).
    2) Trim the first 30s and the last 30s.
    3) Enumerate all *non-overlapping* 5-second blocks (in order), 
       then randomly pick up to 'num_segments' of them (no overlap, guaranteed).
    4) Generate spectrogram (via spectrogram_from_waveform)
       and image (via image_from_spectrogram).
    5) Save each segment's .wav in out_audio_folder,
       and each spectrogram .png in out_spectrogram_folder.
    """

    # Ensure output directories exist
    os.makedirs(out_audio_folder, exist_ok=True)
    os.makedirs(out_spectrogram_folder, exist_ok=True)

    # 1. Load .wav at global sample_rate = 44100
    audio_data, sr = librosa.load(input_wav_path, sr=sample_rate)

    total_length_sec = len(audio_data) / float(sr)
    print(f"\nProcessing file: {os.path.basename(input_wav_path)}")
    print(f"Original audio length: {total_length_sec:.2f} seconds")

    # 2. Trim 30s from start, 30s from end
    trim_start_sec = 30.0
    trim_end_sec = 30.0
    start_sample = int(trim_start_sec * sr)
    end_sample = len(audio_data) - int(trim_end_sec * sr)

    if start_sample >= end_sample:
        print("WARNING: Trimming exceeds audio length. Skipping this file.")
        return

    audio_data = audio_data[start_sample:end_sample]
    trimmed_length_sec = len(audio_data) / float(sr)
    print(f"Trimmed audio length: {trimmed_length_sec:.2f} seconds")

    # 3a. Compute how many 5-second blocks are possible
    segment_length_sec = 5.0
    segment_length_samples = int(segment_length_sec * sr)
    total_samples = len(audio_data)

    # If there's not even 5 seconds, we can't form a single segment
    if total_samples < segment_length_samples:
        print("Not enough audio left for even one 5-second segment.")
        return

    # 3b. Build the list of all possible 5-second blocks in series
    #     e.g. block 1: [0 : 5s], block 2: [5s : 10s], ...
    all_blocks = []
    idx = 0
    block_count = 0
    while idx + segment_length_samples <= total_samples:
        block_count += 1
        all_blocks.append((idx, idx + segment_length_samples))
        idx += segment_length_samples

    # 3c. Randomly choose up to num_segments blocks from these
    # If the user wants more segments than exist, they'll just get them all
    chosen_blocks_count = min(num_segments, len(all_blocks))
    chosen_blocks = random.sample(all_blocks, chosen_blocks_count)

    print(f"Total possible 5-second blocks: {len(all_blocks)}")
    print(f"Number of blocks randomly chosen: {len(chosen_blocks)} (limit was {num_segments})")

    # Optionally, if you want to process them in chronological order:
    # chosen_blocks.sort(key=lambda x: x[0])

    # 4. Process each chosen block
    base_name = os.path.splitext(os.path.basename(input_wav_path))[0]

    for i, (seg_start, seg_end) in enumerate(chosen_blocks, start=1):
        seg_samples = audio_data[seg_start:seg_end]
        seg_sec = len(seg_samples) / float(sr)
        print(f"  Segment {i}: Start={seg_start}, End={seg_end}, Duration={seg_sec:.2f} seconds")

        # Generate spectrogram with original function (unchanged)
        spec = spectrogram_from_waveform(seg_samples, sample_rate=sr)

        # Dynamically compute max_volume for image
        current_max = np.max(spec)
        if current_max < 1e-10:
            current_max = 1.0
        power_for_image = 0.25
        max_volume = np.ceil(np.power(current_max, power_for_image))

        # Create image with original function (unchanged)
        img = image_from_spectrogram(spec, max_volume=max_volume, power_for_image=power_for_image)

        # Save segment audio as .wav
        audio_output_path = os.path.join(out_audio_folder, f"{base_name}_segment_{i}.wav")
        seg_tensor = torch.from_numpy(seg_samples).unsqueeze(0)  # shape (1, samples)
        torchaudio.save(audio_output_path, seg_tensor, sr)

        # Save spectrogram as .png
        image_output_path = os.path.join(out_spectrogram_folder, f"{base_name}_segment_{i}.png")
        img.save(image_output_path)

        print(f"    -> Saved segment audio to      {audio_output_path}")
        print(f"    -> Saved spectrogram image to  {image_output_path}")

        # Optional: clear memory
        gc.collect() 
#        


def process_all_wav_in_folder(
    input_folder: str,
    output_audio_folder: str,
    output_spectrogram_folder: str
):
    """
    Processes all .wav files in 'input_folder' (non-recursive).
    For each file, outputs 5-second segments (up to a limit set
    in process_single_wav_file_5sec_segments) as .wav in 'output_audio_folder',
    and .png in 'output_spectrogram_folder'.
    """
    wav_files = glob.glob(os.path.join(input_folder, "*.wav"))
    if not wav_files:
        print(f"No .wav files found in {input_folder}")
        return

    print(f"Found {len(wav_files)} wav file(s) in {input_folder}\n")

    os.makedirs(output_audio_folder, exist_ok=True)
    os.makedirs(output_spectrogram_folder, exist_ok=True)

    for wav_path in wav_files:
        process_single_wav_file_5sec_segments(
            input_wav_path=wav_path,
            out_audio_folder=output_audio_folder,
            out_spectrogram_folder=output_spectrogram_folder,
            # Optionally, pass num_segments=... if you want a custom number 
            # (default is 10 in the function definition).
        )

    print("\nAll .wav files have been processed!")
    

process_all_wav_in_folder("/work3/s222948/data/raw/1000dataset/wav_files", "/work3/s222948/data/raw/1000dataset_10seg/audio", "/work3/s222948/data/raw/1000dataset_10seg/specs")
#process_all_wav_in_folder("C:/Users/spbsp/OneDrive - Danmarks Tekniske Universitet/Skrivebord/DiffAudio/data/raw/test", 
#                          "C:/Users/spbsp/OneDrive - Danmarks Tekniske Universitet/Skrivebord/DiffAudio/data/raw/testaa",
#                          "C:/Users/spbsp/OneDrive - Danmarks Tekniske Universitet/Skrivebord/DiffAudio/data/raw/tests")