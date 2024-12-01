import os
import sys
import pandas as pd
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
import traceback
import gc 


#final
def mp3_to_melspectrogram(mp3_file, audio_output_folder, spec_output_folder, trim_seconds=30, 
                          include_legend=False, nmels=512, segment_duration=5, save_audio_segments=True):
    """
    Converts an MP3 file to mel spectrograms and optionally saves the audio segments.

    Parameters:
        mp3_file (str): Path to the MP3 file.
        audio_output_folder (str): Folder to save the audio segments.
        spec_output_folder (str): Folder to save the spectrogram images.
        trim_seconds (int): Number of seconds to trim from both ends of the audio.
        include_legend (bool): Whether to include legends and axes in the spectrogram plots.
        nmels (int): Number of mel bands to generate in the spectrogram.
        segment_duration (int): Duration of each segment in seconds. If None, processes the entire audio.
        save_audio_segments (bool): Whether to save the corresponding audio segments.
    """
    # Ensure both output folders exist
    os.makedirs(audio_output_folder, exist_ok=True)
    os.makedirs(spec_output_folder, exist_ok=True)

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
                spec_output_folder, f"{os.path.splitext(os.path.basename(mp3_file))[0]}_segment_{segment_count}.png"
            )
            plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0 if not include_legend else 0.1)
            plt.close()

            print(f"Spectrogram saved to {output_image_path}")

            # Save the corresponding audio segment if required
            if save_audio_segments:
                audio_segment = audio_trimmed[start_sample // sr * 1000:(start_sample + samples_per_segment) // sr * 1000]
                audio_segment_path = os.path.join(
                    audio_output_folder, f"{os.path.splitext(os.path.basename(mp3_file))[0]}_segment_{segment_count}.mp3"
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
                spec_output_folder, f"{os.path.splitext(os.path.basename(mp3_file))[0]}_full.png"
            )
            plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0 if not include_legend else 0.1)
            plt.close()

            print(f"Full audio spectrogram saved to {output_image_path}")

    # Explicitly release memory
        del audio, audio_trimmed, y, segment
        gc.collect()

    except Exception as e:
        print(f"Error processing {mp3_file}: {e}")
        traceback.print_exc()  # Print detailed error traceback


def process_all_mp3_in_folder(folder_path, audio_output_folder, spec_output_folder):
    # Ensure the output folders exist
    os.makedirs(audio_output_folder, exist_ok=True)
    os.makedirs(spec_output_folder, exist_ok=True)

    # Loop through all MP3 files in the folder
    for file in os.listdir(folder_path):
        if file.endswith(".mp3"):
            mp3_file = os.path.join(folder_path, file)

            # Check if the first spectrogram segment exists
            first_segment_path = os.path.join(
                spec_output_folder, f"{os.path.splitext(file)[0]}_segment_1.png"
            )
            if os.path.exists(first_segment_path):
                print(f"Skipping {mp3_file}, spectrogram already exists.")
                continue

            print(f"Processing file: {mp3_file}")
            mp3_to_melspectrogram(mp3_file, audio_output_folder, spec_output_folder)
            
            
process_all_mp3_in_folder("/work3/s222948/DiffAudio/data/raw/1000dataset", "/work3/s222948/DiffAudio/data/processed/1000dataset_5/audio", "/work3/s222948/DiffAudio/data/processed/1000dataset_5/audio_specs")