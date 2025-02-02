import os
from pathlib import Path
from subprocess import run, CalledProcessError
from tqdm import tqdm  # For progress bar

# ===== CONFIG =====
input_folder = "/work3/s222948/data/raw/1000dataset"  # Path to folder with MP3 files
output_folder = os.path.join(input_folder, "wav_files")  # Output subfolder
# ==================

# Create output folder if it doesn't exist
Path(output_folder).mkdir(parents=True, exist_ok=True)

# Get all MP3 files in input folder
mp3_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.mp3')]

print(f"Found {len(mp3_files)} MP3 files. Converting to WAV...")

# Convert each file using FFmpeg
for filename in tqdm(mp3_files):
    try:
        # Build paths
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, f"{Path(filename).stem}.wav")
        
        # FFmpeg command for conversion
        command = [
            "ffmpeg", 
            "-y",               # Overwrite output without asking
            "-i", input_path,   # Input file
            "-ar", "44100",     # Set sample rate (optional, 44.1kHz is standard)
            "-ac", "2",         # Set number of audio channels (optional, 2 for stereo)
            output_path         # Output file
        ]
        
        # Run FFmpeg
        run(command, check=True)

    except CalledProcessError as e:
        print(f"\nFailed to convert {filename}: {str(e)}")
    except Exception as e:
        print(f"\nUnexpected error for {filename}: {str(e)}")

print(f"\nConversion complete! WAV files saved to: {output_folder}")