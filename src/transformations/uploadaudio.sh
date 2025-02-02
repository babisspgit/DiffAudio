#!/bin/bash

#BSUB -J UploadAudio            # Job name
#BSUB -q hpc                    # Queue name
#BSUB -n 4                      # Number of cores
#BSUB -W 24:00                  # Wall-clock time (hh:mm)
#BSUB -R "rusage[mem=4GB]"      # Memory requirements per core
#BSUB -o Audio_Output_%J.out    # Standard output file
#BSUB -e Audio_Output_%J.err    # Standard error file
#BSUB -u s222948@dtu.dk         # Email for notifications
#BSUB -B                        # Notify when the job begins
#BSUB -N                        # Notify when the job ends     


#module load pandas/2.1.3-python-3.11.7
source /zhome/15/5/181507/thesisenv/bin/activate

# Variables
LOCAL_FOLDER="C:/User/sspbsp/OneDrive - Danmarks Tekniske Universitet/Skrivebord/DiffAudio/data/raw/audio"  # Replace with the path to your local folder
HPC_USER="s222948"          # Replace with your HPC username
HPC_HOST="login1.hpc.dtu.dk"          # Replace with your HPC host name or IP
HPC_REMOTE_FOLDER="/work3/s222948/data/raw/audio" # Replace with the path to the destination folder on the HPC

# Create the remote folder if it does not exist
mkdir -p $HPC_REMOTE_FOLDER

#scp -r "C:/User/sspbsp/OneDrive - Danmarks Tekniske Universitet/Skrivebord/DiffAudio/data/raw/audio" s222948@login1.hpc.dtu.dk:"/work3/s222948/data/raw/audio"
scp -r $LOCAL_FOLDER s222948@login1.hpc.dtu.dk:$HPC_REMOTE_FOLDER

# Use rsync to upload the entire folder
#rsync -avz --progress "$LOCAL_FOLDER" "$HPC_USER@$HPC_HOST:$HPC_REMOTE_FOLDER"

# Check if the transfer was successful
if [ $? -eq 0 ]; then
    echo "Folder successfully uploaded to $HPC_HOST:$HPC_REMOTE_FOLDER"
else
    echo "Error: Folder upload failed. Check your connection and paths."
fi
