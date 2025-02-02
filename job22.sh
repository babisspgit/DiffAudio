#!/bin/bash

#BSUB -J Mp3toWav            # Job name
#BSUB -q hpc                    # Queue name
#BSUB -n 4                      # Number of cores
#BSUB -W 24:00                  # Wall-clock time (hh:mm)
#BSUB -R "rusage[mem=4GB]"      # Memory requirements per core
#BSUB -o Audio_Output_%J.out    # Standard output file
#BSUB -e Audio_Output_%J.err    # Standard error file
#BSUB -u s222948@dtu.dk         # Email for notifications
#BSUB -B                        # Notify when the job begins
#BSUB -N                        # Notify when the job ends     



source /zhome/15/5/181507/thesisenv/bin/activate

python3 -u src/to_wav.py 

nvidia-smi
