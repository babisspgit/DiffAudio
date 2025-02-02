#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpua100
### -- set the job Name --
#BSUB -J img2imgM_job
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 GPU in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm -- maximum 24 hours for GPU-queues right now
#BSUB -W 2:00
### -- request 5GB of system memory --
#BSUB -R "rusage[mem=10GB]"
### -- Specify the output and error file. %J is the job-id --
#BSUB -o gpu_%J.out
#BSUB -e gpu_%J.err



source /zhome/15/5/181507/thesisenv/bin/activate

python3 -u src/img2imgM.py 

nvidia-smi
