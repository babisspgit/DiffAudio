#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpua100
### -- set the job Name --
#BSUB -J Textual_Inversion(sadbuttrue)
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 GPU in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm -- maximum 24 hours for GPU-queues right now
#BSUB -W 6:00
### -- request 6GB of system memory --
#BSUB -R "rusage[mem=6GB]"
#BSUB -u s222948@dtu.dk         # Email for notifications
#BSUB -B                        # Notify when the job begins
#BSUB -N                        # Notify when the job ends    
#BSUB -o TxtInv_Output_%J.out    # Standard output file
#BSUB -e TxtInv_Output_%J.err    # Standard error file



source /zhome/15/5/181507/thesisenv/bin/activate
cd /zhome/15/5/181507/git/DiffAudio/diffusers/examples/textual_inversion

accelerate launch textual_inversion.py \
  --pretrained_model_name_or_path="riffusion/riffusion-model-v1" \
  --train_data_dir="/work3/s222948/data/processed/Textual_Inversion/Training_Files/Metallica/sadbuttrue" \
  --learnable_property="style" \
  --placeholder_token="<Metallica-song>" \
  --initializer_token="music" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=3000 \
  --learning_rate=5.0e-04 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --no_safe_serialization \
  --output_dir="/work3/s222948/models/Textual_Inversion/metallica_music_sadbuttrue" 
  


nvidia-smi