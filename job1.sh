#!/bin/bash
#BSUB -q hpc

#BSUB -J TransformData

#BSUB -n 1 
#### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"

#BSUB -R "rusage[mem=8GB]"

#### -- Notify me by email when execution begins --

#BSUB -B

#### -- Notify me by email when execution ends   --

#BSUB -N

#### -- email address --

#BSUB -u s222948@dtu.dk 

#BSUB -o job.out  # sou dimiourgei arxeio me to output

#BSUB -e job.err  # sou dimiourgei arxeio me ta errors


#BSUB -W 24:00

#### -- Output File --

#BSUB -o Output_%J.out

#### -- Error File --

#BSUB -e Output_%J.err

#### -- estimated wall clock time (execution time): hh:mm -- 

##BSUB -W 24:00 

### -- Specify the distribution of the cores: on a single node --
### -- end of LSF options -- 




# loads automatically also numpy and python3 and underlying dependencies for our python 3.11.7
#module load pandas/2.1.3-python-3.11.7

# in case you have created a virtual environment,
# activate it first:
source /zhome/15/5/181507/thesisenv/bin/activate

# 1) Use this for LSF to collect the stdout & stderr
#python3 helloworld.py

# 2) Use this for unbuffered output, so that you can check in real-time
# (with tail -f Output_.out Output_.err)
# what your program was printing "on the screen"
python3 -u src/transform_data_hpc.py 

# 3) Use this for just piping everything into a file, 
# the program knows then, that it's outputting to a file
# and not to a screen, and also combine stdout&stderr
#python3 helloworld.py > joboutput_$LSB_JOBID.out 2>&1