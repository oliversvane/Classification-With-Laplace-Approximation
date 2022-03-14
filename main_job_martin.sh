#!/bin/sh
#BSUB -q gpua100
#BSUB -J Laplace_A100
#BSUB -n 1
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -R "rusage[mem=40GB]"
#BSUB -u Martin@illum.info
#BSUB -B
#BSUB -N
#BSUB -o log.out_martin
#BSUB -e log.err_martin


# Load the cuda module
module load python3/3.8.11
module load cuda/11.3
module load cudnn/v8.2.0.53-prod-cuda-11.3

python3 main.py
