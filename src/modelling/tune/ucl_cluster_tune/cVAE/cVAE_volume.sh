#This is for the Pretrain with the SH file

#$ -l tmem=11G
#$ -l gpu=true
#$ -l h_rt=100:00:00 
#$ -j y
#$ -N FA IMG
#$ -S /bin/bash

# activate the virtual env

source /home/ericdeng/myenv/bin/activate

module load python/3.8.5

# Source file for CUDA11.0
# 24/02/23

source /share/apps/source_files/cuda/cuda-11.0.source

nvidia-smi

hostname
date

# enter the project path
cd /home/ericdeng/lca_norm/
#gpu=2,3

# running command
################################################

python cVAE_tune.py --data_path "/home/ericdeng/lca_psydx/processed_data" --feature_type "cortical_volume" --project_title "cVAE_cortical_volume_UCL_hyper_tune"