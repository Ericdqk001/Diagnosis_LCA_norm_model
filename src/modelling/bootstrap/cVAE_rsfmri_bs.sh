#$ -l tmem=11G
#$ -l gpu=true
#$ -l h_rt=100:00:00 
#$ -j y
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
cd /home/ericdeng/lca_psydx/
#gpu=2,3

# running command
################################################

python cVAE_bootstrap.py --data_path "/home/ericdeng/lca_psydx/processed_data" --feature_type "cortical_surface_area" --project_title "cVAE_cortical_surface_area_UCL_bootstrap" --batch_size 256 --learning_rate 0.0005 --latent_dim 10 --hidden_dim "30-30" --bootstrap_num 1000