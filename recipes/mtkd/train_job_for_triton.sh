#!/bin/bash -l

#SBATCH -J train-mtkd-fi
#SBATCH -p gpu-v100-16g
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --export=HOME,USER,TERM,WRKDIR
#SBATCH -o train_%j.out

module load mamba
module load cuda
source activate tea

cd $WRKDIR/GitHub/summer-internship-2026

tea train-mtkd \
    mtkd.linguality=Multilingual \
    mtkd.language=FI \
    mtkd.session=6 \
    mtkd.epochs=20 \
    mtkd.lr=2e-5 \
    mtkd.batch_size=16 \
    mtkd.noise.use=false \
    mtkd.noise.contam_prob=0.5 \
    mtkd.noise.snr_min=15 \
    mtkd.noise.snr_max=30 