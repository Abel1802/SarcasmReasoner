#!/bin/bash

#SBATCH --job-name=mcsd_genrm_sft
#SBATCH --partition=gpu_h100

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1

#SBATCH --mem=406G
#SBATCH --time=24:00:00

#SBATCH --output=logs/mcsd_genrm_sft_%j.out
#SBATCH --error=logs/mcsd_genrm_sft_%j.err

#SBATCH --no-requeue

set -e
set -o pipefail

module load 2025
module load CUDA/12.8.0
source activate ms2

bash configs/mcsd/genrm/qwen25_omni_3b_1gpu.sh
