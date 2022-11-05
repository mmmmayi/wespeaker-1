#!/bin/bash
#SBATCH -o resnet.out
#SBATCH --gres=gpu:1
#SBATCH -w ttnusa11
#SBATCH -p new
./run.sh

