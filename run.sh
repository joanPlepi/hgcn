#!/bin/bash
#
#SBATCH --job-name=bert_classification
#SBATCH --output=/ukp-storage-1/plepi/users_sarcasm/outputs/train_gat_bert.txt
#SBATCH --mail-user=joanplepi@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --partition=cai
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1

source /ukp-storage-1/plepi/anaconda3/etc/profile.d/conda.sh
export HGCN_HOME=$(pwd)
export LOG_DIR="$HGCN_HOME/logs"
export PYTHONPATH="$HGCN_HOME:$PYTHONPATH"
export DATAPATH="$HGCN_HOME/data"
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64
source activate hgcn  # replace with source hgcn/bin/activate if you used a virtualenv
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/ukp-storage-1/plepi/anaconda3/lib/

python train.py --task nc --dataset twitter --model HGCN --dim 50 --lr 0.01 --dim 50 --num-layers 2 --act relu --bias 1 --dropout 0 --weight-decay 0 --manifold PoincareBall --log-freq 5 --cuda 0


