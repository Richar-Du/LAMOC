export EXP_NAME=$1
export DATASET=$4
OMP_NUM_THREADS=20 CUDA_VISIBLE_DEVICES=$2 python cot.py --caption_dir {$DATASET}_caption_dir/$EXP_NAME --output_dir {$DATASET}_cot_dir/$EXP_NAME --model $3