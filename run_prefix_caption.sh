export EXP_NAME=$1
export DATASET=$4

for seed in 0 1 2 3 4
do
    OMP_NUM_THREADS=20 CUDA_VISIBLE_DEVICES=$2 python prefix_caption.py --pth_path $3 --output_dir {$DATASET}_caption_dir/$EXP_NAME --seed $seed --dataset $DATASET
done
