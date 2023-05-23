export EXP_NAME=$1
export GPU_ID=$2
export MODEL_PATH=$3
export MODEL_SIZE=$4
export GT_PATH=$5
export DATASET=$6
export MODE=$7
echo 'captioning...'
bash run_prefix_caption.sh $EXP_NAME $GPU_ID $MODEL_PATH $DATASET
echo 'answering...'
bash run_cot.sh $EXP_NAME $GPU_ID $MODEL_SIZE $DATASET
echo 'evaluating...'
bash run_consistency.sh {$DATASET}_cot_dir/$EXP_NAME $GT_PATH $MODE