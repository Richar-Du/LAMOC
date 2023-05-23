OMP_NUM_THREADS=20 CUDA_VISIBLE_DEVICES=6 python -m torch.distributed.run --nproc_per_node=1 evaluate.py --cfg-path lavis/projects/blip/eval/okvqa_eval.yaml
