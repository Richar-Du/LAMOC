OMP_NUM_THREADS=20 CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.run --nproc_per_node=2 train.py --cfg-path lavis/projects/blip/train/aokvqa_ft.yaml
