OMP_NUM_THREADS=20 CUDA_VISIBLE_DEVICES=6 python -m torch.distributed.run --nproc_per_node=1 train.py --cfg-path lavis/projects/blip/train/vqav2_ft.yaml
