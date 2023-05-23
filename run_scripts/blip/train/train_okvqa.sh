OMP_NUM_THREADS=20 CUDA_VISIBLE_DEVICES=1,2,6,7 python -m torch.distributed.run --nproc_per_node=4 train.py --cfg-path lavis/projects/blip/train/okvqa_ft.yaml
