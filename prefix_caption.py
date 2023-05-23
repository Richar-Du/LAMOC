import os
import torch
from PIL import Image
import json
import os
import numpy as np
import tqdm
from lavis.models import load_model_and_preprocess
import argparse

def init_seed(seed):
    r""" init random seed for random functions in numpy, torch, cuda and cudnn

    Args:
        seed (int): random seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="prefix_caption")
    parser.add_argument("--pth_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()

    init_seed(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.dataset == 'aokvqa':
        vqa_file = open('cache/aokvqa/annotations/aokvqa_v1p0_val.json', 'r')
    elif args.dataset == 'okvqa':
        vqa_file = open('cache/okvqa/annotations/vqa_val_eval.json', 'r')
    # vqa_file = open('cache/coco/annotations/vqa_train_4000.json', 'r')
    print("loading data...")
    vqa = json.load(vqa_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("loading model...")
    model, vis_processors, txt_processors = load_model_and_preprocess("blip_caption", model_type="large_coco", is_eval=True, device=torch.device("cpu"))
    # model, vis_processors, txt_processors = load_model_and_preprocess("blip_vqa_caption", model_type="aokvqa", is_eval=True, device=torch.device("cpu"))
    if args.pth_path != 'origin':
        print(f"loading model from {args.pth_path}")
        checkpoint = torch.load(args.pth_path, map_location="cpu")
        if "model" in checkpoint.keys():
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
    model.text_decoder = model.text_decoder.to(device)
    model.visual_encoder = model.visual_encoder.to(device)
    caption_file = open(f'{args.output_dir}/{args.seed}.json', 'w')
    images = []
    captions = []
    batch_size = 32
    for data in tqdm.tqdm(vqa):
        img_path = os.path.join('/mnt/duyifan/data/coco', data['image'])
        raw_image = Image.open(img_path).convert("RGB")
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        images.append(image.to(torch.device('cpu')))
    images = torch.cat(images, dim=0)#.to(device)
    for i in tqdm.trange(0, len(vqa), batch_size):
        end_idx = min(len(vqa), i+batch_size)
        image = images[i:end_idx].to(device)
        samples = {"image": image}
        with torch.no_grad():
            caption = model.generate(samples, num_captions = 10, use_nucleus_sampling = False, top_p=0.9, top_k = 50,  num_beams=1)
        # print(captions)
        captions.extend(caption)
    for i in tqdm.trange(len(vqa)):
        caption = captions[i]
        data = vqa[i]
        caption_file.write(json.dumps({'image': data['image'], 'question': data['question'], 'caption': caption})+'\n')
        # caption_file.write(json.dumps({'image': data['image'], 'question': data['question'], 'choices': data['choices'],'correct_choice_idx':data['correct_choice_idx'], 'caption': captions})+'\n')
    caption_file.close()