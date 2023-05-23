import requests
import json
import tqdm
import time
# API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-xxl"
# headers = {"Authorization": "Bearer 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"}

# def query(payload):
# 	response = requests.post(API_URL, headers=headers, json=payload)
# 	return response.json()
	
# vqa_file = open('aokvqa_caption.json','r')
# # cot_output_file = open('aokvqa_cot.json', 'w')
# for line in tqdm.tqdm(vqa_file.readlines()[:30]):
#     vqa = json.loads(line)
#     caption = vqa['caption'][0]
#     question = vqa['question']
#     output = query({
#         "inputs": f"Please Answer the following queston.\n{caption}. {question}",
#         "wait_for_model": True,
#     })
#     # import ipdb
#     # ipdb.set_trace()
#     time.sleep(1)
#     prediction = output
#     # cot_output_file.write(json.dumps({'prediction': prediction, 'answer':vqa['choices'][vqa['correct_choice_idx']]})+'\n')
#     print(prediction)
    
# vqa_file.close()
# # cot_output_file.close()
# print(a)


import argparse
import json
import tqdm
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, OPTForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
import os
def init_seed(seed):
    r""" init random seed for random functions in numpy, torch, cuda and cudnn

    Args:
        seed (int): random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="prefix_caption")
    parser.add_argument("--caption_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--model", type=str)
    args = parser.parse_args()

    init_seed(42)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print("loading model...")
    # model = OPTForCausalLM.from_pretrained("/home/duyifan/.cache/huggingface/hub/models--facebook--opt-30b/snapshots/463007d7da4e87fe962909a027811a8c0b32ede8")
    # tokenizer = GPT2Tokenizer.from_pretrained("/home/duyifan/.cache/huggingface/hub/models--facebook--opt-30b/snapshots/463007d7da4e87fe962909a027811a8c0b32ede8")
    if args.model == 'xxl':
        device = torch.device('cuda')
        model = AutoModelForSeq2SeqLM.from_pretrained("/mnt/duyifan/ckpt/google/flan-t5-xxl/").to(device)
        tokenizer = AutoTokenizer.from_pretrained("/mnt/duyifan/ckpt/google/flan-t5-xxl/")
    elif args.model == 'xl':
        device = torch.device('cuda')
        model = AutoModelForSeq2SeqLM.from_pretrained('/mnt/duyifan/ckpt/google/flan-t5-xl').to(device)
        tokenizer = AutoTokenizer.from_pretrained('/mnt/duyifan/ckpt/google/flan-t5-xl')
    elif args.model == 'large':
        device = torch.device('cuda')
        model = AutoModelForSeq2SeqLM.from_pretrained('/mnt/duyifan/ckpt/google/flan-t5-large').to(device)
        tokenizer = AutoTokenizer.from_pretrained('/mnt/duyifan/ckpt/google/flan-t5-large')
    elif args.model == 'base':
        device = torch.device('cuda')
        model = AutoModelForSeq2SeqLM.from_pretrained('/mnt/duyifan/ckpt/google/flan-t5-base').to(device)
        tokenizer = AutoTokenizer.from_pretrained('/mnt/duyifan/ckpt/google/flan-t5-base')
    # examplar = "Q: What does the bright red light on the vehicle mean?\nChoices: (a) stop; (b) run; (c) accident; (d) normal\nA: Let's imagine flashing red lights on a vehicle in traffic. When a vehicle turns on a bright light, it means the vehicle stops. So the answer is (a).\nQ: What is the dog going to do next?\nChoices: (a) jumping; (b) sitting; (c) eating; (d) sleeping\nA: Let's imagine a dog standing on the floor is looking at the oven. The imagination is not enough, please imagine what is the posture of the dog first. So the answer is unsure.\nQ: What is the object in the persons hand called?\nChoices: (a) cloth; (b) helmet; (c) shoes; (d) pole\nA: Let's imagine an image of woman skiing on mountain slope. People hand poles to slide when skiing. So the answer is (d).\nQ: What artist is famous to painting images similar to the one over the fireplace?\nChoices: (a) van gogh; (b) da vinci; (c) monet; (d) tomas\nA: Let's imagine a couch, piano and a lamp in a room. The imagination is not enough, please imagine what is on the picture on the stove first. So the answer is unsure.\nQ: What is on the sofa?\nChoices: (a) flower; (b) cat; (c) remote; (d) dog\nA: Let's imagine a white cat sitting on a black sofa looking out of the window. A cat is on the sofa. So the answer is (b).\nQ: What is the equipment the man is holding used for?\nChoices: (a) cutting; (b) sweeping floor; (c) cooking meal; (d) cleaning\nA: Let's imagine a man holding a pan in the kichen. A pan is used to cook. So the answer is (c).\n"
    examplar = "Q: What artist is famous to painting images similar to the one over the fireplace?\nChoices: (a) van gogh; (b) da vinci; (c) monet; (d) tomas\nA: Let's imagine a couch, piano and a lamp in a room. The imagination is not enough, please imagine what is on the picture on the stove first. So the answer is unsure.\n"
    # Q: Which animal usually use the object the cat is playing with?\nChoices: (a) bird; (b) human; (c) tiger; (d) sheep\nA: Let's imagine a cat is bitting a hat. People sometimes wear a hat on their head. So the answer is (b).\n
    open_examplar = "Q: What does the bright red light on the vehicle mean?\nA: lashing red lights on a vehicle in traffic. When a vehicle turns on a bright light, it means the vehicle stops. So the answer: stop\nQ: What is the object in the persons hand called?\nA: an image of woman skiing on mountain slope. People hand poles to slide when skiing. So the answer: pole\nQ: What is on the sofa?\nA: a white cat sitting on a black sofa looking out of the window. A cat is on the sofa. So the answer: cat\nQ: What is the equipment the man is holding used for?\nA: a man holding a pan in the kichen. A pan is used to cook. So the answer: cooking"
    force_words_ids = tokenizer(['A', 'B', 'C', 'D'], add_special_tokens=False, return_tensors="pt").input_ids
    vocab_ids = list(range(tokenizer.vocab_size))
    for ele in reversed(force_words_ids):
        del vocab_ids[ele[0]]
    bad_words_ids = [[ele] for ele in vocab_ids]
    # vqa_file = open('aokvqa_caption.json','r')
    caption_file_list = os.listdir(args.caption_dir)
    for caption_file_name in caption_file_list:
        caption_file = open(f"{args.caption_dir}/{caption_file_name}", 'r') # json.load(open('aokvqa_multi_caption_llm_rl.json', 'r'))
        cot_output_file = open(f'{args.output_dir}/{caption_file_name}', 'w')
        # lines = vqa_file.readlines()
        caption_lines = caption_file.readlines()
        for i, line in enumerate(tqdm.tqdm(caption_lines)):
            vqa = json.loads(caption_lines[i])
            question = vqa['question']
            # choices = f"(a) {vqa['choices'][0]}; (b) {vqa['choices'][1]}; (c) {vqa['choices'][2]}; (d) {vqa['choices'][3]}"
            # captions = vqa['caption']#[0]
            captions = json.loads(caption_lines[i])['caption']
            # captions = [caption_file[i]['caption']]

            cot = []
            nlls = []
            prompts = []
            # verify_prompts = []
            for caption in captions:
                # prompt = f"{examplar}Q: {question}\nChoices: {choices}\nA: Let's imagine {caption}."      # few shot
                # prompt = f"C:{caption}\nQ:{question}\nA:"       # few shot
                # prompt = f"{open_examplar}\nQ: {question}\nA: {caption}"        # few shot
                # prompt = f"Q:{question}\nOptions: {choices}\nA: Let's think step by step. {caption}"      # zero shot
                # prompt = f"Q:{question}\nOptions: {choices}\nA: {caption}"      # zero shot nocot
                # prompt = f"Answer the following question in one word.\nQ:{question}\nA: Let's think step by step. {caption}"  # zero shot open
                # prompt = f"Answer the following question in one word.\nQ: {caption} {question}\nA:"
                # prompt = f"Answer the following question in one word.\nQ: {caption} {question}"
                prompt = f"Please answer the following question.\n {caption}. {question}"
                # verify_prompt = f"Question: {question} Caption: {caption}\nTo what degree does the caption relate to the question:\nA: 0%\nB: 25%\nC: 50%\nD:75%"
                # prompt = f"{caption}. {question} Let's think step by step."
                # prompt = f"{caption}. {question}"
                # prompt = question
                prompts.append(prompt)
                # verify_prompts.append(verify_prompt)
                
            inputs = tokenizer(prompts, return_tensors="pt", padding='longest')
            # verify_inputs = tokenizer(verify_prompts, return_tensors='pt', padding='longest')
            with torch.no_grad():
                output = model.generate(inputs = inputs.input_ids.to(device), attention_mask = inputs.attention_mask.to(device), max_new_tokens=50, output_scores = True, return_dict_in_generate = True)
                # verify_output = model.generate(inputs = verify_inputs.input_ids.to(device), attention_mask = verify_inputs.attention_mask.to(device), max_new_tokens=1, output_scores = True, return_dict_in_generate = True, bad_words_ids=bad_words_ids)
            generate_ids = output['sequences']
            # verify_ids = verify_output['sequences']
            total_prob = torch.ones(generate_ids.size()[0]).unsqueeze(-1).to(generate_ids.device)
            for i in range(len(output['scores'])):
                prob = nn.functional.softmax(output['scores'][i], dim=-1)
                index = generate_ids[:, i+1].reshape(generate_ids.size()[0], 1)
                index_prob = torch.gather(prob, 1, index)
                zero_indicator = torch.zeros(index_prob.size()[0]).unsqueeze(-1).to(index_prob.device)
                zero_indicator[index_prob<1e-3] = 1
                index_prob = index_prob + zero_indicator
                total_prob = total_prob * index_prob
            nll = -torch.log(total_prob)/(len(output['scores']))
            # nlls.append(nll.item())
            answer = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)#[0]
            # verify = tokenizer.batch_decode(verify_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            # print(answer, round(nll.item(),4))
            # print(answer)
            # cot.append(answer)
            # cot_output_file.write(json.dumps({"image":vqa['image'], 'question':vqa['question'], 'choices':vqa['choices'], 'correct_choice_idx':vqa['correct_choice_idx'], 'caption':captions, 'prompt':prompt, 'cot': cot, 'nll': nlls})+'\n')
            cot_output_file.write(json.dumps({"image":vqa['image'], 'question':vqa['question'], 'caption':captions, 'prompt':prompt, 'cot': answer, 'nll': nll.tolist()})+'\n')
            # print('='*50)
            # cot_output_file.write('='*50+'\n')