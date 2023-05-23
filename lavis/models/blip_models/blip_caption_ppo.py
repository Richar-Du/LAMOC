"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from lavis.common.registry import registry

from lavis.models.blip_models.blip_caption import BlipCaption
from lavis.models.blip_models.blip_outputs import (
    BlipOutput,
    BlipIntermediateOutput,
)

from lavis.models.med import XBertEncoder, XBertLMHeadDecoderPolicy
from lavis.models.vit import VisionTransformerEncoder

class ValueHead(nn.Module):
    """The ValueHead class implements a head for GPT2 that returns a scalar for each output token."""
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super().__init__()
        self.detach_head = False
        self.summary = nn.Linear(input_dim, output_dim)
        self.first_dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, cls_index=None):
        output = hidden_states
        output = self.first_dropout(output)
        output = self.summary(output)
        return output


@registry.register_model("blip_caption_ppo")
class BlipCaptionPPO(BlipCaption):
    
    PRETRAINED_MODEL_CONFIG_DICT = {
        # "base_coco": "configs/models/blip_caption_policy.yaml",
        "large_coco": "configs/models/blip_caption_ppo.yaml",
    }

    def __init__(self, image_encoder, text_decoder, prompt=None, max_txt_len=40, feedback='nll', text_encoder = None, has_encoder = False, cfg = None):
        super().__init__(image_encoder, text_decoder, prompt, max_txt_len)
        self.feedback = feedback

        self.has_encoder = has_encoder
        self.text_encoder = text_encoder
        config_file = 'lavis/' + cfg.get('med_config_path', None)
        config = open(config_file, 'r')
        model_config = json.load(config)
        self.v_head = ValueHead(model_config['hidden_size'], 1)

    def forward_encoder(self, samples):
        if self.has_encoder:
            questions = samples["text_input"]
            questions = self.tokenizer(
                questions,
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(self.device)
            questions.input_ids[:, 0] = self.tokenizer.enc_token_id
            samples.update({"tokenized_text": questions})

            image_embeds = self.visual_encoder.forward_features(samples["image"])
            encoder_output = self.text_encoder.forward_automask(
                tokenized_text=samples["tokenized_text"], visual_embeds=image_embeds
            )

            return encoder_output, image_embeds
        else:
            image_embeds = self.visual_encoder.forward_features(samples["image"])
            return image_embeds

    def forward_decoder(self, samples, image_embeds):
        # prepare inputs for forwarding decoder
        raw_text = samples["text_input"]
        text = self.tokenizer(
            raw_text,
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(self.device)
        text.input_ids[:, 0] = self.tokenizer.bos_token_id

        # prepare targets for forwarding decoder
        decoder_targets = text.input_ids.masked_fill(
            text.input_ids == self.tokenizer.pad_token_id, -100
        )
        decoder_targets[:, : self.prompt_length] = -100

        # forward decoder
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            self.device
        )
        decoder_output = self.text_decoder(
            input_ids=text.input_ids,
            attention_mask=text.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            labels=decoder_targets,
            return_dict=True,
        )

        return decoder_output, decoder_targets

    # def compute_scores(self, samples, captions):
    #     questions = samples['text_input']
    #     # # feedback1: 大模型生成答案的nll
    #     if self.feedback == 'nll':
    #         prompts = []
    #         probs = []
    #         for i in range(len(captions)):
    #             caption = captions[i]
    #             question = questions[i]
    #             prompt = f"{caption}. {question}"
    #             # prompt = f"Answer the following question in one word.\nQ: {caption} {question}"
    #             prompts.append(prompt)
    #             # inputs = self.LLM_tokenizer(prompt, return_tensors="pt")
    #             # output = self.LLM.generate(inputs.input_ids.to(self.LLM.device), max_new_tokens=50, output_scores = True, return_dict_in_generate = True)
    #             # generate_ids = output['sequences']
    #             # total_prob = 1
    #             # for i in range(len(output['scores'])):
    #             #     prob = nn.functional.softmax(output['scores'][i], dim=-1)
    #             #     total_prob *= prob[0][output['sequences'][0][i+1]]
    #             # probs.append(total_prob)
    #         inputs = self.LLM_tokenizer(prompts, return_tensors="pt", padding='longest')
    #         with torch.no_grad():
    #             output = self.LLM.generate(inputs.input_ids.to(self.LLM.device), max_new_tokens=50, output_scores = True, return_dict_in_generate = True)
    #         generate_ids = output['sequences']
    #         text = self.LLM_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    #         total_prob = torch.ones(generate_ids.size()[0]).unsqueeze(-1).to(generate_ids.device)
    #         # reward = torch.zeros(generate_ids.size()[0]).unsqueeze(-1).to(generate_ids.device)

    #         for i in range(len(output['scores'])):
    #             prob = nn.functional.softmax(output['scores'][i], dim=-1)
    #             index = generate_ids[:, i+1].reshape(generate_ids.size()[0], 1)
    #             index_prob = torch.gather(prob, 1, index)
    #             zero_indicator = torch.zeros(index_prob.size()[0]).unsqueeze(-1).to(index_prob.device)
    #             zero_indicator[index_prob<1e-3] = 1
    #             index_prob = index_prob + zero_indicator
    #             total_prob = total_prob * index_prob
    #             # reward = reward - torch.log(index_prob)
    #         rewards = total_prob.tolist()

    #     # # feedback2: caption是否包含答案
    #     elif self.feedback == 'caption':
    #         cum_idx = torch.cumsum(samples['n_answers'], dim=0)     # 累积下标
    #         cum_idx = [0] + cum_idx.tolist()
    #         rewards = []
    #         answer_reward = dict(zip(samples['answer'], samples['weight'].tolist()))
    #         for i, caption in enumerate(captions):          # 每一个caption有一个reward
    #             reward = 0
    #             answers = samples['answer'][cum_idx[i]:cum_idx[i+1]]
    #             for j, answer in enumerate(answers):
    #                 if answer in caption:
    #                     reward = reward + round(answer_reward[answer], 2)
    #             rewards.append(reward)
    #         for i in range(len(rewards)):
    #             if rewards[i] == 0:
    #                 rewards[i] = -1

    #     # feedback3: 大模型能否生成答案 (vqa score）
    #     elif self.feedback == 'answer':
    #         prompts = []
    #         for i in range(len(captions)):
    #             caption = captions[i]
    #             question = questions[i]
    #             # prompt = f"{caption}. {question}"
    #             prompt = f"Answer the following question in one word.\nQ: {caption} {question}"
    #             prompts.append(prompt)
    #         inputs = self.LLM_tokenizer(prompts, return_tensors="pt", padding='longest')
    #         with torch.no_grad():
    #             output = self.LLM.generate(inputs = inputs.input_ids.to(self.LLM.device), attention_mask = inputs.attention_mask.to(self.LLM.device), max_new_tokens=50, output_scores = True, return_dict_in_generate = True)
    #         generate_ids = output['sequences']
    #         text = self.LLM_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            
    #         cum_idx = torch.cumsum(samples['n_answers'], dim=0)     # 累积下标
    #         cum_idx = [0] + cum_idx.tolist()
    #         rewards = []
    #         from nltk.corpus import stopwords
    #         stop_words = stopwords.words('english')
    #         answer_reward = dict(zip(samples['answer'], samples['weight'].tolist()))
    #         for i, ans in enumerate(text):          # 每一个answer有一个reward
    #             if len(ans) > 1:
    #                 ans = [ele for ele in ans if ele not in stop_words]
    #                 ans = ' '.join(e for e in ans if e.isalnum())
    #             ground_truth = samples['answer'][cum_idx[i]:cum_idx[i+1]]
    #             num_match = sum([ans == gt for gt in ground_truth])
    #             vqa_acc = min(1.0, num_match / 3.0)
    #             rewards.append(vqa_acc)

    #     # feedback4: 大模型直接判断caption和question的相关程度
    #     elif self.feedback == 'confidence':
    #         prompts = []
    #         rewards = []
    #         for i in range(len(captions)):
    #             caption = captions[i]
    #             question = questions[i]
    #             prompt = f"Question: {question} Caption: {caption}\nTo what degree does the caption relate to the question:\nA: 0%\nB: 25%\nC: 50%\nD:75%"
    #             prompts.append(prompt)
    #         inputs = self.LLM_tokenizer(prompts, return_tensors="pt", padding='longest')
            
    #         with torch.no_grad():
    #             output = self.LLM.generate(inputs = inputs.input_ids.to(self.LLM.device), attention_mask = inputs.attention_mask.to(self.LLM.device), max_new_tokens=1, output_scores = True, return_dict_in_generate = True, bad_words_ids = self.bad_words_ids)
    #         generate_ids = output['sequences']
    #         text = self.LLM_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    #         for i in range(len(text)):
    #             if 'A' in text[i]:
    #                 rewards.append(0)
    #             elif 'B' in text[i]:
    #                 rewards.append(0.25)
    #             elif 'C' in text[i]:
    #                 rewards.append(0.5) 
    #             elif 'D' in text[i]:
    #                 rewards.append(0.75)
        
    #     return torch.tensor(rewards)

    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=3,
        max_length=30,
        min_length=10,
        top_p=0.9,
        repetition_penalty=1.0,
        num_captions=1,
        is_train = False,
    ):
        prompt = [self.prompt] * samples['image'].size(0)
        prompt = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt.input_ids[:, 0] = self.tokenizer.bos_token_id
        prompt.input_ids = prompt.input_ids[:, :-1]
        prompt_length = prompt.input_ids.size()[1]

        # prepare inputs for decoder generation.
        if self.has_encoder:
            encoder_out, image_embeds = self.forward_encoder(samples)
            concat_output = torch.cat([image_embeds, encoder_out.last_hidden_state], dim=1)
            concat_output = torch.repeat_interleave(concat_output, num_captions, 0)
            concat_attention_mask = torch.cat([torch.ones(image_embeds.size()[0], image_embeds.size()[1]).to(self.device), samples['tokenized_text'].attention_mask], dim=1)
            decoder_out = self.text_decoder.generate(
                input_ids=prompt.input_ids,
                max_length=max_length,
                min_length=min_length,
                do_sample=True,
                top_p=top_p,
                num_return_sequences=1,
                eos_token_id=self.tokenizer.sep_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=repetition_penalty,
                encoder_hidden_states=concat_output,
                encoder_attention_mask=concat_attention_mask,
                output_scores = True,
                return_dict_in_generate = True,)
        else:
            encoder_out = self.forward_encoder(samples)
            image_embeds = torch.repeat_interleave(encoder_out, num_captions, 0)
            # get decoded text
            decoder_out = self.text_decoder.generate_from_encoder(
                tokenized_prompt=prompt,
                visual_embeds=image_embeds,
                sep_token_id=self.tokenizer.sep_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                use_nucleus_sampling=use_nucleus_sampling,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
        # generate_ids: [bz, seq_len]
        generate_ids = decoder_out['sequences'][:, prompt_length:] 

        # tuple: len=seq_len hidden_state[i]就是第i个token所有层的hidden state，也是tuple, hidden_state[i][j]就是第i个token第j层
        # hidden_states = decoder_out['hidden_states']

        # if is_train:
        #     generate_length = torch.sum(generate_ids!=0, dim=1).unsqueeze(-1)
        #     all_logprob = []
        #     # scores, probs: [bz, seq_len, vocab_size]
        #     scores = torch.stack(decoder_out['scores']).transpose(0,1)
        #     probs = nn.functional.softmax(scores, dim=-1)
        #     for i in range(len(probs)):     # 遍历每一条文本
        #         gen_len = generate_length[i][0].item()
        #         index_prob = probs[i][:gen_len].gather(1, generate_ids[i][:gen_len].unsqueeze(-1))
        #         all_logprob.append(index_prob.squeeze(-1))
        outputs = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)
        captions = outputs # [output[len(self.prompt) :] for output in outputs]
        if is_train:
            return generate_ids, captions#, all_logprob
        else:
            return captions

    def compute_logits_values(self, generate_ids, samples):
        prompt = [self.prompt] * samples['image'].size(0)
        prompt = self.tokenizer(prompt, return_tensors="pt").to(self.text_decoder.device)
        prompt.input_ids[:, 0] = self.tokenizer.bos_token_id
        prompt.input_ids = prompt.input_ids[:, :-1]
        attention_mask = torch.zeros_like(generate_ids)
        attention_mask[generate_ids!=0]=1
        attention_mask = torch.cat([torch.ones_like(prompt.input_ids), attention_mask], dim=1)
        visual_embeds = self.visual_encoder.forward_features(samples["image"])
        image_atts = torch.ones(visual_embeds.size()[:-1], dtype=torch.long).to(visual_embeds.device)
        input_ids = torch.cat([prompt.input_ids, generate_ids], dim=1)
        
        
        last_hidden_states = self.text_decoder(input_ids=input_ids, attention_mask=attention_mask, encoder_hidden_states=visual_embeds, encoder_attention_mask=image_atts, return_last_hidden_states=True)
        logits = self.text_decoder.cls(last_hidden_states)[:, :-1, :].contiguous()[:,self.prompt_length-1:,:]       # 只保留生成caption部分的logits
        values = self.v_head(last_hidden_states[:,self.prompt_length:,:])
        # generate_length = torch.sum(generate_ids!=0, dim=-1).tolist()
        # all_values = []
        # for i in range(len(values)):
        #     all_values.append(values[i][:generate_length[i]].squeeze(-1))
        return logits, values

    @classmethod
    def from_config(cls, cfg):
        # vision encoder
        image_encoder = VisionTransformerEncoder.from_config(cfg)
        if cfg.get("encoder", False):
            text_encoder = XBertEncoder.from_config(cfg)
        else:
            text_encoder = None
        # text encoder + multimodal decoder
        text_decoder = XBertLMHeadDecoderPolicy.from_config(cfg)

        prompt = cfg.get("prompt", None)
        max_txt_len = cfg.get("max_txt_len", 40)
        feedback = cfg.get("feedback", "nll")

        # llm_path = cfg.get('llm_path', '/mnt/duyifan/ckpt/google/flan-t5-large')
        # print(f"loading LLM from {llm_path}")
        # LLM = AutoModelForSeq2SeqLM.from_pretrained(llm_path)
        # LLM_tokenizer = AutoTokenizer.from_pretrained(llm_path)
        model = cls(image_encoder, text_decoder, prompt=prompt, max_txt_len=max_txt_len, feedback = feedback, text_encoder = text_encoder, has_encoder = cfg.get("encoder", False), cfg = cfg)
        model.load_checkpoint_from_config(cfg)

        return model