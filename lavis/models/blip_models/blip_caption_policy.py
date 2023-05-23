"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

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


@registry.register_model("blip_caption_policy")
class BlipCaptionPolicy(BlipCaption):
    
    PRETRAINED_MODEL_CONFIG_DICT = {
        # "base_coco": "configs/models/blip_caption_policy.yaml",
        "large_coco": "configs/models/blip_caption_policy.yaml",
    }

    def __init__(self, image_encoder, text_decoder, llm_path, prompt=None, max_txt_len=40, feedback='nll', text_encoder = None, has_encoder = False, cfg=None):
        super().__init__(image_encoder, text_decoder, prompt, max_txt_len)
        # self.LLM = LLM
        # self.LLM_tokenizer = LLM_tokenizer
        print(f"loading LLM from {llm_path}")
        self.LLM = AutoModelForSeq2SeqLM.from_pretrained(llm_path)
        try:
            self.LLM_tokenizer = AutoTokenizer.from_pretrained(llm_path)
        except:
            self.LLM_tokenizer = AutoTokenizer.from_pretrained('/mnt/duyifan/ckpt/google/flan-t5-large')
        for name, param in self.LLM.named_parameters():
            param.requires_grad = False
        self.feedback = feedback
        if self.feedback == 'confidence':
            force_words_ids = self.LLM_tokenizer(['A', 'B', 'C', 'D'], add_special_tokens=False, return_tensors="pt").input_ids
            vocab_ids = list(range(self.LLM_tokenizer.vocab_size))
            for ele in reversed(force_words_ids):
                del vocab_ids[ele[0]]
            self.bad_words_ids = [[ele] for ele in vocab_ids]
        self.has_encoder = has_encoder
        self.text_encoder = text_encoder
        self.cfg = cfg

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
        raw_text = samples["caption"]
        for ele in raw_text:
            ele = self.prompt + ele
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

    def forward(self, samples):
        num_beams = self.cfg.get('num_beams', 3)
        num_return_sequences = self.cfg.get('num_return_sequences', 1)
        top_k = self.cfg.get('top_k', 100)
        top_p = self.cfg.get('top_p', 0.9)
        use_nucleus_sampling = self.cfg.get('use_nucleus_sampling', True)

        captions, nll, image_embeds = self.generate(samples=samples, use_nucleus_sampling=use_nucleus_sampling, num_beams=num_beams, num_return_sequences=num_return_sequences, top_k=top_k, top_p=top_p, is_train = True)
        questions = samples['text_input']
        decoder_output, decoder_targets = self.forward_decoder(samples, image_embeds)
        lm_loss = decoder_output.loss
        # # feedback1: 大模型生成答案的nll
        if self.feedback == 'nll':
            prompts = []
            probs = []
            for i in range(len(captions)):
                caption = captions[i]
                question = questions[i]
                prompt = f"Please answer the following question.\n {caption}. {question}"
                # prompt = f"{caption}. {question}"
                # prompt = f"Answer the following question in one word.\nQ: {caption} {question}"
                prompts.append(prompt)
                # inputs = self.LLM_tokenizer(prompt, return_tensors="pt")
                # output = self.LLM.generate(inputs.input_ids.to(self.LLM.device), max_new_tokens=50, output_scores = True, return_dict_in_generate = True)
                # generate_ids = output['sequences']
                # total_prob = 1
                # for i in range(len(output['scores'])):
                #     prob = nn.functional.softmax(output['scores'][i], dim=-1)
                #     total_prob *= prob[0][output['sequences'][0][i+1]]
                # probs.append(total_prob)
            inputs = self.LLM_tokenizer(prompts, return_tensors="pt", padding='longest')
            with torch.no_grad():
                output = self.LLM.generate(inputs.input_ids.to(self.LLM.device), max_new_tokens=50, output_scores = True, return_dict_in_generate = True)
            generate_ids = output['sequences']
            text = self.LLM_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            total_prob = torch.ones(generate_ids.size()[0]).unsqueeze(-1).to(generate_ids.device)
            # reward = torch.zeros(generate_ids.size()[0]).unsqueeze(-1).to(generate_ids.device)

            for i in range(len(output['scores'])):
                prob = nn.functional.softmax(output['scores'][i], dim=-1)
                index = generate_ids[:, i+1].reshape(generate_ids.size()[0], 1)
                index_prob = torch.gather(prob, 1, index)
                zero_indicator = torch.zeros(index_prob.size()[0]).unsqueeze(-1).to(index_prob.device)
                zero_indicator[index_prob<1e-3] = 1
                index_prob = index_prob + zero_indicator
                total_prob = total_prob * index_prob
                # reward = reward - torch.log(index_prob)
            rewards_mean = torch.mean(total_prob)
            total_prob = total_prob - rewards_mean     # 归一化reward
            rl_loss = total_prob.to(nll.device) * nll

        # # feedback2: caption是否包含答案
        elif self.feedback == 'caption':
            cum_idx = torch.cumsum(samples['n_answers'], dim=0)     # 累积下标
            cum_idx = [0] + cum_idx.tolist()
            rewards = []
            answer_reward = dict(zip(samples['answer'], samples['weight'].tolist()))
            for i, caption in enumerate(captions):          # 每一个caption有一个reward
                reward = 0
                answers = samples['answer'][cum_idx[i]:cum_idx[i+1]]
                for j, answer in enumerate(answers):
                    if answer in caption:
                        reward = reward + round(answer_reward[answer], 2)
                rewards.append(reward)
            rewards = torch.tensor(rewards).unsqueeze(-1).to(nll.device)
            rewards[rewards==0] = -1
            rl_loss = rewards * nll

        # feedback3: 大模型能否生成答案 (vqa score）
        elif self.feedback == 'answer':
            prompts = []
            for i in range(len(captions)):
                caption = captions[i]
                question = questions[i]
                # prompt = f"{caption}. {question}"
                prompt = f"Answer the following question in one word.\nQ: {caption} {question}"
                prompts.append(prompt)
            inputs = self.LLM_tokenizer(prompts, return_tensors="pt", padding='longest')
            with torch.no_grad():
                output = self.LLM.generate(inputs = inputs.input_ids.to(self.LLM.device), attention_mask = inputs.attention_mask.to(self.LLM.device), max_new_tokens=50, output_scores = True, return_dict_in_generate = True)
            generate_ids = output['sequences']
            text = self.LLM_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            
            cum_idx = torch.cumsum(samples['n_answers'], dim=0)     # 累积下标
            cum_idx = [0] + cum_idx.tolist()
            rewards = []
            from nltk.corpus import stopwords
            stop_words = stopwords.words('english')
            answer_reward = dict(zip(samples['answer'], samples['weight'].tolist()))
            for i, ans in enumerate(text):          # 每一个answer有一个reward
                if len(ans) > 1:
                    ans = [ele for ele in ans if ele not in stop_words]
                    ans = ' '.join(e for e in ans if e.isalnum())
                ground_truth = samples['answer'][cum_idx[i]:cum_idx[i+1]]
                num_match = sum([ans == gt for gt in ground_truth])
                vqa_acc = min(1.0, num_match / 3.0)
                rewards.append(vqa_acc)
            rewards = torch.tensor(rewards).unsqueeze(-1).to(nll.device)
            rewards = rewards - torch.mean(rewards)
            rl_loss = rewards * nll

        elif self.feedback == 'confidence':
            prompts = []
            rewards = []
            for i in range(len(captions)):
                caption = captions[i]
                question = questions[i]
                prompt = f"Question: {question} Caption: {caption}\nTo what degree does the caption relate to the question:\nA: 0%\nB: 25%\nC: 50%\nD:75%"
                prompts.append(prompt)
            inputs = self.LLM_tokenizer(prompts, return_tensors="pt", padding='longest')
            
            with torch.no_grad():
                output = self.LLM.generate(inputs = inputs.input_ids.to(self.LLM.device), attention_mask = inputs.attention_mask.to(self.LLM.device), max_new_tokens=1, output_scores = True, return_dict_in_generate = True, bad_words_ids = self.bad_words_ids)
            generate_ids = output['sequences']
            text = self.LLM_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for i in range(len(text)):
                if 'A' in text[i]:
                    rewards.append(0)
                elif 'B' in text[i]:
                    rewards.append(0.25)
                elif 'C' in text[i]:
                    rewards.append(0.5) 
                elif 'D' in text[i]:
                    rewards.append(0.75)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(nll.device).reshape(nll.size()[0], 1)
            rewards_mean = torch.mean(rewards)
            rewards = rewards - rewards_mean
            assert rewards.size() == nll.size()
            rl_loss = rewards * nll
        alpha = self.cfg.get('alpha', 0.8)
        factor = self.cfg.get('factor', 1)
        loss = (1-alpha)*lm_loss + alpha*factor*rl_loss.mean()
        return {'loss': loss.mean(), 'rewards': rewards_mean, 'rl_loss': rl_loss.mean(), 'lm_loss': lm_loss}

    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=3,
        num_return_sequences=1,
        max_length=30,
        min_length=10,
        top_p=0.9,
        top_k=0,
        repetition_penalty=1.0,
        num_captions=1,
        is_train = False,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
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
                num_return_sequences=num_return_sequences,
                max_length=max_length,
                min_length=min_length,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )
        # generate_ids: [bz, length]
        generate_ids = decoder_out['sequences'][:, prompt_length:]     
        if is_train:
            generate_length = torch.sum(generate_ids!=0, dim=1).unsqueeze(-1)
            nll = torch.zeros([image_embeds.size()[0]]).unsqueeze(-1).to(generate_ids.device)
            # probability = torch.ones([image_embeds.size()[0]]).unsqueeze(-1).to(generate_ids.device)
            for i in range(len(decoder_out['scores'])):         # 遍历文本长度(不加prompt)
                # prob: [bz, vocab_size]
                prob = nn.functional.softmax(decoder_out['scores'][i], dim=-1)      
                # # generate_ids[:, i]: bz, index: [bz, 1]
                index = generate_ids[:, i].reshape(generate_ids.size()[0], 1)
                # # index_prob: [bz,1]
                index_prob = torch.gather(prob, 1, index)
                zero_indicator = torch.zeros(index_prob.size()[0]).unsqueeze(-1).to(index_prob.device)
                zero_indicator[index_prob==0] = 1
                index_prob = index_prob + zero_indicator            # 短句子生成结束后面的概率都是0，防止nll变成-inf，用1补上
                nll = nll - torch.log(index_prob)
                # probability = probability * index_prob
            # nll = -torch.log(total_prob)/(len(decoder_out['scores']))
            nll_avg = nll / generate_length
        
        outputs = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)
        captions = outputs # [output[len(self.prompt) :] for output in outputs]
        if is_train:
            return captions, nll_avg, image_embeds
        else:
            return captions


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

        llm_path = cfg.get('llm_path', '/mnt/duyifan/ckpt/google/flan-t5-large')
        # print(f"loading LLM from {llm_path}")
        # LLM = AutoModelForSeq2SeqLM.from_pretrained(llm_path)
        # try:
        #     LLM_tokenizer = AutoTokenizer.from_pretrained(llm_path)
        # except:
        #     LLM_tokenizer = AutoTokenizer.from_pretrained('/mnt/duyifan/ckpt/google/flan-t5-large')
        model = cls(image_encoder, text_decoder, llm_path=llm_path, prompt=prompt, max_txt_len=max_txt_len, feedback = feedback, text_encoder = text_encoder, has_encoder = cfg.get("encoder", False), cfg=cfg)
        model.load_checkpoint_from_config(cfg)

        return model