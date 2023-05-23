"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch
import torch.nn as nn
from lavis.common.registry import registry

from lavis.models.blip_models.blip_vqa import BlipVQA
from lavis.models.blip_models.blip_outputs import (
    BlipOutput,
    BlipIntermediateOutput,
)
from lavis.models.med import XBertEncoder, XBertLMHeadDecoder
from lavis.models.vit import VisionTransformerEncoder


@registry.register_model("blip_vqa_caption")
class BlipVQACaption(BlipVQA):
    
    PRETRAINED_MODEL_CONFIG_DICT = {
        # "base_coco": "configs/models/blip_caption_base_coco.yaml",
        "aokvqa": "configs/models/blip_vqa_caption.yaml",
    }

    def __init__(self, image_encoder, text_encoder, text_decoder, max_txt_len=40):
        super().__init__(image_encoder, text_encoder, text_decoder, max_txt_len)
        self.prompt = 'a picture of '
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids) - 1

    def forward_encoder(self, samples, num_captions = 1):
        questions = samples["text_input"]
        questions = self.tokenizer(
            questions,
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(self.device)
        questions.input_ids[:, 0] = self.tokenizer.enc_token_id
        samples.update({"tokenized_question": questions})

        image_embeds = self.visual_encoder.forward_features(samples["image"])
        # if num_captions != 1:           # 只有生成时才会num_captions!=1
        #     encoder_outputs = []
        #     for i in range(num_captions):
        #         sample_idx = torch.randint(low = 1, high = image_embeds.size(1), size = (1,100))
        #         image_embeds=image_embeds[:,sample_idx[0],:]
        #         encoder_output = self.text_encoder.forward_automask(
        #         tokenized_text=samples["tokenized_question"], visual_embeds=image_embeds
        #         )
        #         encoder_outputs.append(encoder_output)
        #     return encoder_outputs
        # else:
        encoder_output = self.text_encoder.forward_automask(
            tokenized_text=samples["tokenized_question"], visual_embeds=image_embeds
        )

        return encoder_output, image_embeds
            # return image_embeds

    def forward_decoder(self, samples, encoder_out, **kwargs):
        for i in range(len(samples['caption'])):
            samples['caption'][i] = self.prompt + samples['caption'][i]
        question = samples["tokenized_question"]
        # ====================================================================
        raw_text = samples["caption"]        # 带有prompt的caption
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
        concat_output = torch.cat([kwargs['image_embeds'], encoder_out.last_hidden_state], dim=1)
        concat_attention_mask = torch.cat([torch.ones(kwargs['image_embeds'].size()[0], kwargs['image_embeds'].size()[1]).to(self.device), question.attention_mask], dim=1)
        # image_attention_mask = torch.ones(kwargs['image_embeds'].size()[0], kwargs['image_embeds'].size()[1]).to(self.device)
        answer_output = self.text_decoder(
            text.input_ids,
            attention_mask=text.attention_mask,
            # encoder_hidden_states=question_output.last_hidden_state,
            encoder_hidden_states=concat_output,
            # encoder_attention_mask=question.attention_mask,
            encoder_attention_mask=concat_attention_mask,
            labels=decoder_targets,
            return_dict=True,
            # reduction="none",
        )
        # bsz = samples["image"].size(0)
        # loss = answer_output.loss.sum() / bsz

        return answer_output, decoder_targets

    # def forward_decoder(self, samples, image_embeds):
    #     # prepare inputs for forwarding decoder
    #     for i in range(len(samples['caption'])):
    #         samples['caption'][i] = self.prompt + samples['caption'][i]
    #     raw_text = samples["caption"]        # 带有prompt的caption
    #     text = self.tokenizer(
    #         raw_text,
    #         padding="longest",
    #         truncation=True,
    #         max_length=self.max_txt_len,
    #         return_tensors="pt",
    #     ).to(self.device)
    #     text.input_ids[:, 0] = self.tokenizer.bos_token_id

    #     # prepare targets for forwarding decoder
    #     decoder_targets = text.input_ids.masked_fill(
    #         text.input_ids == self.tokenizer.pad_token_id, -100
    #     )
    #     decoder_targets[:, : self.prompt_length] = -100

    #     # forward decoder
    #     image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
    #         self.device
    #     )
    #     # text.input_ids, text.attention_mask, decoder_targets: [bz, txt_len]
    #     # image_embeds: [bz, img_len, dim]
    #     # image_atts: [bz, img_len]
    #     decoder_output = self.text_decoder(
    #         input_ids=text.input_ids,
    #         attention_mask=text.attention_mask,
    #         encoder_hidden_states=image_embeds,
    #         encoder_attention_mask=image_atts,
    #         labels=decoder_targets,
    #         return_dict=True,
    #     )
    #     # bsz = samples["image"].size(0)
    #     # loss = decoder_output.loss.sum() / bsz

    #     return decoder_output, decoder_targets

    def forward(self, samples):
        # image_embeds = self.forward_encoder(samples)
        encoder_output, image_embeds = self.forward_encoder(samples)
        decoder_output, decoder_targets = self.forward_decoder(
            samples=samples, encoder_out=encoder_output, image_embeds = image_embeds
        )
        # decoder_output, decoder_targets = self.forward_decoder(samples=samples, image_embeds=image_embeds)


        return BlipOutput(
            loss=decoder_output.loss,
            loss_lm=decoder_output.loss,
            intermediate_output=BlipIntermediateOutput(
                image_embeds=image_embeds,
                encoder_output=encoder_output,
                decoder_output=decoder_output,
                decoder_labels=decoder_targets,
            ),
        )

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
        # prepare inputs for decoder generation.
        encoder_outs, image_embeds = self.forward_encoder(samples, num_captions=num_captions)
        sample_image_embeds = []
        for i in range(num_captions):
            sample_idx = torch.randint(low = 1, high = image_embeds.size(1), size = (1,100))
            sample_image_embeds.append(image_embeds[0][sample_idx])
        sample_image_embeds = torch.cat(sample_image_embeds, dim=0)
        encoder_outs = torch.repeat_interleave(encoder_outs.last_hidden_state, num_captions, 0)
        
        prompt = [self.prompt] * sample_image_embeds.size(0)
        prompt = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt.input_ids[:, 0] = self.tokenizer.bos_token_id
        prompt.input_ids = prompt.input_ids[:, :-1]

        concat_output = torch.cat([sample_image_embeds, encoder_outs], dim=1)
        question_mask = torch.repeat_interleave(samples['tokenized_question'].attention_mask, num_captions, 0)
        concat_attention_mask = torch.cat([torch.ones(sample_image_embeds.size()[0], sample_image_embeds.size()[1]).to(self.device), question_mask], dim=1)
        

        # get decoded text
        decoder_out = self.text_decoder.generate(
            input_ids=prompt.input_ids,
            sep_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            use_nucleus_sampling=use_nucleus_sampling,
            # num_beams=num_beams,
            max_length=max_length,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            encoder_hidden_states=concat_output,
            encoder_attention_mask=concat_attention_mask,
        )
        outputs = self.tokenizer.batch_decode(decoder_out, skip_special_tokens=True)
        captions = [output[len(self.prompt) :] for output in outputs]
        
        # question_outputs = encoder_outs
        # captions = []
        # for i in range(num_captions):
        #     question_output = question_outputs[i]
        #     question_states = question_output.last_hidden_state.repeat_interleave(
        #         num_beams, dim=0
        #     )
        #     question_atts = torch.ones(question_states.size()[:-1], dtype=torch.long).to(
        #         self.device
        #     )

        #     model_kwargs = {
        #         "encoder_hidden_states": question_states,
        #         "encoder_attention_mask": question_atts,
        #     }

        #     bsz = samples["image"].size(0)
        #     bos_ids = torch.full(
        #         (bsz, 1), fill_value=self.tokenizer.bos_token_id, device=self.device
        #     )

        #     outputs = self.text_decoder.generate(
        #         input_ids=bos_ids,
        #         max_length=max_length,
        #         min_length=min_length,
        #         num_beams=num_beams,
        #         eos_token_id=self.tokenizer.sep_token_id,
        #         pad_token_id=self.tokenizer.pad_token_id,
        #         **model_kwargs
        #     )
        #     caption = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        #     captions.append(caption)
        return captions


    @classmethod
    def from_config(cls, cfg=None):
        image_encoder = VisionTransformerEncoder.from_config(cfg)

        # text encoder + multimodal encoder
        text_encoder = XBertEncoder.from_config(cfg)
        text_decoder = XBertLMHeadDecoder.from_config(cfg)

        max_txt_len = cfg.get("max_txt_len", 35)

        model = cls(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            text_decoder=text_decoder,
            max_txt_len=max_txt_len,
        )

        model.load_checkpoint_from_config(cfg)

        return model
