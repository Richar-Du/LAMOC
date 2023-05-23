"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import os
import torch
import torch.nn as nn

from lavis.common.dist_utils import main_process
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask


@registry.register_task("captioning")
class CaptionTask(BaseTask):
    def __init__(self, num_beams, max_len, min_len, evaluate, report_metric=True):
        super().__init__()

        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len
        self.evaluate = evaluate

        self.report_metric = report_metric

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        num_beams = run_cfg.num_beams
        max_len = run_cfg.max_len
        min_len = run_cfg.min_len
        evaluate = run_cfg.evaluate

        report_metric = run_cfg.get("report_metric", True)

        return cls(
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            evaluate=evaluate,
            report_metric=report_metric,
        )

    def valid_step(self, model, samples):
        results = []

        # run_cfg = slf.cfg.run_cfg
        captions = model.generate(
            samples,
            use_nucleus_sampling=False,
            num_beams=self.num_beams,
            max_length=self.max_len,
            min_length=self.min_len,
        )

        img_ids = samples["image_id"]
        for caption, img_id in zip(captions, img_ids):
            results.append({"caption": caption, "image_id": int(img_id)})

        return results

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        if 'image_id' in val_result[0].keys():
            remove_duplicate = 'image_id'
        elif 'question_id' in val_result[0].keys():
            remove_duplicate = 'question_id'
        eval_result_file = self.save_result(
            result=val_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_epoch{}".format(split_name, epoch),
            remove_duplicate=remove_duplicate,
        )

        if self.report_metric:
            metrics = self._report_metrics(
                eval_result_file=eval_result_file, split_name=split_name
            )
        else:
            metrics = {"agg_metrics": 0.0}

        return metrics

    @main_process
    def _report_metrics(self, eval_result_file, split_name):

        # TODO better way to define this
        coco_gt_root = os.path.join(registry.get_path("cache_root"), "coco_gt")
        coco_val = coco_caption_eval(coco_gt_root, eval_result_file, split_name)

        agg_metrics = coco_val.eval["CIDEr"] + coco_val.eval["Bleu_4"]
        log_stats = {split_name: {k: v for k, v in coco_val.eval.items()}}

        with open(
            os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")

        coco_res = {k: v for k, v in coco_val.eval.items()}
        coco_res["agg_metrics"] = agg_metrics

        return coco_res

@registry.register_task("policy_captioning")
class PolicyCaptionTask(CaptionTask):
    
    def valid_step(self, model, samples):
        results = []
        if model.feedback == 'nll':
            questions = samples['text_input']
            prompts = []
            probs = []
            captions = model.generate(
                samples,
                use_nucleus_sampling=False,
                top_k=50,
                num_beams=self.num_beams,
                max_length=self.max_len,
                min_length=self.min_len,
            )
            for i in range(len(captions)):
                caption = captions[i]
                question = questions[i]
                prompt = f"Please answer the following question.\n {caption}. {question}"
                prompts.append(prompt)
            inputs = model.LLM_tokenizer(prompts, return_tensors="pt", padding='longest')
            with torch.no_grad():
                output = model.LLM.generate(inputs.input_ids.to(model.LLM.device), max_new_tokens=50, output_scores = True, return_dict_in_generate = True)
            generate_ids = output['sequences']
            text = model.LLM_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            total_prob = torch.ones(generate_ids.size()[0]).unsqueeze(-1).to(generate_ids.device)

            for i in range(len(output['scores'])):
                prob = nn.functional.softmax(output['scores'][i], dim=-1)
                index = generate_ids[:, i+1].reshape(generate_ids.size()[0], 1)
                index_prob = torch.gather(prob, 1, index)
                zero_indicator = torch.zeros(index_prob.size()[0]).unsqueeze(-1).to(index_prob.device)
                zero_indicator[index_prob<1e-3] = 1
                index_prob = index_prob + zero_indicator
                total_prob = total_prob * index_prob
            rewards = total_prob.squeeze(-1).tolist()

        if model.feedback == 'confidence':
            force_words_ids = model.LLM_tokenizer(['A', 'B', 'C', 'D'], add_special_tokens=False, return_tensors="pt").input_ids
            vocab_ids = list(range(model.LLM_tokenizer.vocab_size))
            for ele in reversed(force_words_ids):
                del vocab_ids[ele[0]]
            bad_words_ids = [[ele] for ele in vocab_ids]

            # run_cfg = slf.cfg.run_cfg
            captions = model.generate(
                samples,
                use_nucleus_sampling=False,
                num_beams=self.num_beams,
                max_length=self.max_len,
                min_length=self.min_len,
            )
            prompts = []
            rewards = []
            for i in range(len(captions)):
                caption = captions[i]
                question = samples['text_input'][i]
                prompt = f"Question: {question} Caption: {caption}\nTo what degree does the caption relate to the question:\nA: 0%\nB: 25%\nC: 50%\nD:75%"
                prompts.append(prompt)
            inputs = model.LLM_tokenizer(prompts, return_tensors="pt", padding='longest')
            
            with torch.no_grad():
                output = model.LLM.generate(inputs = inputs.input_ids.to(model.LLM.device), attention_mask = inputs.attention_mask.to(model.LLM.device), max_new_tokens=1, output_scores = True, return_dict_in_generate = True, bad_words_ids = bad_words_ids)
            generate_ids = output['sequences']
            text = model.LLM_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for i in range(len(text)):
                if 'A' in text[i]:
                    rewards.append(0)
                elif 'B' in text[i]:
                    rewards.append(0.25)
                elif 'C' in text[i]:
                    rewards.append(0.5) 
                elif 'D' in text[i]:
                    rewards.append(0.75)
        
        # # 算reward的时候用label
        # cum_idx = torch.cumsum(samples['n_answers'], dim=0)     # 累积下标
        # rewards = []
        # answer_reward = dict(zip(samples['answer'], samples['weight'].tolist()))
        # for i, caption in enumerate(captions):          # 每一个caption有一个reward
        #     reward = 0
        #     answers = samples['answer'][:cum_idx[i]]
        #     for j, answer in enumerate(answers):
        #         if answer in caption:
        #             reward = reward + round(answer_reward[answer], 2)
        #     rewards.append(reward)
        if 'image_id' in samples.keys():
            img_ids = samples["image_id"]
            for caption, img_id, reward in zip(captions, img_ids, rewards):
                results.append({"caption": caption, "image_id": int(img_id), "reward": reward})
        elif 'question_id' in samples.keys():
            question_ids = samples['question_id']
            for caption, question_id, reward in zip(captions, question_ids, rewards):
                results.append({"caption": caption, "question_id": int(question_id), "reward": reward})

        return results

    @main_process
    def _report_metrics(self, eval_result_file, split_name):
        eval_result = json.load(open(eval_result_file, 'r'))
        reward = 0
        for result in eval_result:
            reward += result['reward']
        reward_avg = (reward)/len(eval_result)
        return {"agg_metrics": reward_avg}
        
@registry.register_task("aokvqa_captioning")
class AOKVQACaptionTask(CaptionTask):
    
    def valid_step(self, model, samples):

        # run_cfg = slf.cfg.run_cfg
        loss = model(samples)['loss']
        results = [{"loss": loss.item()}]
        
        return results

    @main_process
    def _report_metrics(self, eval_result_file, split_name):
        eval_result = json.load(open(eval_result_file, 'r'))
        loss = 0
        for result in eval_result:
            loss += result['loss']
        loss_avg = (loss)/len(eval_result)
        return {"agg_metrics": 1/loss_avg}

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        eval_result_file = self.save_result(
            result=val_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_epoch{}".format(split_name, epoch),
        )

        if self.report_metric:
            metrics = self._report_metrics(
                eval_result_file=eval_result_file, split_name=split_name
            )
        else:
            metrics = {"agg_metrics": 0.0}

        return metrics


# TODO better structure for this.
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO
from torchvision.datasets.utils import download_url


def coco_caption_eval(coco_gt_root, results_file, split):
    urls = {
        "val": "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val_gt.json",
        "test": "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test_gt.json",
    }
    filenames = {
        "val": "coco_karpathy_val_gt.json",
        "test": "coco_karpathy_test_gt.json",
    }

    download_url(urls[split], coco_gt_root)
    annotation_file = os.path.join(coco_gt_root, filenames[split])

    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    # coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    # coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f"{metric}: {score:.3f}")

    return coco_eval
