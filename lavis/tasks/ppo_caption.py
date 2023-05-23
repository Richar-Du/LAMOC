import json
import os
import collections
import logging
import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from lavis.common.dist_utils import main_process
from lavis.common.registry import registry
from lavis.common.logger import MetricLogger, SmoothedValue
from lavis.tasks.captioning import CaptionTask
from lavis.datasets.data_utils import prepare_sample
import wandb

WANDB_PADDING = -1

@registry.register_task("captioning_ppo")
class PPOCaptionTask(CaptionTask):
    default_params = {
        "lr": 1.41e-5,
        "adap_kl_ctrl": True,
        "init_kl_coef":0.2,
        "target": 6,
        "horizon":10000,
        "gamma":1,
        "lam":0.95,
        "cliprange": .2,
        "cliprange_value":.2,
        "vf_coef":.1,
        "ppo_epochs": 4,
        "batch_size": 4
    }

    def __init__(self, num_beams, max_len, min_len, evaluate, report_metric=True):
        super().__init__(num_beams, max_len, min_len, evaluate, report_metric)
        self.ppo_params = self.default_params

        if self.ppo_params['adap_kl_ctrl']:
            self.kl_ctl = AdaptiveKLController(self.ppo_params['init_kl_coef'],
                                               self.ppo_params['target'],
                                               self.ppo_params['horizon'])
        else:
            self.kl_ctl = FixedKLController(self.ppo_params['init_kl_coef'])

    def batched_forward_pass(self, model, ref_model, samples):        # 即trl中的batched_forward_pass函数
        # 前向计算logp，ref_logp和values
        with torch.no_grad():
            generate_ids, captions = model.generate(samples=samples, use_nucleus_sampling=True, is_train = True, top_k = 0, top_p = 1)
        # prompt = [model.prompt] * samples['image'].size(0)
        # prompt = model.tokenizer(prompt, return_tensors="pt").to(model.device)
        # prompt.input_ids[:, 0] = model.tokenizer.bos_token_id
        # prompt.input_ids = prompt.input_ids[:, :-1]
        # prompt_length = prompt.input_ids.size()[1]
        # attention_mask = torch.zeros_like(generate_ids)
        # attention_mask[generate_ids!=0]=1
        # attention_mask = torch.cat([torch.ones_like(prompt.input_ids), attention_mask], dim=1)
        # visual_embeds = ref_model.visual_encoder.forward_features(samples["image"])
        # image_atts = torch.ones(visual_embeds.size()[:-1], dtype=torch.long).to(ref_model.device)
        # input_ids = torch.cat([prompt.input_ids, generate_ids], dim=1)
        # logits, all_values = model.compute_logits_values(input_ids=input_ids, attention_mask=attention_mask, encoder_hidden_states=visual_embeds, encoder_attention_mask=image_atts)
        # ref_logits = ref_model.text_decoder(input_ids=input_ids, attention_mask=attention_mask, encoder_hidden_states=visual_embeds, encoder_attention_mask=image_atts, return_logits=True)
        # ref_logits = ref_logits[:,prompt_length-1:,:]
        with torch.no_grad():               # 新模型和旧模型分别算p(a|s)
            # logits: [bz, seq_len, vocab_size]
            logits, values = model.compute_logits_values(generate_ids=generate_ids, samples=samples)
            ref_logits, _ = ref_model.compute_logits_values(generate_ids=generate_ids, samples=samples)
        all_logprobs = []
        all_ref_logprobs = []
        all_values = []
        # scores, probs: [bz, seq_len, vocab_size]
        logprobs = nn.functional.log_softmax(logits, dim=-1)
        ref_logprobs = nn.functional.log_softmax(ref_logits, dim=-1)
        generate_length = torch.sum(generate_ids!=0, dim=1)
        for i in range(len(logprobs)):     # 遍历每一条文本
            gen_len = generate_length[i].item()
            # index_prob: [seq_len, 1], logprobs[i][:gen_len]: [seq_len, vocab_size]   generate_ids[i][:gen_len].unsqueeze(-1): [seq_len, 1]
            index_prob = logprobs[i][:gen_len].gather(1, generate_ids[i][:gen_len].unsqueeze(-1))
            ref_index_prob = ref_logprobs[i][:gen_len].gather(1, generate_ids[i][:gen_len].unsqueeze(-1))
            all_logprobs.append(index_prob.squeeze(-1))
            all_ref_logprobs.append(ref_index_prob.squeeze(-1))
            all_values.append(values[i][:generate_length[i]].squeeze(-1))
        return all_logprobs, all_ref_logprobs, all_values, generate_ids

    def compute_scores(self, LLM, LLM_tokenizer, feedback, samples, captions):
        questions = samples['text_input']
        # # feedback1: 大模型生成答案的nll
        if feedback == 'nll':
            prompts = []
            probs = []
            for i in range(len(captions)):
                caption = captions[i]
                question = questions[i]
                prompt = f"{caption}. {question}"
                # prompt = f"Answer the following question in one word.\nQ: {caption} {question}"
                prompts.append(prompt)
                # inputs = LLM_tokenizer(prompt, return_tensors="pt")
                # output = LLM.generate(inputs.input_ids.to(LLM.device), max_new_tokens=50, output_scores = True, return_dict_in_generate = True)
                # generate_ids = output['sequences']
                # total_prob = 1
                # for i in range(len(output['scores'])):
                #     prob = nn.functional.softmax(output['scores'][i], dim=-1)
                #     total_prob *= prob[0][output['sequences'][0][i+1]]
                # probs.append(total_prob)
            inputs = LLM_tokenizer(prompts, return_tensors="pt", padding='longest')
            with torch.no_grad():
                output = LLM.generate(inputs.input_ids.to(LLM.device), max_new_tokens=50, output_scores = True, return_dict_in_generate = True)
            generate_ids = output['sequences']
            text = LLM_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
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
            rewards = total_prob.tolist()

        # # feedback2: caption是否包含答案
        elif feedback == 'caption':
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
            for i in range(len(rewards)):
                if rewards[i] == 0:
                    rewards[i] = -1

        # feedback3: 大模型能否生成答案 (vqa score）
        elif feedback == 'answer':
            prompts = []
            for i in range(len(captions)):
                caption = captions[i]
                question = questions[i]
                # prompt = f"{caption}. {question}"
                prompt = f"Answer the following question in one word.\nQ: {caption} {question}"
                prompts.append(prompt)
            inputs = LLM_tokenizer(prompts, return_tensors="pt", padding='longest')
            with torch.no_grad():
                output = LLM.generate(inputs = inputs.input_ids.to(LLM.device), attention_mask = inputs.attention_mask.to(LLM.device), max_new_tokens=50, output_scores = True, return_dict_in_generate = True)
            generate_ids = output['sequences']
            text = LLM_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            
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

        # feedback4: 大模型直接判断caption和question的相关程度
        elif feedback == 'confidence':
            force_words_ids = LLM_tokenizer(['A', 'B', 'C', 'D'], add_special_tokens=False, return_tensors="pt").input_ids
            vocab_ids = list(range(LLM_tokenizer.vocab_size))
            for ele in reversed(force_words_ids):
                del vocab_ids[ele[0]]
            bad_words_ids = [[ele] for ele in vocab_ids]
            prompts = []
            rewards = []
            for i in range(len(captions)):
                caption = captions[i]
                question = questions[i]
                prompt = f"Question: {question} Caption: {caption}\nTo what degree does the caption relate to the question:\nA: 0%\nB: 25%\nC: 50%\nD:75%"
                prompts.append(prompt)
            inputs = LLM_tokenizer(prompts, return_tensors="pt", padding='longest')
            with torch.no_grad():
                output = LLM.generate(inputs = inputs.input_ids.to(LLM.device), attention_mask = inputs.attention_mask.to(LLM.device), max_new_tokens=1, output_scores = True, return_dict_in_generate = True, bad_words_ids = bad_words_ids)
            generate_ids = output['sequences']
            text = LLM_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for i in range(len(text)):
                if 'A' in text[i]:
                    rewards.append(0.0)
                elif 'B' in text[i]:
                    rewards.append(0.25)
                elif 'C' in text[i]:
                    rewards.append(0.5) 
                elif 'D' in text[i]:
                    rewards.append(0.75)
        
        return torch.tensor(rewards)

    def compute_rewards(self, scores, logprobs, ref_logprobs):
        """Compute per token rewards from scores and KL-penalty."""
        rewards, non_score_rewards = [], []
        for score, logprob, ref_logprob in zip(scores, logprobs, ref_logprobs):
            kl = logprob - ref_logprob
            non_score_reward = -self.kl_ctl.value * kl          # self.kl_ctl.value是kl的系数
            non_score_rewards.append(non_score_reward)
            reward = non_score_reward.clone()
            reward[-1] += score             # 只有最后一个token才有反馈的打分，其他的token都只有kl divergence
            rewards.append(reward)
        return rewards, non_score_rewards

    def loss(self, model, old_logprobs, values, rewards, samples, generate_ids):
        """Calculate policy and value losses."""
        # old_logprobs, values, rewards, response: [bz, seq_len]
        # model_input是query+response
        lastgaelam = 0
        advantages_reversed = []
        gen_len = generate_ids.size()[1]
        # generate_length = torch.sum(generate_ids!=0, dim=-1)
        
        # import ipdb
        # ipdb.set_trace()
        for t in reversed(range(gen_len)):      # 每一个token都计算一个advantage
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0           # values不应该换成rewards？
            # nextvalues = values[:, t + 1] if t < gen_len - 1 else torch.zeros(values.size()[0])
            delta = rewards[:, t] + self.ppo_params['gamma'] * nextvalues - values[:, t]            
            lastgaelam = delta + self.ppo_params['gamma'] * self.ppo_params['lam'] * lastgaelam         # lastgaelam是啥
            advantages_reversed.append(lastgaelam)
        # advantages: [bz, seq_len]
        advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)
        
        # [bz, seq_len]
        returns = advantages + values           # ???
        advantages = whiten(advantages)
        advantages = advantages.detach()
        
        # logits: [bz, seq_len, vocab_size], vpred: [bz, seq_len, 1]
        logits, vpred = model.compute_logits_values(generate_ids=generate_ids, samples=samples)
        logprob = nn.functional.log_softmax(logits, dim=-1)
        # logprob: [1, gen_len]
        logprob = logprob[0][:gen_len].gather(1, generate_ids[0][:gen_len].unsqueeze(-1)).T
        # [bz, gen_len]
        vpred = vpred[:,:gen_len,:].squeeze(-1)

        vpredclipped = clip_by_value(vpred,
                                     values - self.ppo_params["cliprange_value"],
                                     values + self.ppo_params["cliprange_value"])           # 为什么是values为基础不是vpred为基础？

        # vf_losses1: [bz, gen_len]
        vf_losses1 = (vpred - returns)**2               # vf_loss是什么loss？
        vf_losses2 = (vpredclipped - returns)**2
        vf_loss = .5 * torch.mean(torch.max(vf_losses1, vf_losses2))
        vf_clipfrac =  torch.mean(torch.gt(vf_losses2, vf_losses1).double())

        # ratio: [bz, gen_len]
        ratio = torch.exp(logprob - old_logprobs)

        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio,
                                               1.0 - self.ppo_params['cliprange'],
                                               1.0 + self.ppo_params['cliprange'])

        pg_loss = torch.mean(torch.max(pg_losses, pg_losses2))
        pg_clipfrac = torch.mean(torch.gt(pg_losses2, pg_losses).double())

        loss = pg_loss + self.ppo_params['vf_coef'] * vf_loss

        entropy = torch.mean(entropy_from_logits(logits))
        approxkl = .5 * torch.mean((logprob - old_logprobs)**2)
        policykl = torch.mean(logprob - old_logprobs)
        return_mean, return_var = torch.mean(returns), torch.var(returns)
        value_mean, value_var = torch.mean(values), torch.var(values)

        stats = dict(
            loss=dict(policy=pg_loss, value=vf_loss, total=loss),
            policy=dict(entropy=entropy, approxkl=approxkl,policykl=policykl, clipfrac=pg_clipfrac,
                        advantages=advantages, advantages_mean=torch.mean(advantages), ratio=ratio),
            returns=dict(mean=return_mean, var=return_var),
            val=dict(vpred=torch.mean(vpred), error=torch.mean((vpred - returns) ** 2),
                     clipfrac=vf_clipfrac, mean=value_mean, var=value_var),
        )
        return pg_loss, self.ppo_params['vf_coef'] * vf_loss, flatten_dict(stats)

    def train_step(self, optimizer, LLM, LLM_tokenizer, feedback, model, ref_model, samples, scaler, accum_grad_iters):
        # print(model.text_decoder.bert.encoder.layer[0].attention.self.query.weight)
        use_amp = scaler is not None
        with torch.cuda.amp.autocast(enabled=use_amp):
            all_logprobs, all_ref_logprobs, all_values, generate_ids = self.batched_forward_pass(model, ref_model, samples)
            captions = model.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)
            scores = self.compute_scores(LLM, LLM_tokenizer, feedback, samples, captions)
            wandb.log({"scores": scores.mean()})
            rewards, non_score_reward = self.compute_rewards(scores, all_logprobs, all_ref_logprobs)
            bs = len(all_logprobs)
            idxs = list(range(bs))
            all_stats = []
        losses = []
        for i in range(self.ppo_params['ppo_epochs']):      # batch_size个样本重复用4次
            """Train one PPO minibatch"""
            # with torch.cuda.amp.autocast(enabled=use_amp):
            #     loss_p, loss_v, train_stats = self.loss(model=model, old_logprobs=all_logprobs, values=all_values, rewards=rewards, samples=samples, generate_ids=generate_ids)       # query相当于image+question，response相当于caption，model_input是二者拼起来，都是input_ids
            #     loss = loss_p + loss_v
            # losses.append(loss)
            # if use_amp:
            #     scaler.scale(loss).backward()
            # else:
            #     loss.backward()
            # # update gradients every accum_grad_iters iterations
            # if (i + 1) % accum_grad_iters == 0:
            #     optimizer.step()
            #     optimizer.zero_grad()
            # all_stats.append(train_stats)

            random.shuffle(idxs)
            for i in range(bs):                 # batch_size个样本一个个地过
                idx = idxs[i]
                gen_len = torch.sum(generate_ids[idx]!=0, dim=-1)
                sub_sample = {'image': samples['image'][idx].unsqueeze(0)}
                try:
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        loss_p, loss_v, train_stats = self.loss(model=model, old_logprobs=all_logprobs[idx].unsqueeze(0), values=all_values[idx].unsqueeze(0), rewards=rewards[idx].unsqueeze(0), samples=sub_sample, generate_ids=generate_ids[idx][:gen_len].unsqueeze(0))       # query相当于image+question，response相当于caption，model_input是二者拼起来，都是input_ids
                        loss = loss_p + loss_v
                    losses.append(loss)
                    # after_train_step()
                    if use_amp:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 0.1) 
                    # update gradients every accum_grad_iters iterations
                    if (i + 1) % accum_grad_iters == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                    all_stats.append(train_stats)
                except:
                    print("some sample error... continue")
                    continue
        
        train_stats = stack_dicts(all_stats)
        # reshape advantages/ratios such that they are not averaged.
        train_stats['policy/advantages'] = torch.flatten(train_stats['policy/advantages']).unsqueeze(0)
        train_stats['policy/advantages'] = torch.nan_to_num(train_stats['policy/advantages'], WANDB_PADDING)
        train_stats['policy/ratio'] = torch.flatten(train_stats['policy/ratio']).unsqueeze(0)
        stats = self.record_step_stats(scores=scores, logprobs=all_logprobs, ref_logprobs=all_ref_logprobs,
                                       non_score_reward=non_score_reward, train_stats=train_stats,
                                       kl_coef=self.kl_ctl.value)
        stats = stats_to_np(stats)
        self.kl_ctl.update(stats['objective/kl'], self.ppo_params['batch_size'])
        
        return torch.tensor(losses).mean(), stats

    def train_epoch(
        self,
        epoch,
        LLM,
        LLM_tokenizer,
        feedback,
        model,
        ref_model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
    ):
        return self._train_inner_loop(
            epoch=epoch,
            iters_per_epoch=len(data_loader),
            LLM=LLM,
            LLM_tokenizer=LLM_tokenizer,
            feedback=feedback,
            model=model,
            ref_model=ref_model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
        )

    def _train_inner_loop(
        self,
        epoch,
        iters_per_epoch,
        LLM,
        LLM_tokenizer,
        feedback,
        model,
        ref_model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        start_iters=None,
        log_freq=50,
        cuda_enabled=False,
        accum_grad_iters=1,
    ):
        """
        An inner training loop compatible with both epoch-based and iter-based training.

        When using epoch-based, training stops after one epoch; when using iter-based,
        training stops after #iters_per_epoch iterations.
        """

        if not hasattr(data_loader, "__next__"):
            # convert to iterator if not already
            data_loader = iter(data_loader)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        # if iter-based runner, schedule lr based on inner epoch.
        logging.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                epoch, iters_per_epoch
            )
        )
        header = "Train: data epoch: [{}]".format(epoch)
        if start_iters is None:
            # epoch-based runner
            inner_epoch = epoch
        else:
            # In iter-based runner, we schedule the learning rate based on iterations.
            inner_epoch = start_iters // iters_per_epoch
            header = header + "; inner epoch [{}]".format(inner_epoch)

        for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
            # if using iter-based runner, we stop after iters_per_epoch iterations.
            if i >= iters_per_epoch:
                break

            samples = next(data_loader)         # 一个batch的样本

            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            samples.update(
                {
                    "epoch": inner_epoch,
                    "num_iters_per_epoch": iters_per_epoch,
                    "iters": i,
                }
            )

            lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i+inner_epoch*iters_per_epoch)
            loss, stats= self.train_step(optimizer=optimizer, LLM=LLM, LLM_tokenizer=LLM_tokenizer, feedback=feedback, model=model, ref_model = ref_model, samples=samples, scaler=scaler, accum_grad_iters=accum_grad_iters)

            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            wandb.log({"lr": optimizer.param_groups[0]["lr"], "loss": loss.item()})

        # after train_epoch()
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        return {
            k: "{:.3f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }

        # def batched_forward_pass(self, model, ref_model, samples, input_ids):

    def evaluation(self, model, data_loader, LLM, LLM_tokenizer, feedback, cuda_enabled=True):
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        # TODO make it configurable
        print_freq = 10

        results = []

        for samples in metric_logger.log_every(data_loader, print_freq, header):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

            eval_output = self.valid_step(model=model, samples=samples, LLM=LLM, LLM_tokenizer=LLM_tokenizer, feedback=feedback)
            results.extend(eval_output)

        # dist.barrier()

        return results

    def valid_step(self, model, samples, LLM, LLM_tokenizer, feedback):
        force_words_ids = LLM_tokenizer(['A', 'B', 'C', 'D'], add_special_tokens=False, return_tensors="pt").input_ids
        vocab_ids = list(range(LLM_tokenizer.vocab_size))
        for ele in reversed(force_words_ids):
            del vocab_ids[ele[0]]
        bad_words_ids = [[ele] for ele in vocab_ids]

        results = []
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
        inputs = LLM_tokenizer(prompts, return_tensors="pt", padding='longest')
        
        with torch.no_grad():
            output = LLM.generate(inputs = inputs.input_ids.to(LLM.device), attention_mask = inputs.attention_mask.to(LLM.device), max_new_tokens=1, output_scores = True, return_dict_in_generate = True, bad_words_ids = bad_words_ids)
        generate_ids = output['sequences']
        text = LLM_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for i in range(len(text)):
            if 'A' in text[i]:
                rewards.append(0)
            elif 'B' in text[i]:
                rewards.append(0.25)
            elif 'C' in text[i]:
                rewards.append(0.5) 
            elif 'D' in text[i]:
                rewards.append(0.75)
        
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

    def record_step_stats(self, kl_coef, **data):
        """Record training step statistics."""
        kl_list = [logprobs-ref_logprobs for logprobs, ref_logprobs in zip(data['logprobs'], data['ref_logprobs'])]
        mean_kl = torch.mean(torch.stack([torch.sum(kl) for kl in kl_list]))
        mean_entropy = torch.mean(torch.stack([torch.sum(-log_probs) for log_probs in data['logprobs']]))
        mean_non_score_reward =torch.mean(torch.stack([torch.sum(non_score_reward) for non_score_reward in data['non_score_reward']]))
        stats = {
            'objective/kl': mean_kl,
            'objective/kl_dist': kl_list,
            'objective/logprobs': data['logprobs'],
            'objective/ref_logprobs': data['ref_logprobs'],
            'objective/kl_coef': kl_coef,
            'objective/entropy': mean_entropy,
            'ppo/mean_non_score_reward': mean_non_score_reward,
        }

        for k, v in data['train_stats'].items():
            stats[f'ppo/{k}'] = torch.mean(v, axis=0)
        stats['ppo/val/var_explained'] = 1 - stats['ppo/val/error'] / stats['ppo/returns/var']
        return stats


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """
    def __init__(self, init_kl_coef, target, horizon):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current, n_steps):
        target = self.target
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult

# Cell

class FixedKLController:
    """Fixed KL controller."""
    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current, n_steps):
        pass

def whiten(values, shift_mean=True):
    """Whiten values."""
    mean, var = torch.mean(values), torch.var(values)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened

def clip_by_value(x, tensor_min, tensor_max):
    """
    Tensor extenstion to torch.clamp
    https://github.com/pytorch/pytorch/issues/2793#issuecomment-428784713
    """
    clipped = torch.max(torch.min(x, tensor_max), tensor_min)
    return clipped

def entropy_from_logits(logits):
    """Calculate entropy from logits."""
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, axis=-1) - torch.sum(pd*logits, axis=-1)
    return entropy

def flatten_dict(nested, sep='/'):
    """Flatten dictionary and concatenate nested keys with separator."""
    def rec(nest, prefix, into):
        for k, v in nest.items():
            if sep in k:
                raise ValueError(f"separator '{sep}' not allowed to be in key '{k}'")
            if isinstance(v, collections.Mapping):
                rec(v, prefix + k + sep, into)
            else:
                into[prefix + k] = v
    flat = {}
    rec(nested, '', flat)
    return flat

def stack_dicts(stats_dicts):
    """Stack the values of a dict."""
    results = dict()
    for k in stats_dicts[0]:
        stats_list = [torch.flatten(d[k]) for d in stats_dicts]
        results[k] = pad_sequence(stats_list, batch_first=True, padding_value=WANDB_PADDING)
    return results

def stats_to_np(stats_dict):
    """Cast all torch.tensors in dict to numpy arrays."""
    new_dict = dict()
    for k, v in stats_dict.items():
        if isinstance(v, torch.Tensor):
            new_dict[k] = v.detach().cpu().numpy()
        else:
            new_dict[k] = v
        if np.isscalar(new_dict[k]):
            new_dict[k] = float(new_dict[k])
    return new_dict