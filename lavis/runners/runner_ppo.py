import datetime
import logging
import os
import time

import torch
import torch.distributed as dist
import webdataset as wds
from lavis.common.dist_utils import download_cached_file, is_main_process, main_process
from lavis.common.registry import registry
from lavis.common.utils import is_url
from lavis.datasets.data_utils import concat_datasets, reorg_datasets_by_split
from lavis.runners.runner_base import RunnerBase
from torch.utils.data.dataset import ChainDataset
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

@registry.register_runner("runner_ppo")
class RunnerPPO(RunnerBase):
    def __init__(self, cfg, task, model, ref_model, datasets, job_id):
        super().__init__(cfg, task, model, datasets, job_id)
        self._ref_model = ref_model
        self._wrapped_ref_model = None
        for name, parameter in self._ref_model.named_parameters():
            parameter.requires_grad = False
        llm_path = cfg.model_cfg.llm_path
        print(f"loading LLM from {llm_path}")
        self.LLM = AutoModelForSeq2SeqLM.from_pretrained(llm_path)
        self.LLM_tokenizer = AutoTokenizer.from_pretrained(llm_path)
        self.feedback = cfg.model_cfg.feedback
            
    @property
    def model(self):
        """
        A property to get the DDP-wrapped model on the device.
        """
        # move model to device
        if self._model.device != self.device:
            self._model = self._model.to(self.device)         # 更通用的写法
            # self._model.visual_encoder = self._model.visual_encoder.to(self.device)
            # try:
            #     self._model.text_encoder = self._model.text_encoder.to(self.device)
            # except:
            #     pass
            # self._model.text_decoder = self._model.text_decoder.to(self.device)
            # self._model.v_head = self._model.v_head.to(self.device)


            # distributed training wrapper
            if self.use_distributed:
                if self._wrapped_model is None:
                    self._wrapped_model = DDP(
                        self._model, device_ids=[self.config.run_cfg.gpu]
                    )
            else:
                self._wrapped_model = self._model

        return self._wrapped_model

    @property
    def ref_model(self):
        # move model to device
        if self._ref_model.device != self.device:
            self._ref_model = self._ref_model.to(self.device)         # 更通用的写法
            # self._ref_model.visual_encoder = self._ref_model.visual_encoder.to(self.device)
            # try:
            #     self._ref_model.text_encoder = self._ref_model.text_encoder.to(self.device)
            # except:
            #     pass
            # self._ref_model.text_decoder = self._ref_model.text_decoder.to(self.device)
            # self._ref_model.v_head = self._ref_model.v_head.to(self.device)

            # distributed training wrapper
            if self.use_distributed:
                if self._wrapped_ref_model is None:
                    self._wrapped_model = DDP(
                        self._ref_model, device_ids=[self.config.run_cfg.gpu]
                    )
            else:
                self._wrapped_ref_model = self._ref_model

        return self._wrapped_ref_model

    def train_epoch(self, epoch):
        # train
        self.model.eval()
        self.ref_model.eval()

        return self.task.train_epoch(
            epoch=epoch,
            LLM=self.LLM,
            LLM_tokenizer=self.LLM_tokenizer,
            feedback=self.feedback,
            model=self.model,
            ref_model=self.ref_model,
            data_loader=self.train_loader,
            optimizer=self.optimizer,
            scaler=self.scaler,
            lr_scheduler=self.lr_scheduler,
            cuda_enabled=self.cuda_enabled,
            log_freq=self.log_freq,
            accum_grad_iters=self.accum_grad_iters,
        )

    @torch.no_grad()
    def eval_epoch(self, split_name, cur_epoch, skip_reload=False):
        """
        Evaluate the model on a given split.

        Args:
            split_name (str): name of the split to evaluate on.
            cur_epoch (int): current epoch.
            skip_reload_best (bool): whether to skip reloading the best checkpoint.
                During training, we will reload the best checkpoint for validation.
                During testing, we will use provided weights and skip reloading the best checkpoint .
        """
        data_loader = self.dataloaders.get(split_name, None)
        assert data_loader, "data_loader for split {} is None.".format(split_name)

        # TODO In validation, you need to compute loss as well as metrics
        # TODO consider moving to model.before_evaluation()
        model = self.unwrap_dist_model(self.model)
        if not skip_reload and cur_epoch == "best":
            model = self._reload_best_model(model)
        model.eval()

        self.task.before_evaluation(
            model=model,
            dataset=self.datasets[split_name],
        )
        results = self.task.evaluation(model, data_loader, self.LLM, self.LLM_tokenizer, self.feedback)

        if results is not None:
            return self.task.after_evaluation(
                val_result=results,
                split_name=split_name,
                epoch=cur_epoch,
            )
