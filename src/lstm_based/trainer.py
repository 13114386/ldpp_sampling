from __future__ import unicode_literals, print_function, division

import os
import torch
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
# from torch.optim import Adam
from common.metric_meter import MetricMeter, ProgressCounter
from common.timer_deco import timerdeco
from utility.utility import *
from data_processor.data_manager import batch2Inputs_new, collect_data
from checkpoint import Checkpoint


class Trainer():
    def __init__(self, model, options, validator, optimizers):
        self.model = model
        self.optimizers = optimizers
        self.validator = validator
        self.checkpoint = Checkpoint(options)
        log_dir = os.path.join(options['modeldata_root'],
                                options['model_path'],
                                options['model_name'],
                                "run")
        self.summary_writer = SummaryWriter(log_dir)

    def train_epoch(self, iepoch, train_dataloader, val_dataloader, options, log,
                    avg_cost_mtr,
                    train_glb_counter, valid_glb_counter):
        log_freq = 20
        early_stop = False
        steps = options["n_iterations"]
        # Reset lr scheduler per epoch
        lr_schedulers = self.__reset_scheduler(self.optimizers, options["optimizers"])
        for ibatch, batch in zip(range(steps), train_dataloader):
            samples, _ = batch(options)
            # Train model
            self.model.train()
            self.model.zero_grad(set_to_none=True)
            # [optimizer.zero_grad(set_to_none=True) for optimizer in self.optimizers]
            output = self.model(samples, training=True, iepoch=iepoch)
            cost = output["cost"]
            cost_data = cost.item()
            avg_cost_mtr(cost_data)
            train_glb_counter += 1
            self.summary_writer.add_scalar("Loss/train", cost_data, train_glb_counter.count)
            self.summary_writer.add_scalar("Avg loss/train", avg_cost_mtr.average(), train_glb_counter.count)
            cost.backward()
            if options["grad_clip"] > 0 and \
               train_glb_counter.count % options["grad_clip_freq"] == 0:
                torch.nn.utils.clip_grad_value_(self.model.parameters(), options["grad_clip"])
            [optimizer.step() for optimizer in self.optimizers]
            # Adjust learning rate
            [scheduler.step() for scheduler in lr_schedulers]

            if (ibatch+1) % log_freq == 0:
                log.log('Train: Epoch %d, iBatch %d: Cost %f, AvgCost %f'%\
                        (iepoch, ibatch, cost_data, avg_cost_mtr.average()))
            # Snapshot validation & checkpoint
            index = train_glb_counter.count
            if (options['sample'] and (index >= options['sampleMin']) and \
                (index % options['sampleFreq']) == 0):
                score = self.validator(iepoch=iepoch, ibatch=ibatch, steps=steps,
                                        model=self.model, val_dataloader=val_dataloader,
                                        options=options, summary_writer=self.summary_writer,
                                        log=log, training=True,
                                        valid_glb_counter=valid_glb_counter)
                log.log("This is a check point")
                early_stop = self.checkpoint(iepoch, ibatch, score,
                                            self.model, self.optimizers, options, log,
                                            reason="best")
                if early_stop:
                    break
        if (ibatch+1) % log_freq != 0: # Log remaning batch results
            log.log('Train: Epoch %d, iBatch %d: Cost %f, AvgCost %f'%\
                    (iepoch, ibatch, cost_data, avg_cost_mtr.average()))
        # Run regular validation per epoch
        score = self.validator(iepoch=iepoch, ibatch=ibatch, steps=steps,
                                model=self.model, val_dataloader=val_dataloader,
                                options=options, summary_writer=self.summary_writer,
                                log=log, training=True,
                                valid_glb_counter=valid_glb_counter)
        self.summary_writer.flush()
        return score, ibatch, early_stop

    @timerdeco()
    def __call__(self, options, dataset, log):
        start_epoch = options["train_state"]["start_epoch"]
        avg_cost_mtr = MetricMeter(0.95)
        train_dataloader, val_dataloader = dataset
        train_glb_counter = ProgressCounter(start_epoch*options["n_iterations"])
        valid_glb_counter = ProgressCounter(start_epoch*options["n_iterations"])
        for iepoch in range(start_epoch, options["max_epochs"]):
            log.log('Epoch %d'%(iepoch))
            torch.cuda.empty_cache()
            score, ibatch, early_stop = \
                self.train_epoch(iepoch, train_dataloader, val_dataloader,
                                options, log, avg_cost_mtr,
                                train_glb_counter, valid_glb_counter)
            if early_stop:
                break
            if options['save_mode']["on_epoch"] and \
                (iepoch+1) % options['save_mode']["freq"] == 0:
                self.checkpoint(iepoch, ibatch, score,
                                self.model, self.optimizers,
                                options, log)
        # Save the last one if it wasn't saved.
        if options['save_mode']["on_epoch"] and \
            (iepoch+1) % options['save_mode']["freq"] != 0:
            self.checkpoint(iepoch, ibatch, score,
                            self.model, self.optimizers,
                            options, log)

    def __reset_scheduler(self, optimizers, optim_cfg):
        cfgs = [optim_cfg["main"], optim_cfg["secondary"]]
        lr_schedulers = []
        for optimizer, cfg in zip(optimizers, cfgs[:len(optimizers)]):
            for param_group in optimizer.param_groups:
                param_group['lr'] = cfg["lr"]
            # lr_schedulers += [lr_scheduler.CosineAnnealingLR(optimizer, cfg["lr_decay"])]
            lr_schedulers += [lr_scheduler.ExponentialLR(optimizer, cfg["lr_decay"])]
        return lr_schedulers
