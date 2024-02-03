from __future__ import unicode_literals, print_function, division
'''
    Refactor from https://github.com/huggingface/transformers/blob/master/examples/pytorch/summarization/run_summarization_no_trainer.py
'''
import torch
from common.metric_meter import MetricMeter, ProgressCounter
from common.timer_deco import timerdeco
from common.ml_except import EarlyStopException
from common.earlystop_cond import EarlyStopConditionByCount
from utility.utility import *
from checkpoint import get_start_point

class Trainer():
    def __init__(self, validator, checkpoint):
        self.validator = validator
        self.checkpoint = checkpoint

    @timerdeco()
    def train_epoch(
        self,
        iepoch,
        start_iteration,
        options,
        train_dataloader,
        val_dataloader,
        model,
        optimizer,
        lr_scheduler,
        accelerator,
        max_steps_stop,
        avg_cost_mtr,
        train_glb_counter,
        log_freq,
        summary_writer,
        logger,
    ):
        early_stop = False
        model.train()
        n_train_iterations = options.training.n_train_iterations
        for ibatch, batch in zip(range(start_iteration, n_train_iterations), train_dataloader):
            outputs = model(batch, options=options, iepoch=iepoch)
            loss = outputs["cost"]
            cost_data = loss.item()
            avg_cost_mtr(cost_data)
            train_glb_counter += 1
            if summary_writer:
                summary_writer.add_scalar("Loss/train", cost_data, train_glb_counter.count)
                summary_writer.add_scalar("Avg loss/train", avg_cost_mtr.average(), train_glb_counter.count)

            if (ibatch+1) % log_freq == 0:
                logger.info('Train: Epoch %d, iBatch %d: Cost %f, AvgCost %f'%\
                            (iepoch, ibatch, cost_data, avg_cost_mtr.average()))

            # Normalize loss to account for batch accumulation.
            loss = loss / options.training.gradient_accumulation_steps

            accelerator.backward(loss)

            # Weights update on gradient accumulation.
            if (ibatch+1) % options.training.gradient_accumulation_steps == 0 or \
                ibatch == n_train_iterations - 1:
                optimizer.step()
                lr_scheduler.step()
                model.zero_grad(set_to_none=True)
                # progress_bar.update(1)
                # Clip weights.
                if options.training.grad_clip > 0 and \
                    train_glb_counter.count % options.training.grad_clip_freq == 0:
                    accelerator.clip_grad_value_(model.parameters(), options.training.grad_clip)
                # Stop check.
                max_steps_stop.incr()
                early_stop = max_steps_stop()
                if early_stop:
                    break

        if (ibatch+1) % log_freq != 0: # Log remaning batch results
            logger.info('Train: Epoch %d, iBatch %d: Cost %f, AvgCost %f'%\
                        (iepoch, ibatch, cost_data, avg_cost_mtr.average()))
        # Run regular validation per epoch
        try:
            self.validator(iepoch=iepoch,
                            ibatch=ibatch,
                            options=options,
                            model=model,
                            val_dataloader=val_dataloader,
                            accelerator=accelerator,
                            glb_count=train_glb_counter.count,
                            summary_writer=summary_writer,
                            logger=logger)
        except EarlyStopException as ex:
            logger.warning(str(ex))
            early_stop = True

        if summary_writer:
            summary_writer.flush()
        return ibatch, early_stop

    @timerdeco("session")
    def __call__(
        self,
        args,
        options,
        datasets,
        model,
        optimizer,
        lr_scheduler,
        accelerator,
        max_train_steps,
        summary_writer,
        logger
    ):
        avg_cost_mtr = MetricMeter(0.95)
        train_dataloader, val_dataloader = datasets
        (start_epoch, start_iteration) = get_start_point(options.training.n_train_iterations,
                                                        options.training.train_state.train_global_count)
        logger.info(f"start_epoch, start_iteration: ({start_epoch}, {start_iteration})")

        if start_epoch >= options.training.max_epochs:
            logger.warning(f"Start epoch {start_epoch} should be less than "
                           f"max epochs {options.training.max_epochs}")
            return

        # progress_bar = tqdm(range(args.max_train_steps),
        #                   disable=not accelerator.is_local_main_process)
        max_steps_stop = EarlyStopConditionByCount(max_train_steps)
        train_glb_counter = ProgressCounter(options.training.train_state.train_global_count)
        for iepoch in range(start_epoch, options.training.max_epochs):
            logger.info(f'Epoch {iepoch}')
            torch.cuda.empty_cache()
            ibatch, early_stop = \
                self.train_epoch(iepoch=iepoch,
                                 start_iteration=start_iteration,
                                 options=options,
                                 train_dataloader=train_dataloader,
                                 val_dataloader=val_dataloader,
                                 model=model,
                                 optimizer=optimizer,
                                 lr_scheduler=lr_scheduler,
                                 accelerator=accelerator,
                                 max_steps_stop=max_steps_stop,
                                 avg_cost_mtr=avg_cost_mtr,
                                 train_glb_counter=train_glb_counter,
                                 log_freq=args.log_freq,
                                 summary_writer=summary_writer,
                                 logger=logger)

            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            reason = "last" if early_stop or (iepoch+1==options.training.max_epochs) else "regular"
            early_stop_2 = self.checkpoint(iepoch=iepoch,
                                          ibatch=ibatch,
                                          args=args,
                                          options=options,
                                          train_global_count=train_glb_counter.count,
                                          score=avg_cost_mtr.average(),
                                          model=unwrapped_model,
                                          optimizers=[optimizer],
                                          accelerator=accelerator,
                                          reason=reason,
                                          logger=logger)
            early_stop = early_stop or early_stop_2
            # Reset start iteration.
            start_iteration = 0

            if early_stop:
                break

        logger.info(f"Training completes @ epoch {iepoch}.")
