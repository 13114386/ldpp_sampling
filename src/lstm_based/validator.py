from __future__ import unicode_literals, print_function, division
import torch
from common.metric_meter import MetricMeter

class Validator():
    def __init__(self, part):
        self.part = part

    def __call__(self, iepoch, ibatch, steps, model, val_dataloader,
                 options, summary_writer, log, training,
                 valid_glb_counter):
        model.eval()
        log_freq = 20
        avg_cost_mtr = MetricMeter(0.95)
        for idx, batch in zip(range(steps), val_dataloader):
            samples, _ = batch(options)
            with torch.no_grad():
                output = model(samples, training=training)
                cost = output["cost"].item()
                avg_cost_mtr(cost)
            if (idx+1) % log_freq == 0:
                log.log('Validate: Epoch %d, iBatch %d: Cost %f, AvgCost %f'%\
                        (iepoch, idx, cost, avg_cost_mtr.average()))

            valid_glb_counter += 1
            summary_writer.add_scalar("Loss/val", cost, valid_glb_counter.count)
            summary_writer.add_scalar("Avg loss/val", avg_cost_mtr.average(), valid_glb_counter.count)

        if (idx+1) % log_freq != 0: # Log remaning batch results
            log.log('Validate: Epoch %d, iBatch %d: Cost %f, AvgCost %f'%\
                    (iepoch, idx, cost, avg_cost_mtr.average()))
        return avg_cost_mtr.average()
