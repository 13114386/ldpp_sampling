from __future__ import unicode_literals, print_function, division
from model.modelutil import save_model_checkpoint

class Checkpoint():
    def __init__(self, options):
        if options['reload']:
            self.best_score = options["train_state"]['best_score']
            self.batch_count = options["train_state"]['batch_count']
        else:
            self.best_score = 1e99
            self.batch_count = 0

    def __call__(self, iepoch, ibatch, score,
                 model, optimizers, options, log, reason="regular"):
        log.log('Score is %f, best score is %f'%(score, self.best_score))
        # Learning rates
        lrs = [param_group['lr'] for optimizer in optimizers \
                for param_group in optimizer.param_groups]
        fmt = ', '.join('%.15f' % l for l in lrs)
        fmt = 'Current learning Rate is/are ' + fmt
        log.log(fmt)
        early_stop = False
        if score < self.best_score:
            log.log('Find a better model')
            self.best_score = score
            self.batch_count = 0

            log.log('Update Best Model')
            save_model_checkpoint(model, optimizers, options, log, iepoch, ibatch,
                                self.best_score, self.batch_count, 'best')
            save_model_checkpoint(model, optimizers, options, log, iepoch, ibatch,
                                self.best_score, self.batch_count, 'best_epoch_batch')
        else:
            self.batch_count += options['sampleFreq']
            if (options['earlyStop'] and \
                self.batch_count >= options['earlyStop_bound']):
                log.log('Early Stopping')
                early_stop = True
            if reason == "regular":
                log.log('Update Model on Epoch')
                save_model_checkpoint(model, optimizers, options, log, iepoch, ibatch,
                                    score, self.batch_count, 'epoch_batch')
        return early_stop
