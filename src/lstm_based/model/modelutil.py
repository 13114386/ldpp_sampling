from __future__ import unicode_literals, print_function, division
import os
import torch
from utility.utility import saveToJson


main_model_state_dicts = [
    'encoder_state_dict',
    'enc2dec_state_h_dict',
    'enc2dec_state_c_dict',
    'decoder_state_dict'
]

optional_model_state_dicts = [
    'dpp_search'
]

def parse_reload_options(fpath):
    import glob
    flist = glob.glob(fpath)
    if len(flist) == 0:
        default_fname = "options_best.json"
        fdir, _ = os.path.split(fpath)
        fpath = os.path.join(fdir, default_fname)
        flist = glob.glob(fpath)
    if len(flist) == 0:
        fpath = ""
    elif len(flist) == 1:
        fpath = flist[0]
    else:
        import regex
        pat = regex.compile("options\_\w*\_?epoch\_(\d+)\_batch\_\d+.json")
        epoch_nums = [int(pat.findall(f)[0]) for f in flist]
        last_epoch = max(epoch_nums)
        max_index = epoch_nums.index(last_epoch)
        fpath = flist[max_index]
    return fpath

def save_model_checkpoint(model, optimizers, options, log,
                            epoch_index, batch_index, best_score,
                            batch_count, method = 'best', model_ext=".pth"):
    log.log('Start Saving Options')

    if method == 'best':
        strId = '_best'
    elif method == 'best_epoch':
        strId = '_best_epoch_'+str(epoch_index)
    elif method == 'best_epoch_batch':
        strId = '_best_epoch_'+str(epoch_index)+'_batch_'+str(batch_index)
    elif method == 'epoch':
        strId = '_epoch_'+str(epoch_index)
    elif method == 'epoch_batch':
        strId = '_epoch_'+str(epoch_index)+'_batch_'+str(batch_index)
    elif method == 'check2':
        strId = '_check2_epoch_'+str(epoch_index)+'_batch_'+str(batch_index)
    elif method == 'check2_best':
        strId = '_check2_best'

    # Only need to save whatever is changed during training and
    # model config used to run the training
    training_state = {}
    training_state['start_epoch'] = epoch_index+1
    training_state['best_score'] = best_score
    training_state['batch_count'] = batch_count
    training_state['reload'] = True
    training_state['reload_options'] = 'options'+strId+'.json'
    training_state['reload_model'] = 'model'+strId+model_ext
    save_state = {"model_cfg": options["model_cfg"],
                  "train_state": training_state,
                  "exclude_modules": options["exclude_modules"]}

    log.log('Start saving model checkpoint')
    save_folder = os.path.join(options['modeldata_root'],
                                options['model_path'],
                                options['model_name'])
    os.makedirs(save_folder, exist_ok=True)
    # Save training state
    saveToJson(os.path.join(save_folder, 'options'+strId+'.json'), save_state)
    # Save model & optimizer
    optimizer_state = [optimizer.state_dict() for optimizer in optimizers]
    save_path = os.path.join(save_folder, 'model'+strId+model_ext)
    # Main model state
    saved_modules = ['encoder_state_dict',
                    'enc2dec_state_h_dict',
                    'enc2dec_state_c_dict',
                    'decoder_state_dict']
    model_state = {
        'encoder_state_dict': model.encoder.state_dict(),
        'enc2dec_state_h_dict': model.enc2dec_state_h.state_dict(),
        'enc2dec_state_c_dict': model.enc2dec_state_c.state_dict(),
        'decoder_state_dict': model.decoder.state_dict()
    }
    # Configurable model states
    if "dpp_search" not in options["exclude_modules"]:
        model_state['dpp_search_dict'] = model.dpp_search.state_dict()
        saved_modules.append('dpp_search_dict')

    # Add optimizer state now
    model_state['optimizer_state_dict'] = optimizer_state
    # Save
    torch.save(model_state, save_path)
    log.log(saved_modules)
    log.log("Model checkpoint is saved")

def load_model_checkpoint(model, checkpoint, options, log):
    log.log('Start loading model checkpoint')
    loaded_modules = ['encoder_state_dict',
                    'enc2dec_state_h_dict',
                    'enc2dec_state_c_dict',
                    'decoder_state_dict']
    model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
    model.enc2dec_state_h.load_state_dict(checkpoint['enc2dec_state_h_dict'])
    model.enc2dec_state_c.load_state_dict(checkpoint['enc2dec_state_c_dict'])
    model.decoder.load_state_dict(checkpoint['decoder_state_dict'])
    # Configurable model states
    if "dpp_search_dict" in checkpoint and \
        "dpp_search" not in options["exclude_modules"]:
        model.dpp_search.load_state_dict(checkpoint['dpp_search_dict'])
        loaded_modules.append('dpp_search_dict')

    log.log(loaded_modules)
    log.log("Model is checkpoint initialised")


def setup_single_optimizers(model, options, checkpoint=None):
    optimizers = []
    params = list(model.encoder.parameters()) + \
            list(model.enc2dec_state_h.parameters()) + \
            list(model.enc2dec_state_c.parameters()) + \
            list(model.decoder.parameters())
    if "dpp_search" not in options["exclude_modules"]:
        params += list(model.dpp_search.parameters())

    optim_cfg = options["optimizers"]["main"]
    main_optimizer = eval('torch.optim.' + optim_cfg["optimizer"])
    optimizers.append(main_optimizer(params,
                                    lr=optim_cfg["lr"],
                                    weight_decay=optim_cfg["weight_decay"]))

    # Load checkpoint
    if checkpoint:
        opt_ckpts = checkpoint['optimizer_state_dict']
        assert len(opt_ckpts) == len(optimizers)
        try:
            [optimizer.load_state_dict(opt_ckpts[i]) \
                for i, optimizer in enumerate(optimizers)]
        except:
            # Module inclusion/exclusion between runs may have changed and resulted in
            # the inconsistence with saved optimizer states. But move on.
            pass
    return optimizers


def setup_multi_optimizers(model, options, checkpoint=None):
    optimizers = []

    # Main model optimizer
    params = list(model.encoder.parameters()) + \
                list(model.enc2dec_state_h.parameters()) + \
                list(model.enc2dec_state_c.parameters()) + \
                list(model.decoder.parameters())

    optim_cfg = options["optimizers"]["main"]
    main_optimizer = eval('torch.optim.' + optim_cfg["optimizer"])
    optimizers.append(main_optimizer(params,
                                    lr=optim_cfg["lr"],
                                    weight_decay=optim_cfg["weight_decay"]))

    # Constrained function optimizer
    params = []
    if "dpp_search" not in options["exclude_modules"]:
        params += list(model.dpp_search.parameters())

    if len(params) > 0:
        optim_cfg = options["optimizers"]["secondary"]
        secondary_optimizer = eval('torch.optim.' + optim_cfg["optimizer"])
        optimizers.append(secondary_optimizer(params,
                                            lr=optim_cfg["lr"],
                                            weight_decay=optim_cfg["weight_decay"]))

    # Load checkpoint
    if checkpoint:
        opt_ckpts = checkpoint['optimizer_state_dict']
        assert len(opt_ckpts) == len(optimizers)
        try:
            [optimizer.load_state_dict(opt_ckpts[i]) \
                for i, optimizer in enumerate(optimizers)]
        except:
            # Module inclusion/exclusion between runs may have changed and resulted in
            # the inconsistence with saved optimizer states. But move on.
            pass
    return optimizers
