from __future__ import unicode_literals, print_function, division
import os
from abc import ABC, abstractmethod
from common.auto_config import AutoConfig
from file_utils import get_save_load_dir
from options_loader import OptionsLoader


def extract_base_filename(fpath, file_pattern, logger=None):
    import regex
    pat = regex.compile(file_pattern)
    m = pat.findall(fpath)
    if m:
        if logger is not None:
            logger.info(m)
        return m[-1][0]+m[-1][1]
    else:
        return fpath


class SaveLoadBase(ABC):
    @abstractmethod
    def save(
        self,
        epoch_index,
        batch_index,
        args,
        options,
        model,
        optimizers,
        best_score,
        batch_count,
        train_global_count,
        folder='epoch_batch',
        reason=None,
        regex_file_pattern=None,
        logger=None
    ):
        raise NotImplemented("SaveLoadBase not implemented.")

    def make_id(self, epoch_index, batch_index, method):
        if method == 'best_epoch_batch':
            str_id = '_best_epoch_'+str(epoch_index)+'_batch_'+str(batch_index)
        elif method == 'epoch_batch':
            str_id = '_epoch_'+str(epoch_index)+'_batch_'+str(batch_index)
        else:
            raise ValueError("Unknown checkpoint saving method.")
        return str_id

    def get_save_dir(self, args, model_name, logger, str_id=""):
        save_folder = os.path.join(get_save_load_dir(args, model_name), str_id)
        os.makedirs(save_folder, exist_ok=True)
        logger.info(f"Save directory: {save_folder}")
        return save_folder

    def save_config_state(
        self,
        epoch_index,
        batch_index,
        args,
        options,
        best_score,
        batch_count,
        train_global_count,
        save_folder,
        reason,
        str_id,
        regex_file_pattern,
        logger
    ):
        #== Save training configurations ==#
        # Baseline configuration
        _, baseline_config_name = os.path.split(options.saveload.baseline_option)
        config_path = os.path.join(save_folder, baseline_config_name)
        options.base_model.to_json_file(config_path, use_diff=False)
        logger.info(f"Model config is saved to {config_path}")

        # Auxiliary configurations
        _, aux_config_name = os.path.split(options.saveload.aux_option)
        config_path = os.path.join(save_folder, aux_config_name)
        options.aux_model.to_json_file(config_path, use_diff=False)
        logger.info(f"Aux config is saved to {config_path}")

        # Training state
        # Only need to save whatever is changed during training
        train_state = AutoConfig(**{"start_epoch": epoch_index+1,
                                    "train_global_count": train_global_count,
                                    "best_score": best_score,
                                    "batch_count": batch_count,
                                    "baseline_option": baseline_config_name,
                                    "aux_option": aux_config_name,
                                    "model_id": str_id})
        _, train_state_file = os.path.split(options.saveload.train_state_option)
        train_state_path = os.path.join(save_folder, train_state_file)
        train_state.to_json_file(train_state_path, use_diff=False)
        logger.info(f"Training state is saved to {train_state_path}")
        # Save a signature to identify the best model.
        signature_path = os.path.join(save_folder, f"{reason}.signature")
        with open(signature_path, "w") as fp:
            logger.info("A signature is saved.")

    def load_saved_config(self, mode, args, logger):
        return OptionsLoader.load(mode, args, logger)

    def get_parameter_group(self, model, training_config):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": training_config.optimizer_main["weight_decay"],
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return optimizer_grouped_parameters


class AccelerateSaveLoad(SaveLoadBase):
    def __init__(self, accelerator):
        super().__init__()
        self.accelerator = accelerator

    def save(self,
            epoch_index,
            batch_index,
            args,
            options,
            model,
            optimizers,
            best_score,
            batch_count,
            train_global_count,
            folder='epoch_batch',
            reason=None,
            regex_file_pattern=None,
            logger=None):
        str_id = self.make_id(epoch_index, batch_index, folder)
        save_folder = self.get_save_dir(args=args,
                                        model_name=options.aux_model.model_name,
                                        logger=logger,
                                        str_id=str_id)
        # Save configurations
        if self.accelerator.state.local_process_index == 0:
            self.save_config_state(epoch_index=epoch_index,
                                batch_index=batch_index,
                                args=args,
                                options=options,
                                best_score=best_score,
                                batch_count=batch_count,
                                train_global_count=train_global_count,
                                save_folder=save_folder,
                                reason=reason,
                                str_id=str_id,
                                regex_file_pattern=regex_file_pattern,
                                logger=logger)

        # Save model stuff by accelerator. Different GPUs will have different random states.
        self.accelerator.save_state(save_folder)
        logger.info(f"Model is saved to folder {save_folder}")

    def load(self, args, options, model, logger):
        if options.training.train_state.model_id:
            model_path = os.path.join(get_save_load_dir(args, options.aux_model.model_name),
                                        options.training.train_state.model_id)
            try:
                self.accelerator.load_state(model_path)
                logger.info(f"Model initialised from checkpoint @ {model_path}")
            except FileNotFoundError as ex:
                logger.warning(f"Model checkpoint not found @ {model_path}")
                raise
            except IndexError as ex:
                # When trained on multi GPU but test on a single GPU,
                # the random states beyond first GPU will throw this
                # if 'accelerate config' is not reconfigured to single GPU.
                # By this stage, we have load all other states
                # except custom states which we don't have.
                # So, should be ok to preceed.
                logger.warning(ex)
