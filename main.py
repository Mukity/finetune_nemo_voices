import pytorch_lightning as pl

from nemo.core.config import hydra_runner

from nemo.utils.exp_manager import exp_manager
from nemo.collections.tts.models import FastPitchModel,HifiGanModel
from nemo.collections.common.callbacks import LogEpochTimeCallback

from wavpreprocessing import logger

@hydra_runner(config_path="config", config_name="fastpitch_align_v1.05")
def fastpitch_finetune(cfg):
    if hasattr(cfg.model.optim, 'sched'):
        logger.warning("You are using an optimizer scheduler while finetuning. Are you sure this is intended?")
    if cfg.model.optim.lr > 1e-3 or cfg.model.optim.lr < 1e-5:
        logger.warning("The recommended learning rate for finetuning is 2e-4")
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    model = FastPitchModel(cfg=cfg.model, trainer=trainer)
    model.maybe_init_from_pretrained_checkpoint(cfg=cfg)
    lr_logger = pl.callbacks.LearningRateMonitor()
    epoch_time_logger = LogEpochTimeCallback()
    trainer.callbacks.extend([lr_logger, epoch_time_logger])
    trainer.fit(model)

@hydra_runner(config_path="config", config_name="hifigan")
def hifigan_finetune(cfg):
    if hasattr(cfg.model.optim, 'sched'):
        logger.warning("You are using an optimizer scheduler while finetuning. Are you sure this is intended?")
    if cfg.model.optim.lr > 1e-3 or cfg.model.optim.lr < 1e-5:
        logger.warning("The recommended learning rate for finetuning is 2e-4")
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    model = HifiGanModel(cfg=cfg.model, trainer=trainer)
    model.maybe_init_from_pretrained_checkpoint(cfg=cfg)
    lr_logger = pl.callbacks.LearningRateMonitor()
    epoch_time_logger = LogEpochTimeCallback()
    trainer.callbacks.extend([lr_logger, epoch_time_logger])
    trainer.fit(model)

#TODO: set argparse for fastpitch and hifigan
if __name__=="__main__":
    fastpitch_finetune()
    #example python fastpitch_finetune.py --config_name=<fastpitch_config_name>