import pytorch_lightning as pl

from nemo.utils.exp_manager import exp_manager
from nemo.collections.tts.models import FastPitchModel,HifiGanModel
from nemo.collections.common.callbacks import LogEpochTimeCallback

from helpers import get_logger, load_yaml
logger = get_logger(__file__.split('.')[0])

import argparse


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

def hifigan_finetune(cfg):
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    model = HifiGanModel(cfg=cfg.model, trainer=trainer)
    model.maybe_init_from_pretrained_checkpoint(cfg=cfg)
    trainer.fit(model)

def main():
    parser = argparse.ArgumentParser(description="Finetuning options")
    parser.add_argument("-config_folder", type=str, default="config")
    parser.add_argument("-config_name", type=str, required=True)
    parser.add_argument("-mode", type=str, choices=['specgen', 'vocoder'], required=True)
    args = parser.parse_args()

    conf_path = f"{args.config_folder}/{args.config_name}.yaml"
    conf_cfg = load_yaml(conf_path)

    if args.mode == 'specgen':
        logger.info("finetuning the fastpitch model")
        fastpitch_finetune(conf_cfg)
    else:
        logger.info("finetuning the hifigan model")
        hifigan_finetune(conf_cfg)

if __name__=="__main__":
    main()