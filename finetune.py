import os
import shutil

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

def move_last_checkpoint(cfg, prefix=None):
    base_dir = f"{cfg.exp_manager.exp_dir}/{cfg.name}"
    base_dir_files = sorted(os.listdir(base_dir), reverse=True)
    for time in base_dir_files:
        if os.path.isdir(f"{base_dir}/{time}"):
           break
    time_dir = f"{base_dir}/{time}"
    checkpoint_dir = f"{time_dir}/checkpoints"
    try:
        last_ckpt = [a for a in os.listdir(checkpoint_dir) if a.endswith('-last.ckpt')][0]
    except NotADirectoryError:
        logger.warning(f"{time_dir} is not a directory")
        return
    ckpt_path = f"{checkpoint_dir}/{last_ckpt}"
    if prefix:
        dest = f'{base_dir}/{prefix}_{last_ckpt}'
    else:
        dest = f'{base_dir}/{time}_{last_ckpt}'
    shutil.move(ckpt_path, dest)
    shutil.rmtree(time_dir)
    logger.info(f"last checkpoint moved from {ckpt_path} to {dest}")


def main():
    parser = argparse.ArgumentParser(description="Finetuning options")
    parser.add_argument("-config_name", type=str, required=True)
    #parser.add_argument("-audio_folder", type=str, required=True)
    parser.add_argument("-config_folder", type=str, default="config")
    parser.add_argument("-mode", type=str, choices=['specgen', 'vocoder'], required=True)
    parser.add_argument("-move_last_ckpt", type=bool, default=True)
    parser.add_argument("-checkpoint_prefix", type=str)
    args = parser.parse_args()

    conf_name = args.config_name
    if not conf_name.endswith(".yaml"):
        conf_path = f"{args.config_folder}/{args.config_name}.yaml"
    else:
        conf_path = f"{args.config_folder}/{args.config_name}"
    conf_cfg = load_yaml(conf_path)

    if args.mode == 'specgen':
        logger.info("finetuning the fastpitch model")
        fastpitch_finetune(conf_cfg)
    else:
        logger.info("finetuning the hifigan model")
        hifigan_finetune(conf_cfg)

    if args.move_last_ckpt:
        move_last_checkpoint(conf_cfg, args.checkpoint_prefix)

if __name__=="__main__":
    main()
