import os
import sys
import logging

import omegaconf
from omegaconf import OmegaConf

DEBUG_LEVEL = logging.getLevelName(os.environ.get('DEBUG_LEVEL', 'INFO'))
def get_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter("%(asctime)s: %(name)s[%(lineno)s]: %(levelname)s: %(message)s"))
        logger.addHandler(h)
    logger.setLevel(DEBUG_LEVEL)
    logger.propagate = False
    return logger

def load_yaml(yaml_file):
    cfg = OmegaConf.load(yaml_file)
    return cfg


targets_dict = {
    'nemo.collections.tts.data.datalayers.MelAudioDataset': 'nemo.collections.tts.torch.data.VocoderDataset',
}


def modify_targets(cfg):
    for k , v in cfg.items():
        if k == '_target_':
            for a , b in targets_dict.items():
                if v == a:
                    cfg[k] = b
        elif isinstance(v, omegaconf.dictconfig.DictConfig):
            modify_targets(v)
    return cfg

def get_targets(cfg, targets=[]):
    for k , v in cfg.items():
        if k == '_target_':
            targets.append(v)
        elif isinstance(v, omegaconf.dictconfig.DictConfig):
            get_targets(v, targets)
    return targets