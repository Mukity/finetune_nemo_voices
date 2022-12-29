import os
import sys
import logging

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