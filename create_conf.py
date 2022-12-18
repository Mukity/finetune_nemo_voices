import os
import wget

import pytorch_lightning as pl

from nemo.collections.common.callbacks import LogEpochTimeCallback
from nemo.collections.tts.models import FastPitchModel
from nemo.collections.tts.models import HifiGanModel
#from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

from omegaconf import OmegaConf, open_dict

if not os.path.exists('conf'):
    os.mkdir('conf')

if not os.path.isfile('conf/fastpitch_align_v1.05.yaml'):
    wget.download('https://raw.githubusercontent.com/nvidia/NeMo/main/examples/tts/conf/fastpitch_align_v1.05.yaml', out='conf')

if not os.path.isfile('conf/hifigan.yaml'):
    wget.download('https://raw.githubusercontent.com/nvidia/NeMo/main/examples/tts/conf/hifigan/hifigan.yaml', out='conf')

def modify_config(
    train_dataset: str,
    validation_datasets: str,
    phoneme_dict_path: str,
    heteronyms_path: str,
    whitelist_path: str,
    pitch_dict: dict,
    config_name: str,
    sup_data_path: str=None,
    conf_path: str='conf',
    remove_sched: bool=True,
    specgen: bool=True,
    **kwargs: dict
    ):
    if specgen:
        assert sup_data_path, "sup_data_path must be provided for spectrogram generator"

    mdl_params = kwargs.get('model_params', {})
    mdl_train_dataset = kwargs.get('model_train_dataset', {})
    mdl_train_dataloader = kwargs.get('model_train_dataloader', {})
    mdl_val_dataset = kwargs.get('model_val_dataset', {})
    mdl_val_dataloader = kwargs.get('model_val_dataloader', {})
    preprocessor = kwargs.get('preprocessor', {})
    optim = kwargs.get('optim', {})
    trainer = kwargs.get('trainer', {})
    exp_manager = kwargs.get('exp_manager', {})

    generator = {
        '_target_': 'nemo.collections.tts.modules.hifigan_modules.Generator',
        'resblock': 1,
        'upsample_rates': [8, 8, 2, 2],
        'upsample_kernel_sizes': [16, 16, 4, 4],
        'upsample_initial_channel': 512,
        'resblock_kernel_sizes': [3, 7, 11],
        'resblock_dilation_sizes': [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    }
    generator.update(kwargs.get('generator', {}))

    pitch_keys = ['pitch_mean', 'pitch_std', 'pitch_fmin', 'pitch_fmax', 'sample_rate']
    assert all(a in pitch_dict.keys() for a in pitch_keys), f"provide the following keys {pitch_keys}"

    sc = OmegaConf.load(f'{conf_path}/{config_name}')
    with open_dict(sc):
        name = kwargs.get('name')
        if name:
            sc.name = name
        
        sc.train_dataset = train_dataset
        sc.validation_datasets = validation_datasets
        sc.sup_data_path = sup_data_path
        sc.pitch_fmin = pitch_dict['pitch_fmin']
        sc.pitch_fmax = pitch_dict['pitch_fmax']
        sc.pitch_mean = pitch_dict['pitch_mean']
        sc.pitch_std = pitch_dict['pitch_std']
        sc.sample_rate = pitch_dict['sample_rate']

        sc.phoneme_dict_path = phoneme_dict_path
        sc.heteronyms_path = heteronyms_path
        sc.whitelist_path = whitelist_path

        #sc.model.text_normalizer_call_kwargs
        #sc.model.text_normalizer
        #sc.model.text_tokenizer
        sc.symbols_embedding_dim = kwargs.get('symbols_embedding_dim', 384)#try 512 too
        
        for k in ['train_ds', 'validation_ds']:
            if k not in sc.model.keys():
                sc.model[k] = OmegaConf.create({
                    "dataset": OmegaConf.create(),
                    "dataloader_params": OmegaConf.create()
                    })
        
        sc.model.train_ds.dataset.update(mdl_train_dataset)
        sc.model.train_ds.dataloader_params.update(mdl_train_dataloader)
        
        sc.model.validation_ds.dataset.update(mdl_val_dataset)
        sc.model.validation_ds.dataloader_params.update(mdl_val_dataloader)
        sc.model.preprocessor.update(preprocessor)
        sc.model.optim.update(optim)
        sc.trainer.update(trainer)
        sc.exp_manager.update(exp_manager)
        
        if remove_sched:
            sc.model.optim.pop('sched')
        
        sc.model.update(mdl_params)
        if not specgen:
            sc.model.generator = generator

    OmegaConf.save(sc, f"{conf_path}/{config_name.replace('.yaml', '_modified.yaml')}")
    return sc

#@hydra_runner(config_path="conf", config_name="fastpitch_align_44100")
def specgen_main(cfg):
    if hasattr(cfg.model.optim, 'sched'):
        logging.warning("You are using an optimizer scheduler while finetuning. Are you sure this is intended?")
    if cfg.model.optim.lr > 1e-3 or cfg.model.optim.lr < 1e-5:
        logging.warning("The recommended learning rate for finetuning is 2e-4")
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    model = FastPitchModel(cfg=cfg.model, trainer=trainer)
    model.maybe_init_from_pretrained_checkpoint(cfg=cfg)
    lr_logger = pl.callbacks.LearningRateMonitor()
    epoch_time_logger = LogEpochTimeCallback()
    trainer.callbacks.extend([lr_logger, epoch_time_logger])
    trainer.fit(model)

#@hydra_runner(config_path="conf/hifigan", config_name="hifigan_44100")
def vocoder_main(cfg):
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    model = HifiGanModel(cfg=cfg.model, trainer=trainer)
    model.maybe_init_from_pretrained_checkpoint(cfg=cfg)
    trainer.fit(model)


if __name__ == '__main__':
    abc = {
        "model_params": {"a":1},
        "model_train_dataset":{"b":2}, 
        "model_train_dataloader": {"c":3},
        "model_val_dataset": {"d":4},
        "model_val_dataloader": {"e":5},
        "preprocessor": {"f": 6},
        "optim": {"g":7},
        "trainer": {"h":8},
        "exp_manager": {"i":9},
        "generator": {"j":10},
        "name": "jacob",
        "symbols_embedding_dim": 512,
    }
    modify_config(1,2,3,4,5,{"pitch_fmin":0, "pitch_fmax": 0, "pitch_mean": 0, "pitch_std": 1, "sample_rate": 12},
    'hifigan.yaml', sup_data_path='abc', specgen=False, **abc)