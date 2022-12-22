import os
import wget

from nemo.collections.tts.models import FastPitchModel
from nemo.collections.tts.models import HifiGanModel

from omegaconf import OmegaConf, open_dict
from wavpreprocessing import logger

dataset_config_folder = "dataset_config_files"
if not os.path.exists(dataset_config_folder):
    os.mkdir(dataset_config_folder)

default_heteronyms = f'{dataset_config_folder}/heteronyms-052722'
if not os.path.isfile(default_heteronyms):
    wget.download('https://raw.githubusercontent.com/NVIDIA/NeMo/main/scripts/tts_dataset_files/heteronyms-052722',
        out=dataset_config_folder)

default_phoneme_dict = f'{dataset_config_folder}/cmudict-0.7b_nv22.10'
if not os.path.isfile(default_phoneme_dict):
    wget.download('https://raw.githubusercontent.com/NVIDIA/NeMo/main/scripts/tts_dataset_files/cmudict-0.7b_nv22.10',
     out=dataset_config_folder)

default_whitelists = f'{dataset_config_folder}/lj_speech.tsv'
if not os.path.isfile(default_whitelists):
    wget.download('https://raw.githubusercontent.com/NVIDIA/NeMo/main/nemo_text_processing/text_normalization/en/data/whitelist/lj_speech.tsv',
     out=dataset_config_folder)

"""
NOTE: all kwargs parameters require dicts
kwargs params:
    model_params
    model_train_dataset
    model_train_dataloader
    model_val_dataset
    model_val_dataloader
    preprocessor
    optim
    trainer
    exp_manager
    generator
    name
    symbols_embedding_dim
"""
def modify_config(
    train_dataset: str,
    validation_datasets: str,
    pitch_dict: dict,
    config_name: str=None,
    phoneme_dict_path: str=None,
    heteronyms_path: str=None,
    whitelist_path: str=None,
    config_path: str=None,
    specgen: bool=True,
    init_nemo: str=None,
    init_pretrained: str=None,
    init_checkpoint: str=None,
    **kwargs: dict
    ):
    for k, v in kwargs.items():
        if v:
            logger.info(f"found {k} with values {v}")
    if specgen:
        config_name = config_name or 'fastpitch_align_v1.05.yaml'
        pitch_keys = [
            'pitch_mean', 'pitch_std', 'pitch_fmin', 'pitch_fmax',
            'sample_rate', 'sup_data_path', 'min_duration', 'max_duration'
        ]
        phoneme_dict_path = phoneme_dict_path or default_phoneme_dict
        heteronyms_path = heteronyms_path or default_heteronyms
        whitelist_path = whitelist_path or default_whitelists
    else:
        config_name = config_name or 'hifigan.yaml'
        pitch_keys = ['sample_rate',]

    assert all(a in pitch_dict.keys() for a in pitch_keys), f"provide the following keys {pitch_keys}"        

    config_path = config_path or 'config'
    mdl_params = kwargs.get('model_params', {})  or {}
    mdl_train_dataset = kwargs.get('model_train_dataset', {})  or {}
    mdl_train_dataloader = kwargs.get('model_train_dataloader', {})  or {}
    mdl_val_dataset = kwargs.get('model_val_dataset', {})  or {}
    mdl_val_dataloader = kwargs.get('model_val_dataloader', {})  or {}
    preprocessor = kwargs.get('preprocessor', {})  or {}
    optim = kwargs.get('optim', {})  or {}
    trainer = kwargs.get('trainer', {})  or {}
    exp_manager = kwargs.get('exp_manager', {})  or {}

    for k in ['min_duration' , 'max_duration']:
        mdl_train_dataset[k] = pitch_dict[k]
        mdl_val_dataset[k] = pitch_dict[k]
    
    if not specgen:
        generator = {
            '_target_': 'nemo.collections.tts.modules.hifigan_modules.Generator',
            'resblock': 1,
            'upsample_rates': [8, 8, 2, 2],
            'upsample_kernel_sizes': [16, 16, 4, 4],
            'upsample_initial_channel': 512,
            'resblock_kernel_sizes': [3, 7, 11],
            'resblock_dilation_sizes': [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        }
        generator.update(kwargs.get('generator', {}) or {})

    sc = OmegaConf.load(f'{config_path}/{config_name}')
    mdl = None
    with open_dict(sc):
        name = kwargs.get('name')
        if name:
            sc.name = name
        
        sc.train_dataset = train_dataset
        sc.validation_datasets = validation_datasets
        if specgen:
            sc.sup_data_path = pitch_dict['sup_data_path']
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
                if not mdl:
                    if specgen:
                        mdl = FastPitchModel.from_pretrained('tts_en_fastpitch')
                    else:
                        mdl = HifiGanModel.from_pretrained('tts_hifigan')
                sc.model[k] = OmegaConf.create(mdl.cfg[k])
        
        if not specgen:
            sc.model.train_ds.dataset.manifest_filepath = train_dataset
            sc.model.validation_ds.dataset.manifest_filepath = validation_datasets
            sc.model.train_ds.dataset._target_ = "nemo.collections.tts.torch.data.VocoderDataset"
            sc.model.validation_ds.dataset._target_ = "nemo.collections.tts.torch.data.VocoderDataset"
            sc.model.train_ds.dataset.sample_rate = pitch_dict['sample_rate']
            sc.model.validation_ds.dataset.sample_rate = pitch_dict['sample_rate']
            
        
        sc.model.train_ds.dataset.update(mdl_train_dataset)
        sc.model.train_ds.dataloader_params.update(mdl_train_dataloader)
        
        sc.model.validation_ds.dataset.update(mdl_val_dataset)
        sc.model.validation_ds.dataloader_params.update(mdl_val_dataloader)
        sc.model.preprocessor.update(preprocessor)
        sc.model.optim.update(optim)
        sc.trainer.update(trainer)
        
        sc.trainer.pop('strategy')
        if 'max_epochs' not in sc.trainer:
            sc.trainer['max_epochs'] = 1000

        if exp_manager.get('exp_dir') is None:
            exp_manager['exp_dir'] = f"models/finetuned"
        sc.exp_manager.update(exp_manager)
        
        sc.model.optim.pop('sched')
        
        sc.model.update(mdl_params)
        if not specgen:
            sc.model.generator = generator
        
        if init_nemo:
            sc.init_from_nemo_model = init_nemo
        elif init_pretrained:
            sc.init_from_pretrained_model = init_pretrained
        elif init_checkpoint:
            sc.init_from_ptl_ckpt - init_checkpoint
        else:
            if specgen:
                sc.init_from_pretrained_model = 'tts_en_fastpitch'
            else:
                sc.init_from_pretrained_model = 'tts_hifigan'

    OmegaConf.save(sc, f"{config_path}/{config_name.replace('.yaml', '_modified.yaml')}")
    return sc