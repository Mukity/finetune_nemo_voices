import os
import wget

import  omegaconf
from omegaconf import OmegaConf, open_dict
from nemo.collections.tts.models.base import SpectrogramGenerator, Vocoder

from wavpreprocessing import logger

specgen_models = []
voc_models = []
for model in SpectrogramGenerator.list_available_models():
    specgen_models.append(model.pretrained_model_name)

for model in Vocoder.list_available_models():
    voc_models.append(model.pretrained_model_name)

BASE_DATASET_FILES_DIR = "tts_dataset_files"
phoneme_dict_path = f"{BASE_DATASET_FILES_DIR}/cmudict-0.7b_nv22.10"
heteronyms_path = f"{BASE_DATASET_FILES_DIR}/heteronyms-052722"
whitelist_path = f"{BASE_DATASET_FILES_DIR}/lj_speech.tsv"

"""
base keys:
    Required:
        pitch_fmin, pitch_fmax, pitch_std, pitch_mean
        train_dataset, validation_datasets, sup_data_path, sample_rate
    optional:
        n_mel_channels, n_window_size, n_window_stride, n_fft, lowfreq, highfreq
        window, sup_data_types, name

model:
    learn_alignment, bin_loss_warmup_epochs, n_speakers, max_token_duration
    symbols_embedding_dim, pitch_embedding_kernel_size

trainer:
exp_manager:
train_dataset:
train_dataloader:
val_dataset:
val_dataloader:

model_kwargs:
    text_normalizer, text_normalizer_call_kwargs, text_tokenizer, preprocessor,
    input_fft, output_fft, alignment_module, duration_predictor, pitch_predictor,
    optim,
"""
def change_configuration(
    audio_folder_name: str,
    base: dict,
    config_path: str="",
    init_from: str="",
    model={},
    trainer={},
    exp_manager={},
    train_dataset={},
    train_dataloader={},
    val_dataset={},
    val_dataloader={},
    specgen=True,
    **model_kwargs
    ):

    if specgen:
        type_ = "SpectrogramGenerator"
    else:
        type_ = "Vocoder"
    base_keys = ["train_dataset", "validation_datasets", "sup_data_path", "sample_rate"]

    if not config_path:
        config_path = setup_configs(specgen)

    cfg = OmegaConf.load(config_path)
    with open_dict(cfg):
        if init_from.endswith('.nemo'):
            base['init_from_nemo_model'] = init_from
        elif init_from.endswith('.ckpt'):
            base['init_from_ptl_ckpt'] = init_from
        else:
            if (specgen and init_from in specgen_models) or (not specgen and init_from in voc_models):
                base['init_from_pretrained_model'] = init_from
            else:
                if not init_from:
                    if specgen:
                        base['init_from_pretrained_model'] = 'tts_en_fastpitch'
                    else:
                        base['init_from_pretrained_model'] = 'tts_hifigan'
                else:
                    raise ValueError(f"{init_from} model is not in list of {type_}")
        
        if specgen:
            base_keys.extend(["pitch_fmin", "pitch_fmax", "pitch_std", "pitch_mean"])
        
        missing_keys = {k for k in base_keys if k not in base}
        assert not missing_keys, f"{type_} missing the following keys {missing_keys}"

        if "phoneme_dict_path" not in base:
            base["phoneme_dict_path"] = phoneme_dict_path
        if "heteronyms_path" not in base:
            base["heteronyms_path"] = heteronyms_path
        if "whitelist_path" not in base:
            base["whitelist_path"] = whitelist_path
        
        if "exp_dir" not in exp_manager:
            exp_manager['exp_dir'] = f"models/finetuned"
        if 'max_epochs' not in cfg.trainer:
            cfg.trainer['max_epochs'] = 1000

        cfg.update(base)
        cfg.trainer.update(trainer)
        cfg.exp_manager.update(exp_manager)
        cfg.model.update(model)

        try:
            cfg.model.train_ds.dataset.update(train_dataset)
            cfg.model.train_ds.dataloader_params.update(train_dataloader)
            cfg.model.validation_ds.dataset.update(val_dataset)
            cfg.model.validation_ds.dataloader_params.update(val_dataloader)

        except omegaconf.errors.ConfigAttributeError:
            if specgen:
                default_cfg = SpectrogramGenerator.from_pretrained('tts_en_fastpitch').cfg
            else:
                default_cfg = Vocoder.from_pretrained('tts_hifigan').cfg
            
            cfg.model.train_ds = default_cfg.train_ds
            cfg.model.validation_ds = default_cfg.validation_ds
            pass

        for model_attribute, v in model_kwargs.items():
            for attr, b in v.items():
                cfg.model[model_attribute][attr] = b
        
        if specgen:
            cfg.model.text_tokenizer.add_blank_at=True

        if "strategy" in cfg.trainer:
            cfg.trainer.pop('strategy')
        if "sched" in cfg.model.optim:
            cfg.model.optim.pop('sched')
    save_dir = config_path.replace('.yaml', f'_{audio_folder_name}.yaml')
    OmegaConf.save(cfg, save_dir)
    logger.info(f"{save_dir} created from {config_path}")
    return cfg

def setup_configs(specgen=True):
    if not os.path.exists("config"):
        os.mkdir("config")
    if specgen:
        yaml_path = "config/fastpitch_align_v1.05.yaml"
        if not os.path.isfile(yaml_path):
            wget.download('https://raw.githubusercontent.com/nvidia/NeMo/main/examples/tts/conf/fastpitch_align_v1.05.yaml',\
                 out='config')
    else:
        yaml_path = "config/hifigan.yaml"
        if not os.path.isfile(yaml_path):
            wget.download('https://raw.githubusercontent.com/nvidia/NeMo/main/examples/tts/conf/hifigan/hifigan.yaml',\
                 out='config')
    return yaml_path