import os
import wget

from omegaconf import OmegaConf, open_dict
import preprocessing

logger = preprocessing.logger

if not os.path.exists('conf'):
    os.mkdir('conf')

if not os.path.isfile('conf/fastpitch_align_v1.05.yaml'):
    wget.download('https://raw.githubusercontent.com/nvidia/NeMo/main/examples/tts/conf/fastpitch_align_v1.05.yaml', out='conf')

if not os.path.isfile('tts_dataset_files/cmudict-0.7b_nv22.10'):
    wget.download(f'https://raw.githubusercontent.com/NVIDIA/NeMo/main/scripts/tts_dataset_files/cmudict-0.7b_nv22.10', out='tts_dataset_files')

if not os.path.isfile('tts_dataset_files/heteronyms-052722'):
    wget.download(f'https://raw.githubusercontent.com/NVIDIA/NeMo/main/scripts/tts_dataset_files/heteronyms-052722', out='tts_dataset_files')

def update_fastpitch(
    preprocessing_dict=preprocessing.run_vd('vd_manifest.json', f'vd_sup_data'),
    phoneme_dict_path='tts_dataset_files/cmudict-0.7b_nv22.10',
    heteronyms_path='tts_dataset_files/heteronyms-052722',
    whitelist_path = preprocessing.normalizer['whitelist'],
    lr = 2e-4 or 1e-3,
    betas = [0.9, 0.999],
    weight_decay = 1e-6,
    optim_name = 'adam',
    temperature = 0.0005,
    max_steps = 1000,
    check_val_every_n_epoch = 25,
    shuffle = True,
    trim = False,
    batch_size = 32,
    num_workers = 12,
    bin_loss_warmup_epochs = 100,
    symbols_embedding_dim = 384,
    conf_filename="conf/fastpitch_align_v1.05.yaml",
    defaults=False,
    save_to_file=True,
    ):
    preprocessing_keys = [
        "train_dataset",
        "validation_datasets",
        "sup_data_path",
        "sample_rate",
        "pitch_fmin",
        "pitch_fmax",
        "pitch_std",
        "pitch_mean",
        "max_duration",
        "min_duration"
    ]
    missing_keys = ""
    for k in preprocessing_keys:
        if k not in preprocessing_dict:
            missing_keys += f"{k}, "

    if missing_keys != "":
        raise KeyError(f"These keys are missing: {missing_keys}")

    fastpitch_cfg = OmegaConf.load(conf_filename)
    with open_dict(fastpitch_cfg):
        fastpitch_cfg.train_dataset = preprocessing_dict['train_dataset']
        fastpitch_cfg.validation_datasets = preprocessing_dict['validation_datasets']
        fastpitch_cfg.sup_data_path = preprocessing_dict['sup_data_path']
        fastpitch_cfg.pitch_fmin = preprocessing_dict['pitch_fmin']
        fastpitch_cfg.pitch_fmax = preprocessing_dict['pitch_fmax']
        fastpitch_cfg.pitch_std = preprocessing_dict['pitch_std']
        fastpitch_cfg.pitch_mean = preprocessing_dict['pitch_mean']
        fastpitch_cfg.sample_rate = preprocessing_dict['sample_rate']
        #configure the following values to have selection between None values and specific values
        fastpitch_cfg.n_mel_channels= 80
        fastpitch_cfg.n_window_size= 1024
        fastpitch_cfg.n_window_stride= 256
        fastpitch_cfg.highfreq= 8000

        fastpitch_cfg.n_fft= 1024
        fastpitch_cfg.lowfreq= 0
        fastpitch_cfg.window= 'hann'

        fastpitch_cfg.phoneme_dict_path = phoneme_dict_path
        fastpitch_cfg.heteronyms_path = heteronyms_path
        fastpitch_cfg.whitelist_path = whitelist_path

        fastpitch_cfg.model.bin_loss_warmup_epochs = bin_loss_warmup_epochs
        fastpitch_cfg.model.symbols_embedding_dim = symbols_embedding_dim
        
        fastpitch_cfg.model.text_normalizer.input_case = preprocessing.normalizer['input_case']
        fastpitch_cfg.model.text_normalizer.lang = preprocessing.normalizer['lang']
        fastpitch_cfg.model.text_normalizer.whitelist = whitelist_path

        fastpitch_cfg.model.text_normalizer_call_kwargs.verbose = preprocessing.text_normalizer_call_kwargs['verbose']
        fastpitch_cfg.model.text_normalizer_call_kwargs.punct_pre_process = preprocessing.text_normalizer_call_kwargs['punct_pre_process']
        fastpitch_cfg.model.text_normalizer_call_kwargs.punct_post_process = preprocessing.text_normalizer_call_kwargs['punct_post_process']
        #fastpitch_cfg.model.text_tokenizer = preprocessing.text_tokenizer
        fastpitch_cfg.model.text_tokenizer.add_blank_at = True

        fastpitch_cfg.model.train_ds.dataset.max_duration = preprocessing_dict['max_duration']
        fastpitch_cfg.model.train_ds.dataset.min_duration = preprocessing_dict['min_duration']
        fastpitch_cfg.model.train_ds.dataset.trim = trim
        fastpitch_cfg.model.train_ds.dataloader_params.shuffle = shuffle
        fastpitch_cfg.model.train_ds.dataloader_params.batch_size = batch_size 
        fastpitch_cfg.model.train_ds.dataloader_params.num_workers = num_workers
        fastpitch_cfg.model.validation_ds.dataset.max_duration = preprocessing_dict['max_duration']
        fastpitch_cfg.model.validation_ds.dataset.min_duration = preprocessing_dict['min_duration']
        fastpitch_cfg.model.validation_ds.dataset.trim = trim
        fastpitch_cfg.model.validation_ds.dataloader_params.shuffle = shuffle
        fastpitch_cfg.model.validation_ds.dataloader_params.batch_size = batch_size
        fastpitch_cfg.model.validation_ds.dataloader_params.num_workers = num_workers

        fastpitch_cfg.model.alignment_module.temperature = temperature

        fastpitch_cfg.model.optim.name = optim_name
        fastpitch_cfg.model.optim.lr = lr
        fastpitch_cfg.model.optim.betas= betas
        fastpitch_cfg.model.optim.weight_decay= weight_decay
        fastpitch_cfg.model.optim.pop('sched')

        #exp_manager.exp_dir=./ljspeech_to_6097_no_mixing_5_mins
        fastpitch_cfg.trainer.auto_lr_find = True
        fastpitch_cfg.trainer.auto_scale_batch_size = True
        fastpitch_cfg.trainer.num_sanity_val_steps = -1
        fastpitch_cfg.trainer.max_steps = max_steps
        fastpitch_cfg.trainer.pop('max_epochs')
        fastpitch_cfg.trainer.check_val_every_n_epoch = check_val_every_n_epoch
        fastpitch_cfg.trainer.devices = 1
        fastpitch_cfg.trainer.strategy = None if fastpitch_cfg.trainer.devices >1 else fastpitch_cfg.trainer.strategy

        if defaults:
            fastpitch_cfg.trainer.num_sanity_val_steps = 2
            fastpitch_cfg.model.optim.name = 'adamw',
            fastpitch_cfg.model.symbols_embedding_dim = 512
            fastpitch_cfg.model.train_ds.dataloader_params.batch_size = 24
            fastpitch_cfg.n_window_size = None
            fastpitch_cfg.n_window_stride = None
            fastpitch_cfg.highfreq = None
        
        if save_to_file:
            if defaults:
                fname = conf_filename.replace('.yaml', '_modified_defaults.yaml')
            else:
                fname = conf_filename.replace('.yaml', '_modified.yaml')
        OmegaConf.save(fastpitch_cfg, fname)
    logger.info(f'file path is: {fname}')
    return fname


if __name__ == '__main__':
    a = {'sample_rate': 48000.0, 'pitch_mean': 173.52857971191406, 'pitch_std': 39.51873016357422, 'pitch_fmin': 94.1154556274414, 'pitch_fmax': 323.9647521972656, 'min_duration': 0.48, 'max_duration': 7.44, 'sup_data_path': 'vd_sup_data', 'train_dataset': 'vd_manifest_train.json', 'validation_datasets': 'vd_manifest_val.json'}
    update_fastpitch(preprocessing_dict=a, defaults=True)