import os
import json
import wget
import logging as logger

from nemo.utils.exp_manager import exp_manager
from nemo.collections.common.callbacks import LogEpochTimeCallback
from nemo.collections.tts.models import FastPitchModel

from omegaconf import OmegaConf, open_dict

import pytorch_lightning as pl

if not os.path.exists('conf') and not os.path.exists('tts_dataset_files'):
    os.mkdir('conf')
    os.mkdir('tts_dataset_files')
    wget.download('https://raw.githubusercontent.com/nvidia/NeMo/main/examples/tts/conf/fastpitch_align_v1.05.yaml', out='conf')
    wget.download('https://raw.githubusercontent.com/NVIDIA/NeMo/main/scripts/tts_dataset_files/cmudict-0.7b_nv22.10', out='tts_dataset_files')
    wget.download('https://raw.githubusercontent.com/NVIDIA/NeMo/main/scripts/tts_dataset_files/heteronyms-052722', out='tts_dataset_files')
    wget.download('https://raw.githubusercontent.com/NVIDIA/NeMo/main/nemo_text_processing/text_normalization/en/data/whitelist/lj_speech.tsv', out='tts_dataset_files')

a = OmegaConf.load('conf/fastpitch_align_v1.05.yaml')
with open_dict(a):
    a.init_from_nemo_model='./tts_en_fastpitch_align.nemo'
    a.config_name = 'conf/fastpitch_align_v1.05.yaml'
    a.train_dataset= 'sample/6097_5_mins/manifest_train.json'
    a.validation_datasets='sample/6097_5_mins/manifest_val.json'
    a.sup_data_path='./fastpitch_sup_data'
    a.phoneme_dict_path='tts_dataset_files/cmudict-0.7b_nv22.10'
    a.heteronyms_path='tts_dataset_files/heteronyms-052722'
    a.whitelist_path='tts_dataset_files/lj_speech.tsv'
    a.exp_manager.exp_dir='./ljspeech_to_6097_no_mixing_5_mins'
    a.trainer.max_steps=1000 
    a.trainer.pop('max_epochs')
    a.trainer.check_val_every_n_epoch=25
    a.trainer.devices=1
    a.trainer.strategy=None
    a.model.train_ds.dataloader_params.batch_size=24
    a.model.validation_ds.dataloader_params.batch_size=24
    a.model.n_speakers=1
    a.model.pitch_mean=121.9
    a.model.pitch_std=23.1
    a.model.pitch_fmin=30
    a.model.pitch_fmax=512
    a.model.optim.lr=2e-4
    a.model.optim.pop('sched')
    a.model.optim.name='adam'
    a.model.text_tokenizer.add_blank_at=True

def get_data():
    url='https://nemo-public.s3.us-east-2.amazonaws.com/6097_5_mins.tar.gz'
    out='sample'
    if not os.path.exists(out):
        os.mkdir(out)
    
    if not os.path.exists('sample/6097_5_mins.tar.gz'):
        wget.download(url, out=out)
    audio_c = 'sample/6097_5_mins.tar.gz'
    os.system(f'tar -xvf {audio_c} && mv {audio_c} sample/')
    audio_f = 'sample/6097_5_mins'
    os.remove(audio_c)
    return audio_f

def create_manifest():
    manifest_path='sample/6097_5_mins/manifest.json'
    data = []
    with open(manifest_path) as f:
        for line in f:
            data.append(json.loads(line))
    content =data
    val = content
    train = []
    for i in range(int(len(val)*0.9)):
        train = val.pop()
    save_to_file(manifest_path.replace('.json', '_val.json'), val)
    save_to_file(manifest_path.replace('.json', '_train.json'), train)

def save_to_file(filename, list_):
    with open(filename, 'w') as f:
        for a in list_:
            f.write(json.dumps(a))
            f.write('\n')

def finetuning(cfg=a):
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

if __name__== '__main__':
    get_data()
    create_manifest()
    finetuning()