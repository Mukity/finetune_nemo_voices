import os
import json
import wget

import librosa
import argparse

import torch
import pytorch_lightning as pl

from tqdm.notebook import tqdm

from nemo.utils.exp_manager import exp_manager
from nemo.collections.tts.models import FastPitchModel
from nemo.collections.tts.torch.data import TTSDataset
from nemo.collections.common.callbacks import LogEpochTimeCallback
from nemo_text_processing.text_normalization.normalize import Normalizer
from nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers import EnglishCharsTokenizer

from wavpreprocessing import logger
from modify_conf import modify_config, default_whitelists

if not os.path.exists('config'):
    os.mkdir('config')

if not os.path.isfile('config/fastpitch_align_v1.05.yaml'):
    wget.download('https://raw.githubusercontent.com/nvidia/NeMo/main/examples/tts/conf/fastpitch_align_v1.05.yaml', out='config')

whitelist = default_whitelists

class SpectroGramConfigPreprocessing:
    def __init__(
        self,
        audio_folder: str,
        manifest_name: str,
        manifest_folder: str=None,
        **kwargs,
        ):
        self.normalizer = {
            "input_case": 'cased',#['cased', 'lowercased']
            "lang": 'en',
            "whitelist": whitelist,
        }
        self.normalizer.update(kwargs.get('normalizer'))
        text_normalizer = Normalizer(
            input_case=self.normalizer['input_case'],
            lang=self.normalizer['lang'],
            whitelist=self.normalizer['whitelist'],
        )
        self.text_normalizer_call_kwargs = {
            "verbose": False,
            "punct_pre_process": True,
            "punct_post_process": True
        }
        self.text_normalizer_call_kwargs.update(kwargs.get('normalizer_kwargs'))

        text_tokenizer = EnglishCharsTokenizer()

        self.audio_folder = audio_folder
        self.manifest_folder = manifest_folder or 'manifests'
        self.manifest_name = manifest_name
        self.text_tokenizer = text_tokenizer
        self.text_normalizer = text_normalizer
        self.text_normalizer_call_kwargs = self.text_normalizer_call_kwargs
        self.out = {}
        if not os.path.exists(self.manifest_folder):
            os.mkdir(self.manifest_folder)

    
    def create_manifest(self, manifest_name: str=None, audio_folder: str=None, manifest_folder: str=None, librispeech: bool=False, speaker_id: str=None):
        audio_folder = audio_folder or self.audio_folder
        manifest_name = manifest_name or self.manifest_name
        manifest_folder = manifest_folder or self.manifest_folder

        audio_dicts = []
        sr = []
        min_duration = 1000
        max_duration = 0
        all_files = os.listdir(audio_folder)
        for a in all_files:
            if (not librispeech and a.endswith('.wav')) or (librispeech and a.endswith('.wav') and a.startswith(speaker_id)):
                fpath = f"{audio_folder}/{a}"
                duration = librosa.get_duration(filename=fpath)
                if librispeech:
                    txt = {
                        "text": ".original.txt",
                        "normalized_text": ".normalized.txt",
                    }
                else:
                    txt = {
                        "text": ".txt"
                    }

                ad = {
                    "audio_filepath": fpath,
                    "duration": duration,
                }
                for k, ext in txt.items():
                    with open(fpath.replace('.wav', ext)) as f:
                        ad[k] = f.read()

                if min_duration > duration:
                    min_duration = duration
                if max_duration < duration:
                    max_duration = duration

                sr.append(librosa.get_samplerate(fpath))
                audio_dicts.append(ad)

        audio_dicts_val = audio_dicts.copy()
        audio_dicts_train = []
        for i in range(int(len(audio_dicts)*0.9)):
            audio_dicts_train.append(audio_dicts_val.pop())

        self.out["sample_rate"] = int(sum(sr)/len(sr))
        self.out["max_duration"] = max_duration
        self.out["min_duration"] = min_duration

        extensions = ['.json', '_val.json', '_train.json']  
        for i, ext in enumerate(extensions):
            mpath = f"{manifest_folder}/{manifest_name}{ext}"
            if i == 0:
                dicts = audio_dicts
            elif i == 1:
                dicts = audio_dicts_val
                self.out['val_dataset'] = mpath
            else:
                dicts = audio_dicts_train
                self.out['train_dataset'] = mpath
            with open(mpath, 'w') as f:
                for d in dicts:
                    f.write(json.dumps(d))
                    f.write('\n')
        return self.out


    def pre_calculate_supplementary_data(self, sup_data_path: str, sample_rate: int=None, delete_sup_folder: bool=True, **kwargs):
        sample_rate = self.out.get('sample_rate') or sample_rate
        assert sample_rate, "provide sample_rate to calculate supplementary data"

        pitch_file = kwargs.pop('pitch_file', 'pitch_file.json')
        manifest_folder=kwargs.pop('manifest_folder', self.manifest_folder)
        manifest_filename = kwargs.pop('manifest_filename', self.manifest_name)
    
        stages = ["train", "val"]
        stage2dl = {}
        for stage in stages:
            if stage == "train":
                fname = f'{manifest_filename}_train.json'
            else:
                fname = f'{manifest_filename}_val.json'
            ds = TTSDataset(
                manifest_filepath=f"{manifest_folder}/{fname}",
                sample_rate=sample_rate,
                sup_data_path=sup_data_path,
                sup_data_types=kwargs.get('sup_data_types', ["align_prior_matrix", "pitch"]),
                n_fft=1024,
                win_length=1024,
                hop_length=256,
                window=kwargs.get('window', "hann"),
                n_mels=80,
                lowfreq=0,
                highfreq=8000,
                text_tokenizer=kwargs.get('text_tokenizer', self.text_tokenizer),
                text_normalizer=kwargs.get('text_normalizer', self.text_normalizer),
                text_normalizer_call_kwargs=kwargs.get('text_normalizer_call_kwargs', self.text_normalizer_call_kwargs)
            ) 
            stage2dl[stage] = torch.utils.data.DataLoader(ds, batch_size=1, collate_fn=ds._collate_fn, num_workers=1)
    
        pitch_mean, pitch_std, pitch_min, pitch_max = None, None, None, None
        for stage, dl in stage2dl.items():
            pitch_list = []
            for batch in tqdm(dl, total=len(dl)):
                tokens, tokens_lengths, audios, audio_lengths, attn_prior, pitches, pitches_lengths = batch
                pitch = pitches.squeeze(0)
                pitch_list.append(pitch[pitch != 0])
    
            if stage == "train":
                pitch_tensor = torch.cat(pitch_list)
                pitch_mean, pitch_std = pitch_tensor.mean().item(), pitch_tensor.std().item()
                pitch_min, pitch_max = pitch_tensor.min().item(), pitch_tensor.max().item()
        
        if delete_sup_folder:
            os.system(f'rm -rf {sup_data_path}')
        
        self.out['sup_data_path'] = sup_data_path
        self.out['pitch_mean'] = pitch_mean
        self.out['pitch_std'] = pitch_std
        self.out['pitch_fmin'] = pitch_min
        self.out['pitch_fmax'] = pitch_max
        with open(pitch_file, 'w') as f:
            json.dump(self.out, f, indent=4)
        return self.out

def finetuning(cfg):
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

def run(
    audio_folder,
    manifest_name,
    manifest_folder,
    sup_data_path,
    config_path,
    config_name,
    phoneme_dict_path,
    heteronyms_path,
    init_nemo,
    init_pretrained,
    init_checkpoint,
    librispeech,
    speaker_id,
    **kwargs
    ):
    norm_keys = ['normalizer', 'normalizer_kwargs']
    norm = {}
    for k in norm_keys:
        norm[k] = kwargs.pop(k) or {}
    
    sgp = SpectroGramConfigPreprocessing(audio_folder, manifest_name, manifest_folder, **norm)
    sgp.create_manifest(librispeech=librispeech, speaker_id=speaker_id)
    sgp.pre_calculate_supplementary_data(sup_data_path)
    logger.info(json.dumps(sgp.out))

    train_dataset = sgp.out.pop('train_dataset')
    validation_datasets = sgp.out.pop('val_dataset')
    pitch_dict = sgp.out
    whitelist_path = sgp.normalizer['whitelist']

    cfg = modify_config(train_dataset, validation_datasets, pitch_dict, config_name,
            phoneme_dict_path, heteronyms_path, whitelist_path, config_path,
            init_nemo=init_nemo, init_checkpoint=init_checkpoint, init_pretrained=init_pretrained,
              **kwargs)
    finetuning(cfg)

def argparser():
    parser = argparse.ArgumentParser(
        prog="spectrogram preprocessing",
        description="spectrogram generator preprocessor",
    )
    parser.add_argument('-audio_folder', type=str, required=True)
    parser.add_argument('-manifest_name', type=str, required=True)
    parser.add_argument('-sup_data_path', type=str, required=True)
    parser.add_argument('-manifest_folder', type=str)
    parser.add_argument('-config_path', type=str)
    parser.add_argument('-config_name', type=str)
    parser.add_argument('-phoneme_dict_path', type=str)
    parser.add_argument('-heteronyms_path', type=str)
    parser.add_argument('-normalizer', type=json.loads, help='dict should be written as {\"key1\":\"value1\"}')
    parser.add_argument('-normalizer_kwargs', type=json.loads, help='dict should be written as {\"key1\":\"value1\"}')
    parser.add_argument('-model_params', type=json.loads)
    parser.add_argument('-model_train_dataset', type=json.loads)
    parser.add_argument('-model_train_dataloader', type=json.loads)
    parser.add_argument('-model_val_dataset', type=json.loads)
    parser.add_argument('-model_val_dataloader', type=json.loads)
    parser.add_argument('-preprocessor', type=json.loads)
    parser.add_argument('-optim', type=json.loads)
    parser.add_argument('-trainer', type=json.loads)
    parser.add_argument('-exp_manager', type=json.loads)
    parser.add_argument('-generator', type=json.loads)
    parser.add_argument('-name', type=json.loads)
    parser.add_argument('-symbols_embedding_dim', type=json.loads)
    parser.add_argument('-init_nemo', type=str)
    parser.add_argument('-init_pretrained', type=str)
    parser.add_argument('-init_checkpoint', type=str)
    parser.add_argument('-librispeech', type=bool, default=False)
    parser.add_argument('-speaker_id', type=str)
    args = parser.parse_args()
    if args.librispeech and not args.speaker_id:
        parser.error('librispeech requires -speaker_id value')
    return args

def main():
    args = argparser()
    audio_folder = args.audio_folder
    manifest_name = args.manifest_name
    manifest_folder = args.manifest_folder
    config_path = args.config_path
    config_name = args.config_name
    phoneme_dict_path = args.phoneme_dict_path
    heteronyms_path = args.heteronyms_path
    init_nemo = args.init_nemo
    init_pretrained = args.init_pretrained
    init_checkpoint = args.init_checkpoint

    sup_data_path = args.sup_data_path
    librispeech = args.librispeech
    speaker_id = args.speaker_id
    
    kwargs = {
        "normalizer": args.normalizer or {},
        "normalizer_kwargs": args.normalizer_kwargs or {},
        "model_params": args.model_params,
        "model_train_dataset": args.model_train_dataset,
        "model_train_dataloader": args.model_train_dataloader,
        "model_val_dataset": args.model_val_dataset,
        "model_val_dataloader": args.model_val_dataloader,
        "preprocessor": args.preprocessor,
        "optim": args.optim,
        "trainer": args.trainer,
        "exp_manager": args.exp_manager,
        "generator": args.generator,
        "name": args.name,
        "symbols_embedding_dim": args.symbols_embedding_dim,
    }
    run(audio_folder, manifest_name, manifest_folder, sup_data_path, config_path, config_name,
        phoneme_dict_path, heteronyms_path, init_nemo, init_pretrained, init_checkpoint, librispeech, speaker_id, **kwargs)

if __name__ == "__main__":
    main()
    """
    python spectrogram.py -manifest_name vd_manifest -sup_data_path vd_sup_data -init_pretrained tts_en_fastpitch -trainer '{"max_steps":1000,"check_val_every_n_epoch":25,"devices":1,"strategy":null}' -model_train_dataloader '{"batch_size":24}' -model_val_dataloader '{"batch_size":24}' -optim '{"name":"adam"}' -audio_folder audios/VD
    """