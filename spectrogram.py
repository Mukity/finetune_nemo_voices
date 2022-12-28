import os
import wget
import json
import shutil

import torch
import librosa

from nemo.collections.tts.torch.data import TTSDataset
from nemo_text_processing.text_normalization.normalize import Normalizer
from nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers import EnglishCharsTokenizer

#from tqdm.notebook import tqdm
from tqdm import tqdm

from wavpreprocessing import logger
from modify_config import change_configuration

dataset_files = 'tts_dataset_files'
if not os.path.exists(dataset_files):
    os.mkdir(dataset_files)

whitelist = f'{dataset_files}/lj_speech.tsv'
if not os.path.isfile(whitelist):
    wget.download('https://raw.githubusercontent.com/NVIDIA/NeMo/main/nemo_text_processing/text_normalization/en/data/whitelist/lj_speech.tsv',
     out=dataset_files)

if not os.path.exists(f'{dataset_files}/cmudict-0.7b_nv22.10'):
    wget.download('https://raw.githubusercontent.com/NVIDIA/NeMo/main/scripts/tts_dataset_files/cmudict-0.7b_nv22.10',
     out=dataset_files)
    
if not os.path.exists(f'{dataset_files}/heteronyms-052722'):
    wget.download('https://raw.githubusercontent.com/NVIDIA/NeMo/main/scripts/tts_dataset_files/heteronyms-052722',
     out=dataset_files)

normalizer_dict = {
    "input_case": 'cased',
    "lang": 'en',
    "whitelist": whitelist,
}

normalizer = Normalizer(
    input_case=normalizer_dict['input_case'],
    lang=normalizer_dict['lang'],
    whitelist=normalizer_dict['whitelist'],
)

normalizer_kwargs = {
    "verbose": False,
    "punct_pre_process": True,
    "punct_post_process": True
}

tokenizer = EnglishCharsTokenizer()

class ModifySpectrogramConfig:
    def __init__(
        self,
        audio_folder:str,
        manifest_root: str="manifests",
        sup_data_root: str="sup_data",
        text_tokenizer=None,
        text_normalizer=None,
        text_normalizer_call_kwargs: dict={},
        ) -> None:

        self.audio_folder = audio_folder
        self.manifest_name = audio_folder.split('/')[-1]
        self.manifest_folder = f"{manifest_root}/{self.manifest_name}"
        self.sup_data_path = f"{sup_data_root}/{self.manifest_name}"

        normalizer_kwargs = {}

        self.text_tokenizer = text_tokenizer or tokenizer
        self.text_normalizer = text_normalizer or normalizer
        self.text_normalizer_call_kwargs = normalizer_kwargs.update(text_normalizer_call_kwargs)

        self.out = {
            "sup_data_path": self.sup_data_path
        }

        if not os.path.exists(self.manifest_folder):
            os.makedirs(self.manifest_folder)
        if not os.path.exists(self.sup_data_path):
            os.makedirs(self.sup_data_path)
    
    def create_manifest_files(
            self,
            audio_folder: str="",
            librispeech: bool=False,
            speaker_id: int=None,
            manifest_folder: str="",
            manifest_name: str="",
        ) -> None:
        manifest_folder = manifest_folder or self.manifest_folder
        manifest_name = manifest_name or self.manifest_name
        audio_folder = audio_folder or self.audio_folder

        if librispeech:
            assert speaker_id, "librispeech audios require speaker_id"
        
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
                self.out['validation_datasets'] = mpath
            else:
                dicts = audio_dicts_train
                self.out['train_dataset'] = mpath
            with open(mpath, 'w') as f:
                for d in dicts:
                    f.write(json.dumps(d))
                    f.write('\n')
            logger.info(f"created {mpath}  file")
        logger.info("params_created are sample_rate, max_duration, min_duration, validation_datasets and train_dataset")
    
    def pre_calculate_supplementary_data(
        self,
        sup_data_path: str="",
        sample_rate: int=22050,
        manifest_folder: str="",
        manifest_name: str="",
        remove_sup_data: bool=True,
        **kwargs
        ) -> tuple:
        sup_data_path = sup_data_path or self.sup_data_path
        sample_rate = self.out.get('sample_rate') or sample_rate

        manifest_folder = manifest_folder or self.manifest_folder
        manifest_filename = manifest_name or self.manifest_name
    
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
        
        self.out['pitch_mean'] = pitch_mean
        self.out['pitch_std'] = pitch_std
        self.out['pitch_fmin'] = pitch_min
        self.out['pitch_fmax'] = pitch_max

        if remove_sup_data:
            shutil.rmtree(self.sup_data_path)
        
        return pitch_mean, pitch_std, pitch_min, pitch_max

def main(
        audio_folder,
        config_path: str="",
        manifest_root: str="manifests",
        sup_data_root: str="sup_data",
        text_normalizer_call_kwargs={},
        init_from: str="",
        model_params: dict={},
        trainer: dict={},
        exp_manager: dict={},
        train_dataset: dict={},
        train_dataloader: dict={},
        val_dataset: dict={},
        val_dataloader: dict={},
        remove_sup_data: bool=True,
        **model_kwargs
    ):
    msc = ModifySpectrogramConfig(
        audio_folder,
        manifest_root,
        sup_data_root,
        text_normalizer_call_kwargs,
    )
    msc.create_manifest_files()
    msc.pre_calculate_supplementary_data(remove_sup_data=remove_sup_data)
    base_keys = msc.out
    max_duration = base_keys.pop('max_duration')
    min_duration = base_keys.pop('min_duration')
    logger.info(base_keys)
    #other_base_keys
    #n_mel_channels, n_window_size, n_window_stride, n_fft, lowfreq, highfreq, window, sup_data_types, name
    
    train_dataset = {
        "min_duration": min_duration,
        "max_duration": max_duration,
    }
    val_dataset = {
        "min_duration": min_duration,
        "max_duration": max_duration,
    }
    change_configuration(
        audio_folder_name=audio_folder.split('/')[-1],
        base=base_keys,
        config_path=config_path,
        init_from=init_from,
        model=model_params,
        trainer=trainer,
        exp_manager=exp_manager,
        train_dataset=train_dataset,
        train_dataloader=train_dataloader,
        val_dataset=val_dataset,
        val_dataloader=val_dataloader,
        **model_kwargs
    )


def finetune_test():
    base = {
        "pitch_fmin":30,
        "pitch_fmax":512,
        "pitch_mean":121.9,
        "pitch_std":23.1,
        "sample_rate":22050,
        "validation_datasets": "manifests/manifest_val.json",
        "train_dataset": "manifests/manifest_train.json",
        "sup_data_path": "sup_data/sample",
    }

    mdl={
        "n_speakers": 1,
    }
    train_dataloader={
        "batch_size":24,
        "num_workers":4,
    }
    validation_dataloader={
        "batch_size":24,
        "num_workers":4,
    }
    mdl_kwargs = {
        "optim": {
            "name": "adam",
            "lr": 2e-4
        }
    }
    trainer ={
        "devices": 1,
        "strategy": None,
        "max_steps": 1000,
        "check_val_every_n_epoch":25,
        "log_every_n_epoch":5
    }
    
    change_configuration(
        base, 
        model=mdl,
        train_dataloader=train_dataloader,
        val_dataloader=validation_dataloader,
        specgen=True,
        trainer=trainer,
        **mdl_kwargs
    )
    logger.info("finetuning the test model")


if __name__ == "__main__":
    mdl={
        "n_speakers": 1,
    }
    train_dataloader={
        "batch_size":24,
        "num_workers":4,
    }
    validation_dataloader={
        "batch_size":24,
        "num_workers":4,
    }
    mdl_kwargs = {
        "optim": {
            "name": "adam",
            "lr": 2e-4
        }
    }
    trainer ={
        "devices": 1,
        "strategy": None,
        "max_steps": 1000,
        "check_val_every_n_epoch":25,
        "log_every_n_steps":5
    }
    main(
        'audios/6097_5_mins',
        remove_sup_data=False,
        model_params=mdl,
        train_dataloader=train_dataloader,
        trainer=trainer,
        **mdl_kwargs,
        )
