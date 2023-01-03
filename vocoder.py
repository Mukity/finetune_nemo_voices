import json

import numpy as np
import soundfile as sf

import torch
import argparse

from nemo.collections.tts.models import FastPitchModel
from nemo.collections.tts.torch.helpers import BetaBinomialInterpolator

from pathlib import Path
from modify_config import change_configuration

class ModifyVocoderConfig:    
    def __init__(
            self,
            manifest_name: str,
            spec_model: str,
            manifest_folder: str=None,
        ):
        self.manifest_folder = manifest_folder or f"manifests/{manifest_name}"
        self.manifest_name = manifest_name

        manifests = ['train', 'val']
        self.manifest_names = [f"{self.manifest_name}_{m}.json" for m in manifests]
        self.manifest_paths = [f"{self.manifest_folder}/{f}" for f in self.manifest_names]
        self.out = {}

        if spec_model.endswith('.nemo'):
            self.spec_model = FastPitchModel.restore_from(spec_model)
        elif spec_model.endswith('.ckpt'):
            self.spec_model = FastPitchModel.load_from_checkpoint(spec_model)
        else:
            self.spec_model = FastPitchModel.from_pretrained(spec_model)
    
    def _load_wav(self, audio_file):
        with sf.SoundFile(audio_file, 'r') as f:
            samples = f.read(dtype='float32')
        return samples.transpose()
    
    def make_manifests(self, spec_model: str=None, manifest_names: list=None, manifest_folder: str=None):
        manifest_names = manifest_names or self.manifest_names
        manifest_folder = manifest_folder or self.manifest_folder
        spec_model = spec_model or self.spec_model

        assert isinstance(manifest_names, list), "manifest_names must be a list"
        for manifest_filename in manifest_names:
            manifest_modified = self._make_manifest(spec_model, f"{manifest_filename}", manifest_folder)
            if "train" in manifest_modified:
                self.out["train_dataset"] = manifest_modified
            elif "val" in manifest_modified:
                self.out["validation_datasets"] = manifest_modified
    
    def _make_manifest(self, spec_model: str, manifest_filename: str, manifest_folder: str=None):
        manifest_folder = manifest_folder or self.manifest_folder
        manifest_path=f"{manifest_folder}/{manifest_filename}"
        spec_model = spec_model or self.spec_model
        spec_model.eval()

        records = []
        with open(manifest_path, "r") as f:
            for i, line in enumerate(f):
                records.append(json.loads(line))

        beta_binomial_interpolator = BetaBinomialInterpolator()

        device = spec_model.device
        save_dir = Path(f"mels/{manifest_filename.replace('.json', '_mel')}")
        save_dir.mkdir(exist_ok=True, parents=True)
        for i, r in enumerate(records):
            audio = self._load_wav(r["audio_filepath"])
            audio = torch.from_numpy(audio).unsqueeze(0).to(device)
            audio_len = torch.tensor(audio.shape[1], dtype=torch.long, device=device).unsqueeze(0)
            with torch.no_grad():
                if "normalized_text" in r:
                    text = spec_model.parse(r["normalized_text"], normalize=False)
                else:
                    text = spec_model.parse(r['text'])
                
                text_len = torch.tensor(text.shape[-1], dtype=torch.long, device=device).unsqueeze(0)
                spect, spect_len = spec_model.preprocessor(input_signal=audio, length=audio_len)
                attn_prior = torch.from_numpy(
                  beta_binomial_interpolator(spect_len.item(), text_len.item())
                ).unsqueeze(0).to(text.device)
                spectrogram = spec_model.forward(
                  text=text, 
                  input_lens=text_len, 
                  spec=spect, 
                  mel_lens=spect_len, 
                  attn_prior=attn_prior,
                )[0]

                save_path = save_dir / f"mel_{i}.npy"
                np.save(save_path, spectrogram[0].to('cpu').numpy())
                r["mel_filepath"] = str(save_path)
        
        manifest_modified = f"{manifest_folder}/voc_{manifest_filename}"
        with open(manifest_modified, "w") as f:
            for r in records:
                f.write(json.dumps(r) + '\n')
        return manifest_modified


def argparser():
    parser = argparse.ArgumentParser(description="modify argument for vocoder")
    parser.add_argument('-audio_folder', required=True, type=str)
    parser.add_argument('-spec_model', required=False, type=str, default='tts_en_fastpitch')
    parser.add_argument('-sample_rate', required=False, type=int, default=22050)
    parser.add_argument('-manifest_folder', type=str, default='')
    parser.add_argument('-config_path', default='', type=str)
    parser.add_argument('-init_from', default='', type=str)
    parser.add_argument('-model_params', default={}, type=json.loads)
    parser.add_argument('-trainer', default={}, type=json.loads)
    parser.add_argument('-exp_manager', default={}, type=json.loads)
    parser.add_argument('-train_dataset', default={}, type=json.loads)
    parser.add_argument('-train_dataloader', default={}, type=json.loads)
    parser.add_argument('-val_dataset', default={}, type=json.loads)
    parser.add_argument('-val_dataloader', default={}, type=json.loads)
    parser.add_argument('-model_kwargs', default={}, type=json.loads)
    parser.add_argument('-base_configs', default={}, type=json.loads)
    return parser.parse_args()

def main():
    args = argparser()
    audio_folder = args.audio_folder
    spec_model = args.spec_model
    sample_rate = args.sample_rate
    manifest_folder = args.manifest_folder
    config_path = args.config_path
    init_from = args.init_from
    model_params = {**{"max_steps":1000}, **args.model_params}
    trainer = {**{"check_val_every_n_epoch":10, "log_every_n_steps": 10}, **args.trainer}
    exp_manager = args.exp_manager
    train_dataset = {**{"min_duration":0,"max_duration":100,"sample_rate":sample_rate}, **args.train_dataset}
    train_dataloader = {**{"batch_size":16,"num_workers":4}, **args.train_dataloader}
    val_dataset = {**{"min_duration":0,"max_duration":100,"sample_rate":sample_rate}, **args.val_dataset}
    val_dataloader = {**{"batch_size":16,"num_workers":4}, **args.val_dataloader}
    model_kwargs = {**{"optim":{"lr": 0.00001}}, **args.model_kwargs}
    base_configs = args.base_configs

    mvc = ModifyVocoderConfig(audio_folder, spec_model, manifest_folder)
    mvc.make_manifests()
    base_dict = {**base_configs, **mvc.out}
    base_dict['sample_rate'] = sample_rate

    cfg = change_configuration(
        audio_folder,
        base_dict,
        config_path,
        init_from,
        model_params,
        trainer,
        exp_manager,
        train_dataset,
        train_dataloader,
        val_dataset,
        val_dataloader,
        specgen=False,
        model_kwargs=model_kwargs,
    )
    return cfg

if __name__ == '__main__':
    main()
    #python vocoder.py -audio_folder VD 