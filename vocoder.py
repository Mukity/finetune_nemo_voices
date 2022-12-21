import os
import wget
import json
import argparse

import numpy as np
import soundfile as sf

import torch
import pytorch_lightning as pl

from nemo.collections.tts.torch.helpers import BetaBinomialInterpolator
from nemo.collections.tts.models import FastPitchModel
from nemo.collections.tts.models import HifiGanModel
from nemo.utils.exp_manager import exp_manager

from pathlib import Path
from modify_conf import modify_config

if not os.path.exists('config'):
    os.mkdir('config')

if not os.path.isfile('config/hifigan.yaml'):
    wget.download('https://raw.githubusercontent.com/nvidia/NeMo/main/examples/tts/conf/hifigan/hifigan.yaml', out='config')

class VocoderConfigPreprocessing:    
    def __init__(self, manifest_name: str, spec_model: str, manifest_folder: str=None, from_file=True):
        self.manifest_folder = manifest_folder or "manifests"
        self.manifest_name = manifest_name

        manifests = ['train', 'val']
        self.manifest_names = [f"{self.manifest_name}_{m}.json" for m in manifests]
        self.manifest_paths = [f"{self.manifest_folder}/{f}" for f in self.manifest_names]
        self.out = {}

        if from_file:
            self.spec_model = FastPitchModel.restore_from(spec_model)
        else:
            self.spec_model = FastPitchModel.load_from_checkpoint(spec_model)
    
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
            manifest_modified = self.make_manifest(spec_model, manifest_filename, manifest_folder)
            if "train" in manifest_modified:
                self.out["train_dataset"] = manifest_modified
            elif "val" in manifest_modified:
                self.out["val_dataset"] = manifest_modified
    
    def make_manifest(self, spec_model: str, manifest_filename: str, manifest_folder: str=None):
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

def finetuning(cfg):
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    model = HifiGanModel(cfg=cfg.model, trainer=trainer)
    model.maybe_init_from_pretrained_checkpoint(cfg=cfg)
    trainer.fit(model)

def run(manifest_name, spec_model, manifest_folder, pretrained, config_name, config_path, pitch_file, **kwargs):
    vc = VocoderConfigPreprocessing(manifest_name, spec_model, manifest_folder, pretrained)
    vc.make_manifests()

    train_dataset = vc.out['train_dataset']
    validation_datasets = vc.out['val_dataset']
    with open(pitch_file) as f:
        pitch_dict = json.load(f)
    cfg = modify_config(train_dataset, validation_datasets, pitch_dict, config_name, config_path=config_path, specgen=False,**kwargs)
    finetuning(cfg)

def argparser():
    parser = argparse.ArgumentParser(
        prog="vocoder preprocessing",
        description="generating vocoder config and yaml files for use in the finetune",
    )
    parser.add_argument('-manifest_name', type=str, required=True)
    parser.add_argument('-manifest_folder', type=str, required=False)
    parser.add_argument('-ckpt_spec_model', type=str, required=False)
    parser.add_argument('-nemo_spec_model', type=str, required=False)
    parser.add_argument('-config_path', type=str)
    parser.add_argument('-config_name', type=str)
    parser.add_argument('-pitch_file', type=str)

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

    args = parser.parse_args()

    if not args.ckpt_spec_model and not args.nemo_spec_model:
        parser.error("either -nemo_spec_model or -ckpt_spec_model is required")
    return args

def main():
    arg = argparser()
    if arg.ckpt_spec_model:
        spec_model = arg.ckpt_spec_model
        pretrained = False
    else:
        spec_model = arg.nemo_spec_model
        pretrained = True
    manifest_folder = arg.manifest_folder
    manifest_name = arg.manifest_name
    config_name = arg.config_name
    config_path = arg.config_path
    pitch_file = arg.pitch_file or 'pitch_file.json'

    kwargs = {
        "model_params": arg.model_params,
        "model_train_dataset": arg.model_train_dataset,
        "model_train_dataloader": arg.model_train_dataloader,
        "model_val_dataset": arg.model_val_dataset,
        "model_val_dataloader": arg.model_val_dataloader,
        "preprocessor": arg.preprocessor,
        "optim": arg.optim,
        "trainer": arg.trainer,
        "exp_manager": arg.exp_manager,
        "generator": arg.generator,
        "name": arg.name,
        "symbols_embedding_dim": arg.symbols_embedding_dim,
    }

    run(manifest_name, spec_model, manifest_folder, pretrained, config_name, config_path, pitch_file, **kwargs)

if __name__ == '__main__':
    main()
    '''
    python vocoder.py -manifest_name vd_manifest
    -ckpt_spec_model nemo_experiments/FastPitch/2022-12-19_12-12-24/checkpoints/FastPitch--val_loss\=3.6752-epoch\=1000-last.ckpt
    -model_train_dataloader '{"batch_size":32}'
    -model_params '{"max_epochs": 1000}'
    -optim '{"lr": 0.00001}'
    -trainer '{"check_val_every_n_epoch": 10}'
    '''