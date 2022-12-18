import os
import sys
import json
import torch
import shutil

import librosa

from nemo.collections.tts.torch.data import TTSDataset
from nemo_text_processing.text_normalization.normalize import Normalizer
from nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers import EnglishCharsTokenizer

from tqdm.notebook import tqdm
from pydub import AudioSegment

import logging

DEBUG_LEVEL = logging.getLevelName(os.environ.get('DEBUG_LEVEL', 'INFO'))
def get_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter("%(asctime)s: name-%(name)s: func-%(funcName)s[%(lineno)s]: %(levelname)s:  %(message)s"))
        logger.addHandler(h)
    logger.setLevel(DEBUG_LEVEL)
    logger.propagate = False
    return logger

logger = get_logger(__name__)

normalizer = {
    "input_case":'cased',#['cased', 'lowercased']
    "lang": 'en',
    "whitelist":"tts_dataset_files/lj_speech.tsv",
}
text_normalizer = Normalizer(
    input_case=normalizer['input_case'],
    lang=normalizer['lang'],
    whitelist=normalizer['whitelist'],
)

text_normalizer_call_kwargs = {
    "verbose": False,
    "punct_pre_process": True,
    "punct_post_process": True
}

text_tokenizer = EnglishCharsTokenizer()

class AudioPreprocessing(object):
    def __init__(
        self,
        audio_folder: str,
        manifest_filename: str,
        manifest_folder: str ='manifests',
        text_tokenizer=text_tokenizer, 
        text_normalizer=text_normalizer,
        text_normalizer_call_kwargs=text_normalizer_call_kwargs
        ):

        self.audio_folder = audio_folder
        self.manifest_folder = manifest_folder
        self.manifest_filename = manifest_filename
        self.text_tokenizer = text_tokenizer
        self.text_normalizer = text_normalizer
        self.text_normalizer_call_kwargs = text_normalizer_call_kwargs
        self.out = {}
        if not os.path.exists(manifest_folder):
            os.mkdir(manifest_folder)

    def flatten_audio_folder(self, audio_folder: str=None, levels: int=2):
        audio_folder = audio_folder or self.audio_folder
        wd = os.getcwd()
        os.chdir(audio_folder)
        destination_dir = os.getcwd()
        l1 = os.listdir()
        for d in l1:
            try:
                os.chdir(d)
                l2 = os.listdir()
                for f in l2:
                    if levels == 2:
                        shutil.move(f, destination_dir)
                    else:
                        os.chdir(f)
                        l3 = os.listdir()
                        for l in l3:
                            shutil.move(f, destination_dir)
                os.chdir('..')
                os.rmdir(d)
            except NotADirectoryError:
                pass
        os.chdir(wd)

    def create_manifest(self, manifest_filename: str=None, audio_folder: str=None, manifest_folder: str=None):
        audio_folder = audio_folder or self.audio_folder
        manifest_filename = manifest_filename or self.manifest_filename
        manifest_folder = manifest_folder or self.manifest_folder

        audio_dicts = []
        sr = []
        min_duration = 1000
        max_duration = 0
        all_files = os.listdir(audio_folder)
        for a in all_files:
            if a.endswith('.wav'):
                fpath = f"{audio_folder}/{a}"
                with open(fpath.replace('.wav', '.txt')) as f:
                    text = f.read()
                duration = librosa.get_duration(filename=fpath)

                if min_duration > duration:
                    min_duration = duration
                if max_duration < duration:
                    max_duration = duration

                ad = {
                    "audio_filepath": fpath,
                    "text": text,
                    "duration": duration,
                }
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
            mpath = f"{manifest_folder}/{manifest_filename}{ext}"
            if i == 0:
                dicts = audio_dicts
            elif i == 1:
                dicts = audio_dicts_val
                self.out['val_manifest'] = mpath
            else:
                dicts = audio_dicts_train
                self.out['train_manifest'] = mpath
            with open(mpath, 'w') as f:
                for d in dicts:
                    f.write(json.dumps(d))
                    f.write('\n')
        return self.out

    def pre_calculate_supplementary_data(self, sup_data_path: str, sample_rate: int=None, delete_sup_folder=True, **kwargs):
        sample_rate = self.out.get('sample_rate') or sample_rate
        assert sample_rate, "provide sample_rate to calculate supplementary data"

        manifest_folder=kwargs.get('manifest_folder', self.manifest_folder)
        manifest_filename = kwargs.get('manifest_filename') or self.manifest_filename
    
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
        return pitch_mean, pitch_std, pitch_min, pitch_max
    
    def stereo_to_mono(self, audio_folder: str=None):
        """
        Function to be used when error RuntimeError: Argument #4: 
        Padding size should be less than the corresponding input dimension, but got: 
        padding (512, 512) at dimension 2 of input
        """
        audio_folder = audio_folder or self.audio_folder
        audio_list = os.listdir(audio_folder)
        for af in audio_list:
            af = f"{audio_folder}/{af}"
            sound = AudioSegment.from_wav(af)
            sound = sound.set_channels(1)
            sound.export(af, format="wav")

def run(audio_folder: str, manifest_filename: str, sup_data_path: str):
    ap = AudioPreprocessing(audio_folder, manifest_filename)
    ap.stereo_to_mono()
    ap.create_manifest()
    ap.pre_calculate_supplementary_data(sup_data_path)
    logger.info(ap.out)
    return ap.out

def run_vd(manifest_filename: str, sup_data_path: str, vd_zip: str='VD.zip'):
    if not os.path.exists('audios/VD'):
        if not os.path.isfile(vd_zip):
            raise FileNotFoundError(f"file {vd_zip} could not be found")
        os.system(f"unzip {vd_zip}")
        os.mkdir("audios")
        os.system("mv VD audios/VD")

    ap = AudioPreprocessing('audios/VD', manifest_filename)
    ap.flatten_audio_folder()
    ap.stereo_to_mono()
    ap.create_manifest()
    ap.pre_calculate_supplementary_data(sup_data_path)
    logger.info(ap.out)
    return ap.out

if __name__ == '__main__':
    run_vd("jave_vd", "aluta")