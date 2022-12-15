import os
import sys
import wget
import json
import shutil

from nemo.collections.tts.torch.data import TTSDataset   
from nemo_text_processing.text_normalization.normalize import Normalizer
from nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers import EnglishCharsTokenizer

import torch
import librosa

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

if not os.path.exists('tts_dataset_files'):
    os.mkdir('tts_dataset_files')

if not os.path.isfile('tts_dataset_files/lj_speech.tsv'):
    wget.download('https://raw.githubusercontent.com/NVIDIA/NeMo/main/nemo_text_processing/text_normalization/en/data/whitelist/lj_speech.tsv', out='tts_dataset_files/')

#more on the normalizer look at nemo_text_processing.text_normalization.normalize
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

#more tokenizers look at nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers
#TODO try different tokenizer
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
    
    def get_sample_rate(self, audio_folder: str=None):
        audio_folder = audio_folder or self.audio_folder
        sr = []
        audio_files = []
        all_files = os.listdir(audio_folder)
        for f in all_files:
            if f.endswith('.wav'):
                file_path = f"{audio_folder}/{f}"
                sr_ = librosa.get_samplerate(file_path)
                audio_files.append(file_path)
                sr.append(sr_)
        return audio_files, sum(sr)/len(sr)
    
    def create_manifest_files_get_min_max_durations(self, audio_list: list, train_val_split: float=0.9, manifest_filename: str=None, manifest_folder: str=None):
        manifest_folder = manifest_folder or self.manifest_folder
        manifest_filename = manifest_filename or self.manifest_filename

        manifest = []
        min_duration = 100
        max_duration = 0
        for w in audio_list:
            txt = w.replace('.wav', '.txt')
            with open(txt) as f:
                text = f.read()
            duration = librosa.get_duration(filename=w)

            if duration < min_duration:
                min_duration = duration
            if duration > max_duration:
                max_duration = duration

            manifest.append({
                "audio_filepath": w,
                "duration": duration,
                "text": text,
                #"text_no_preprocessing": text,
                #"text_normalized": "use normalizer function to generate this"
            })
    
        val_manifest = manifest.copy()
        train_manifest = []
        for i in range(int(len(val_manifest)*train_val_split)):
            train_manifest.append(val_manifest.pop())
        train_dataset = manifest_filename.replace('.json', '_train.json')
        val_dataset = manifest_filename.replace('.json', '_val.json')
        self.save_list_to_file(train_manifest, train_dataset, manifest_folder)
        self.save_list_to_file(val_manifest, val_dataset, manifest_folder)
        self.save_list_to_file(manifest, manifest_filename, manifest_folder)
        return min_duration, max_duration, train_dataset, val_dataset
    
    def save_list_to_file(self, json_list: list, filename: str, manifest_folder: str=None):
        manifest_folder = manifest_folder or self.manifest_folder
        
        fpath = f'{manifest_folder}/{filename}'
        if os.path.isfile(fpath):
            logger.info(f'file {fpath} exists')
            return 
        
        with open(fpath, 'w') as f:
            for line in json_list:
                f.write(json.dumps(line))
                f.write('\n')    
        logger.info(f'file {fpath} created')
    
    def stereo_to_mono(self, audio_list: list):
        """
        Function to be used when error RuntimeError: Argument #4: 
        Padding size should be less than the corresponding input dimension, but got: 
        padding (512, 512) at dimension 2 of input
        """
        for af in audio_list:
            sound = AudioSegment.from_wav(af)
            sound = sound.set_channels(1)
            sound.export(af, format="wav")
    
    def pre_calculate_supplementary_data(self, sup_data_path: str, sample_rate: int, manifest_filename: str='vd_manifest.json', **kwargs):
        sup_data_types=kwargs.get('sup_data_types', ["align_prior_matrix", "pitch"])
        text_tokenizer=kwargs.get('text_tokenizer', self.text_tokenizer)
        text_normalizer=kwargs.get('text_normalizer', self.text_normalizer)
        text_normalizer_call_kwargs=kwargs.get('text_normalizer_call_kwargs', self.text_normalizer_call_kwargs)
        manifest_folder=kwargs.get('manifest_folder', self.manifest_folder)
    
        stages = ["train", "val"]
        stage2dl = {}
        for stage in stages:
            if stage == "train":
                fname = manifest_filename.replace('.json', '_train.json')
            else:
                fname = manifest_filename.replace('.json', '_val.json')
            ds = TTSDataset(
                manifest_filepath=f"{manifest_folder}/{fname}",
                sample_rate=sample_rate,
                sup_data_path=sup_data_path,
                sup_data_types=sup_data_types,
                n_fft=1024,
                win_length=1024,
                hop_length=256,
                window="hann",
                n_mels=80,
                lowfreq=0,
                highfreq=8000,
                text_tokenizer=text_tokenizer,
                text_normalizer=text_normalizer,
                text_normalizer_call_kwargs=text_normalizer_call_kwargs
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
        return pitch_mean, pitch_std, pitch_min, pitch_max

def run(audio_folder: str, manifest_filename: str, sup_data_path: str):
    ap = AudioPreprocessing(audio_folder, manifest_filename)
    audio_list, sr = ap.get_sample_rate()
    ap.stereo_to_mono(audio_list)
    min_duration, max_duration, train_dataset, validation_dataset = ap.create_manifest_files_get_min_max_durations(audio_list)
    pitch_mean, pitch_std, pitch_min, pitch_max = ap.pre_calculate_supplementary_data(sup_data_path, sr)
    return {
        "sample_rate": sr,
        "pitch_mean": pitch_mean,
        "pitch_std": pitch_std,
        "pitch_min": pitch_min,
        "pitch_max": pitch_max,
        "min_duration": min_duration,
        "max_duration": max_duration,
        "sup_data_path": sup_data_path,
        "train_dataset": train_dataset,
        "validation_dataset": validation_dataset,
    }

def run_vd(manifest_filename: str, sup_data_path: str, vd_zip: str='VD.zip'):
    if not os.path.exists('audios/VD'):
        if not os.path.isfile(vd_zip):
            raise FileNotFoundError(f"file {vd_zip} could not be found")
        os.system(f"unzip {vd_zip}")
        os.mkdir("audios")
        os.system("mv VD audios/VD")

    ap = AudioPreprocessing('audios/VD', manifest_filename)
    ap.flatten_audio_folder()
    audio_list, sr = ap.get_sample_rate()
    ap.stereo_to_mono(audio_list)
    min_duration, max_duration, train_dataset, validation_dataset = ap.create_manifest_files_get_min_max_durations(audio_list)
    pitch_mean, pitch_std, pitch_min, pitch_max = ap.pre_calculate_supplementary_data(sup_data_path, sr)
    out =  {
        "sample_rate": sr,
        "pitch_mean": pitch_mean,
        "pitch_std": pitch_std,
        "pitch_fmin": pitch_min,
        "pitch_fmax": pitch_max,
        "min_duration": min_duration,
        "max_duration": max_duration,
        "sup_data_path": sup_data_path,
        "train_dataset": train_dataset,
        "validation_datasets": validation_dataset,
    }
    logger.info(out)
    return out