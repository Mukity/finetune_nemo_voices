import os
import hashlib
import argparse

import soundfile as sf

from nemo.collections.tts.models import FastPitchModel
from nemo.collections.tts.models import HifiGanModel

from helpers import get_logger
logger = get_logger(__file__.split('.')[0])

default_audio_folder = "audios/generated_audios"

def create_audio_file(text: str, filename: str, spec_gen: str, vocoder: str, audio_folder: str=""):
    filename = filename or hashlib.md5(text.encode()).hexdigest() + '.wav'
    audio_folder = audio_folder or default_audio_folder
    if spec_gen.endswith('.nemo'):
        spec_generator = FastPitchModel.restore_from(spec_gen)
    elif spec_gen.endswith('.ckpt'):
        spec_generator = FastPitchModel.load_from_checkpoint(spec_gen)
    else:
        spec_generator = FastPitchModel.from_pretrained(spec_gen)
    
    if vocoder.endswith('.nemo'):
        vocoder = HifiGanModel.restore_from(vocoder)
    elif vocoder.endswith('.ckpt'):
        vocoder = HifiGanModel.load_from_checkpoint(vocoder)
    else:
        vocoder = HifiGanModel.from_pretrained(vocoder)
    
    spec_generator.eval()

    parsed = spec_generator.parse(text)
    spectrogram = spec_generator.generate_spectrogram(tokens=parsed)
    vocoder.to('cpu')
    audio = vocoder.convert_spectrogram_to_audio(spec=spectrogram)
    audio = audio.to('cpu').detach().numpy()[0]
    if not os.path.exists(audio_folder):
        os.mkdir(audio_folder)
    save_path =  f"{audio_folder}/{filename}"
    sf.write(save_path, audio, 22050)
    logger.info(f"audio file generated on path {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Argument parser")
    parser.add_argument("-text", type=str, required=True)
    parser.add_argument("-filename", type=str, required=True)
    parser.add_argument("-specgen", type=str, required=True)
    parser.add_argument("-vocoder", type=str, required=True)
    parser.add_argument("-audio_folder", type=str, required=False)
    args = parser.parse_args()

    create_audio_file(
        args.text,
        args.filename,
        args.specgen,
        args.vocoder,
        args.audio_folder,
    )

if __name__ == '__main__':
    main()
