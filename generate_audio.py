import os

import hashlib

import argparse

import soundfile as sf

from nemo.collections.tts.models.base import SpectrogramGenerator, Vocoder

audio_folder = "audios/generated_audios"

def create_audio_file(text: str, spec_gen_pretrained: str=None, vocoder_pretrained: str=None,
            spec_gen_nemo: str=None, vocoder_nemo: str=None, spec_gen_checkpoint: str=None, vocoder_checkpoint: str=None, audio_folder=audio_folder):

    assert spec_gen_checkpoint or spec_gen_nemo or spec_gen_pretrained, "SpectrogramGenerator is not provided"
    assert vocoder_checkpoint or vocoder_nemo or vocoder_pretrained, "Vocoder is not provided"
    filename = hashlib.md5(text.encode()).hexdigest() + '.wav'
    if spec_gen_pretrained:
        spec_generator = SpectrogramGenerator.from_pretrained(spec_gen_pretrained)
    elif spec_gen_nemo:
        spec_generator = SpectrogramGenerator.restore_from(spec_gen_nemo)
    else:
        spec_generator = SpectrogramGenerator.load_from_checkpoint(spec_gen_checkpoint)
    spec_generator.eval()
    
    if vocoder_pretrained:
        vocoder = Vocoder.from_pretrained(vocoder_pretrained)
    elif vocoder_nemo:
        vocoder = Vocoder.restore_from(vocoder_nemo)
    else:
        vocoder = Vocoder.load_from_checkpoint(vocoder_checkpoint)
    
    parsed = spec_generator.parse(text)
    spectrogram = spec_generator.generate_spectrogram(tokens=parsed)
    audio = vocoder.convert_spectrogram_to_audio(spec=spectrogram)
    audio = audio.to('cpu').detach().numpy()[0]
    if not os.path.exists(audio_folder):
        os.mkdir(audio_folder)
    sf.write(f"{audio_folder}/{filename}", audio, 22050)

def main():
    parser = argparse.ArgumentParser(description="Argument parser")
    parser.add_argument("-text", required=True)
    parser.add_argument("-spec_gen_pretrained")
    parser.add_argument("-spec_gen_nemo")
    parser.add_argument("-spec_gen_checkpoint")
    parser.add_argument("-vocoder_pretrained")
    parser.add_argument("-vocoder_nemo")
    parser.add_argument("-vocoder_checkpoint")
    args = parser.parse_args()

    if not args.spec_gen_pretrained and not args.spec_gen_nemo and not args.spec_gen_checkpoint:
        parser.error("one of the arguments -spec_gen_pretrained, -spec_gen_nemo or -spec_gen_checkpoint must be specified")
    if not args.vocoder_pretrained and not args.vocoder_nemo and not args.vocoder_checkpoint:
        parser.error("one of the arguments -vocoder_pretrained, -vocoder_nemo or -vocoder_checkpoint must be specified")

    create_audio_file(
        args.text,
        args.spec_gen_pretrained,
        args.vocoder_pretrained,
        args.spec_gen_nemo,
        args.vocoder_nemo,
        args.spec_gen_checkpoint,
        args.vocoder_checkpoint,
    )

if __name__ == '__main__':
    main()