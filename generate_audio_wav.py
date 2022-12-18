import soundfile as sf

from nemo.collections.tts.models.base import SpectrogramGenerator, Vocoder

def create_audio_file(text: str, filename: str, spec_gen_pretrained: str=None, vocoder_pretrained: str=None,
            spec_gen_nemo: str=None, vocoder_nemo: str=None, spec_gen_checkpoint: str=None, vocoder_checkpoint: str=None):

    assert spec_gen_checkpoint or spec_gen_nemo or spec_gen_pretrained, "SpectrogramGenerator is not provided"
    assert vocoder_checkpoint or vocoder_nemo or vocoder_pretrained, "Vocoder is not provided"

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
    sf.write(filename, audio, 22050)

if __name__ == '__main__':
    text = "Just keep being true to yourself, if you're passionate about something go for it. Don't sacrifice anything, just have fun."
    create_audio_file(text, "audio.wav", spec_gen_pretrained='tts_en_fastpitch',vocoder_pretrained='tts_hifigan')