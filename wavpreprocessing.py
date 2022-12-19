import os
import sys

import argparse
import shutil

from pydub import AudioSegment

import logging

DEBUG_LEVEL = logging.getLevelName(os.environ.get('DEBUG_LEVEL', 'INFO'))
def get_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter("%(asctime)s: func-%(funcName)s[%(lineno)s]: %(levelname)s:  %(message)s"))
        logger.addHandler(h)
    logger.setLevel(DEBUG_LEVEL)
    logger.propagate = False
    return logger

logger = get_logger(__name__)

class WavPreprocessing:
    def __init__(self, audio_folder):
        self.audio_folder = audio_folder

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
        logger.info(f'{audio_folder} has been flattened with levels {levels}')

    def stereo_to_mono(self, audio_folder: str=None):
        """
        Function to be used when error RuntimeError: Argument #4: 
        Padding size should be less than the corresponding input dimension, but got: 
        padding (512, 512) at dimension 2 of input
        """
        audio_folder = audio_folder or self.audio_folder
        audio_list = os.listdir(audio_folder)
        for af in audio_list:
            if af.endswith('.wav'):
                af = f"{audio_folder}/{af}"
                sound = AudioSegment.from_wav(af)
                sound = sound.set_channels(1)
                sound.export(af, format="wav")
        logger.info(f"audios in {audio_folder} have been converted from stereo to mono")

def argparser():
    parser = argparse.ArgumentParser(
        prog="wav preprocessing",
        description="Wav file preprocessing",
    )
    parser.add_argument('-audio_folder', required=True, type=str)
    parser.add_argument('-levels', type=int)
    return parser.parse_args()

def main():
    arg = argparser()
    audio_folder = arg.audio_folder
    levels = arg.levels
    wp = WavPreprocessing(audio_folder)
    wp.flatten_audio_folder(levels=levels)
    wp.stereo_to_mono()

if __name__ == "__main__":
    main()