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
    def __init__(self, audio_folder: str=None, audio_zip: str=None):
        if not audio_folder and not audio_zip:
            raise Exception("provide either audio_folder or audio_zip")
        self.all_audios = 'audios/'

        self.audio_zip = audio_zip
        if audio_zip:
            self.uncompress()
        else:
            self.audio_folder = audio_folder
        
        if not os.path.exists(self.all_audios):
            os.mkdir(self.all_audios)

    def uncompress(self):
        if self.audio_zip:
            try:
                os.system(f'unzip {self.audio_zip}')
                f = self.audio_zip.replace('.zip', '')
                self.audio_folder = f"{self.all_audios}{f}"
                shutil.move(f, self.audio_folder)
                logger.info(f"uncompressed {self.audio_zip}")
                
            except shutil.Error as e:
                logger.warning(f"{e}")
        else:
            logger.info(f"nothing to uncompress")
        

    def flatten_audio_folder(self, audio_folder: str=None):
        folder =  audio_folder or self.audio_folder
        assert folder, "Audio folder must be provided"
        def move_files(folder=folder, files_path=[], dest=folder, empty_dirs=[]):
            l1 = os.listdir(folder)
            for l in l1:
                path = f'{folder}/{l}'
                if os.path.isfile(path):
                    files_path.append(path)
                    try:
                        shutil.move(path, dest)
                    except shutil.Error:
                        pass
                else:
                    temp_folder = f"{folder}/{l}"
                    empty_dirs.append(temp_folder)
                    files_path, empty_dirs = move_files(temp_folder, files_path, empty_dirs=empty_dirs)
            return files_path, empty_dirs
        _, empty = move_files()
        for a in empty:
            try:
                shutil.rmtree(a)
            except FileNotFoundError:
                pass
        logger.info(f"successfully flattened files in {folder}")

    def stereo_to_mono(self, audio_folder: str=None):
        """
        Function to be used when error RuntimeError: Argument #4: 
        Padding size should be less than the corresponding input dimension, but got: 
        padding (512, 512) at dimension 2 of input
        """
        audio_folder = audio_folder or self.audio_folder
        assert audio_folder, "Audio folder must be provided"
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
    parser.add_argument('-audio_folder', required=False, type=str)
    parser.add_argument('-audio_zip', required=False, type=str)
    arg = parser.parse_args()

    if not arg.audio_folder and not arg.audio_zip:
        parser.error("provide either -audio_folder or -audio_zip")
    return arg

def main():
    arg = argparser()
    wp = WavPreprocessing(audio_folder=arg.audio_folder, audio_zip=arg.audio_zip)
    wp.uncompress()
    wp.flatten_audio_folder()
    wp.stereo_to_mono()

if __name__ == "__main__":
    main()