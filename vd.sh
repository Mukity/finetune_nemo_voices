python spectrogram.py  -audio_folder VD
python finetune.py -mode specgen -config_name fastpitch_align_v1.05_VD
python vocoder.py -audio_folder VD
python finetune.py -mode vocoder -config_name hifigan_VD
