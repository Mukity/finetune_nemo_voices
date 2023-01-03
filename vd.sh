python spectrogram.py  -audio_folder VD -default_samplerate True
python finetune.py -mode specgen -config_name fastpitch_align_v1.05_VD
py vocoder.py -audio_folder VD -spec_model tts_en_fastpitch
python finetune.py -mode vocoder -config_name hifigan_VD