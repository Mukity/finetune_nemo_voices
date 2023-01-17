#python spectrogram.py -audio_folder VD -test_data True
#python finetune.py -mode specgen -config_name fastpitch_align_v1.05_VD -checkpoint_name first
#python vocoder.py -audio_folder VD -trainer '{"max_steps":2000}' -train_dataset '{"n_segments":8192}' -val_dataset '{"n_segments":8192}'
#python finetune.py -mode vocoder -config_name hifigan_VD -checkpoint_name first

#python spectrogram.py -audio_folder VD -test_data True -init_from models/finetuned/VD/FastPitch/first.ckpt
#python finetune.py -mode specgen -config_name fastpitch_align_v1.05_VD -checkpoint_name second
#python vocoder.py -audio_folder VD -trainer '{"max_steps":2000}' -train_dataset '{"n_segments":8192}' -val_dataset '{"n_segments":8192}' -init_from models/finetuned/VD/HifiGan/first.ckpt
#python finetune.py -mode vocoder -config_name hifigan_VD -checkpoint_name second

python spectrogram.py -audio_folder nptel-pure -trainer '{"max_steps":200000000,"max_epochs":120}' -test_data True
python finetune.py -mode specgen -config_name fastpitch_align_v1.05_nptel-pure
python vocoder.py -audio_folder nptel-pure -trainer '{"max_steps":200000000,"max_epochs":120}' -train_dataset '{"n_segments":8192}' -val_dataset '{"n_segments":8192}'
python finetune.py -mode vocoder -config_name hifigan_nptel-pure
