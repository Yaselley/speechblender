import os
import wave

def filter_wav_files(src_dir, dest_dir):
    for dirpath, dirnames, filenames in os.walk(src_dir):
        for filename in filenames:
            if filename.endswith(".wav"):
                src_file = os.path.join(dirpath, filename)
                with wave.open(src_file, 'rb') as wave_file:
                    frame_rate = wave_file.getframerate()
                    num_frames = wave_file.getnframes()
                    if num_frames / frame_rate > 0.04:
                        dest_path = os.path.join(dest_dir, os.path.relpath(dirpath, src_dir))
                        if not os.path.exists(dest_path):
                            os.makedirs(dest_path)
                        dest_file = os.path.join(dest_path, filename)
                        os.link(src_file, dest_file)

src_dir = "/alt/asr/yelkheir/kaldi/egs/arab_align_gop/s5/augmented_data"
dest_dir = "/alt/asr/yelkheir/kaldi/egs/arab_align_gop/s5/augmented_data_2"
filter_wav_files(src_dir, dest_dir)
