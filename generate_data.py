import numpy as np
import os
import audiosegment
from pydub import AudioSegment, effects 
import numpy as np
import random
import librosa
from IPython.display import display, Audio
from pydub.generators import WhiteNoise
import glob
import json
import random
from speechblender import *
import glob
import shutil

def speed_change(sound, speed=1.0):
    # Manually override the frame_rate. This tells the computer how many
    # samples to play per second
    sound_with_altered_frame_rate = sound._spawn(sound.raw_data, overrides={
         "frame_rate": int(sound.frame_rate * speed)
      })
     # convert the sound with altered frame rate to a standard frame rate
     # so that regular playback programs will work right. They often only
     # know how to play audio at standard frame rate (like 44.1k)
    return sound_with_altered_frame_rate.set_frame_rate(sound.frame_rate)


train_list = '/alt/asr/yelkheir/kaldi/egs/arab_align_gop/s5/train.txt'
loc_list = '/alt/asr/yelkheir/kaldi/egs/arab_align_gop/s5/data/wav.scp'
scores_list = '/alt/asr/yelkheir/kaldi/egs/arab_align_gop/s5/data/scores.txt'
text_list  = '/alt/asr/yelkheir/kaldi/egs/arab_align_gop/s5/data/text'
ctm_list = '/alt/asr/yelkheir/kaldi/egs/arab_align_gop/s5/exp/alignment/ctm_phn.txt'

train = []
loc = {}
scores = {}
text = {}
ctm = {}

with open(train_list) as r:
    train = r.readlines()
    train = [line[:-1] for line in train]

with open(loc_list) as r:
    lines = r.readlines()
    for line in lines:
        id, loc_tmp = line.split('\t')[0], line.split('\t')[1][:-1]
        loc[id] = loc_tmp

with open(scores_list) as r:
    lines = r.readlines()
    for line in lines:
        id, score = line.split('\t')[0], int(line.split('\t')[1])
        scores[id] = score

with open(text_list) as r:
    lines = r.readlines()
    for line in lines:
        id, char = line.split('\t')[0], line.split('\t')[1][:-1]
        text[id] = char

past = []
i = 0

with open(ctm_list) as r:
    lines = r.readlines()
    for line in lines:
        split_ = line.split(' ')
        id, start, gap, phn = split_[0], float(split_[2]), float(split_[3]), split_[4][:-1]
        if id not in past:
            past +=[id]
            ctm[id] = dict()
            i = 0

        ctm[id]['{0}_{1}'.format(phn,i)] = {'start':start,'end':start+gap}
        i +=1

print(ctm)

aug_dir = 'augmented_data/*'
chars = glob.glob(aug_dir)
chars_id = [elem.split('/')[-1] for elem in chars]


'''
We are proposing four blending methods:
    - 1: SmoothOverlay with a proportion factor
    - 2: CutMix 
    - 3: Smooth Concatenation 
    - 4: Smooth-Gaussian Overlay
'''

def mix_phone(save_dir, group):
    eval = 0
    dir_path = '/alt/asr/yelkheir/kaldi/egs/arab_align_gop/s5/X_aug_phns_2'
    chars_dirs = [name for name in os.listdir(dir_path)if os.path.isdir(os.path.join(dir_path, name))]

    with open('augmented_0_all/scores.txt','w') as w:
        for audio in train:
            loc_dir = loc[audio]
            score = scores[audio]
            ctms = ctm[audio]
            texts = [*text[audio]]

            if 'UNK_0' not in ctms.keys() and score == 5:
                num = len(texts)

                if num !=0:            
                    items = random.sample(range(len(texts)), num)
                    items = sorted(items,reverse=True)
                    tmp_ctms = []
                    shutil.copyfile(loc_dir,'test.wav')
                    tmp_elms = []
                    for elem in range(len(texts)-1,-1,-1):
                        if elem in items:
                            tmp_ctms +=[ctms['{}_{}'.format(texts[elem],elem)]]
                            tmp_elms += [texts[elem]]
                            w.write(loc_dir.split('/')[-1]+'.{}'.format(elem))
                            w.write('\t')
                            w.write(str(1))
                            w.write('\n')

                        else:
                            w.write(loc_dir.split('/')[-1]+'.{}'.format(elem))
                            w.write('\t')
                            w.write(str(2))
                            w.write('\n')


                    for i, elem in enumerate(tmp_ctms):
                        start = elem['start']
                        end = elem['end']
                        char = tmp_elms[i]
                        ## select random char

                        if eval==1:
                            y1 = random.sample(glob.glob('augmented_data_2/{0}/*'.format(char)), 1)
                        else : 
                            copy = chars_dirs.copy()
                            copy = [x for x in copy if x != char]
                            char = random.sample(copy, 1)
                            y1 = random.sample(glob.glob('augmented_data_2/{0}/*'.format(char[0])), 1)

                        read_audio = AudioSegment.from_wav('test.wav')
                        y2 = read_audio[int(start*1000):int(end*1000)]
                        y2.export("test_phn.wav", format="wav")
                        y2 = 'test_phn.wav'

                        ## For 0 construction pplease uncommene this part
                        # y1 = random.sample(chars_id, 1)
                        # char = y1[0]
                        # y1 = random.sample(glob.glob('augmented_data/{0}/*'.format(char)), 1)


                        if len(y2)/48000> 0.04 :
                            cutmix(y1[0],y2,  sr=48000)
                        else :
                            smoothgaussian(y1[0], y1[0], sr=48000)
                        read_audio_phn = AudioSegment.from_wav('test_phn.wav')
                        
                        # read_audio_phn = AudioSegment.from_wav(y1[0])

                        y_out = read_audio[:start*1000]+read_audio_phn+read_audio[end*1000:]
                        y_out.export("test.wav", format="wav")
                    
                    shutil.copyfile("test.wav",'augmented_0_all/{}.wav'.format(audio))



                    
                    ## select blender method 
                    ## blend to test
                    ## save 


mix_phone('hi',group =2)




