import os
from pydub import AudioSegment

train = 'train.txt'
train_list = []
with open(train) as r :
    lines = r.readlines()
    train_list = [line[:-1] for line in lines]

scores = 'data/scores.txt'
scores_list = {}
with open(scores) as r :
    lines = r.readlines()
    for line in lines:
        scores_list[line.split('\t')[0]] = int(line.split('\t')[1])

loc = 'data/wav.scp'
loc_list = {}
with open(loc) as r :
    lines = r.readlines()
    for line in lines:
        loc_list[line.split('\t')[0]] = line.split('\t')[1][:-1]


inc = 0

ctm = '/alt/asr/yelkheir/kaldi/egs/arab_align_gop/s5/exp/alignment/ctm_phn.txt'
with open(ctm) as r:
    lines = r.readlines()
    for line in lines:
        split_ = line.split(' ')
        file, start, gap, char = split_[0], float(split_[2]), float(split_[3]), split_[4][:-1]

        end = start+gap
        audio = loc_list[file]
        if file in train_list :
            if scores_list[file] == 5:
                inc +=1
                if not os.path.isdir('augmented_data'):
                    os.mkdir('augmented_data')

                
                if int(end*1000)-int(start*1000)>100:                
                    if not os.path.isdir('augmented_data/{}'.format(char)):
                        os.mkdir('augmented_data/{}'.format(char))
                        
                    read_audio = AudioSegment.from_wav(audio)
                    phn = read_audio[int(start*1000):int(end*1000)]
                    phn.export("augmented_data/{0}/test{1}.wav".format(char,inc), format="wav")

                 




