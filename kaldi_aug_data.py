import numpy as np
import os
import numpy as np
import random
import glob
import json
import random
from speechblender import *
import glob
import shutil

text_list  = '/alt/asr/yelkheir/kaldi/egs/arab_align_gop/s5/data/text'
text = {}
with open(text_list) as r:
    lines = r.readlines()
    for line in lines:
        id, char = line.split('\t')[0], line.split('\t')[1][:-1]
        text[id] = char

for elem in ['speechblender/augmented_50', 'speechblender/augmented_50_0']:
    print(elem)
    dir = elem.split('/')[1]
    if not os.path.isdir(dir):
        os.mkdir(dir)
    with open(dir+'/wav.scp','w') as w1, open(dir+'/text','w') as w2, open(dir+'/utt2spk','w') as w3: 
        for subfolder in glob.glob(elem+'/*'):
            print(subfolder)
            name, loc = subfolder.split('/')[-1].split('.')[0], subfolder
            tst = text[name]
            w1.write(name)
            w1.write('\t')
            w1.write(loc)
            w1.write('\n')

            w2.write(name)
            w2.write('\t')
            w2.write(tst)
            w2.write('\n')   

            w3.write(name)
            w3.write('\t')
            w3.write(name)
            w3.write('\n')       
