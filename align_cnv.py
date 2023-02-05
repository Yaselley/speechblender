file = '/alt/asr/yelkheir/kaldi/egs/arab_align_gop/s5/exp/alignment/ctm.txt'
phn_dict_file = '/alt/asr/yelkheir/kaldi/egs/arab_align_gop/s5/lang/phones.txt'

phn_dict = dict()
with open(phn_dict_file,'r') as r:
    lines = r.readlines()
    for line in lines:
        char, id =line.split(' ')[0],line.split(' ')[1]
        phn_dict[int(id)] = char


with open(file,'r') as r, open('exp/alignment/ctm_phn.txt','w') as w:
    lines_r = r.readlines()
    for line in lines_r:
        char = line.split(' ')[-1]
        char = int(char)
        char = phn_dict[char]
        new_line = ' '.join(line.split(' ')[:-1])+' '+char.split('_')[0]+'\n'
        if char != 'SIL':
            w.write(new_line)
            w.flush()