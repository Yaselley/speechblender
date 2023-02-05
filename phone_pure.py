phone_file = '/alt/asr/yelkheir/kaldi/egs/arab_align_gop/s5/lang/phones.txt'

phn_unique = {}
to_phn = {}

with open(phone_file) as r:
    lines = r.readlines()
    i = 0
    for line in lines:
        char = line.split(' ')[0]
        if char[0]!= '#':
            if char.split('_')[0] not in phn_unique.keys():
                phn_unique[char.split('_')[0]] = i
                i+=1

with open('/alt/asr/yelkheir/kaldi/egs/arab_align_gop/s5/lang/pure_phone.int', 'w') as w:
    i = 0
    for line in lines: 
        char, id = line.split(' ')[0].split('_')[0], int(line.split(' ')[1])
        if char[0] != '#':
            w.write(str(i))
            w.write(' ')
            w.write(str(phn_unique[char]))
            w.write('\n')
            i +=1