#!/bin/bash
# Copyright 2020 Tsinghua University (Author: Zhiyuan Tang)
# Adapted Yassine EL KHEIR - RA - QCRI - Speech Learning
# Apache 2.0

. ./path.sh

stage=1
# num of jobs
nj=25
use_gpu=no

# prepare directory directory
data=$1


# models trained with kaldi
baseline=/alt/asr/yelkheir/kaldi/egs/gop_ls/s5 #../../librispeech/s5
lang=$baseline/data/lang
mdl_dir=exp/nnet3_online_ar_en_2.0
mdl=$mdl_dir/final.mdl

mkdir $data/out

if [ $stage -le 1 ]; then
  # Get feats with same config as training ASR.
  # data has kaldi-style structure, including at least
  # wav.scp, text, utt2spk and spk2utt (utt2spk and spk2utt can be fake,
  # i.e., just wav-id to wav-id).

  online2-wav-dump-features --config=exp/nnet3_online_ar_en_2.0/conf/online_nnet3_feat.conf ark:$data/spk2utt.scp \
   scp:$data/wav.scp ark,scp:$data/feats.ark,$data/feats.scp 

  ivector-extract-online2 --config=exp/nnet3_online_ar_en_2.0/conf/ivector_extractor.conf \
    ark:$data/spk2utt.scp scp:$data/feats.scp ark:- \
    | copy-feats --compress=true ark:- \
    ark,scp:$data/ivector_online.ark,$data/ivector_online.scp
fi

echo 'PASS Features and i-vectors'

if [ $stage -le 7 ]; then
  # Split data and make phone-level transcripts
    utils/split_data.sh $data $nj
fi

echo 'PASS Split Data'


if [ $stage -le 2 ]; then
  # Get state level posterior, log(p(s|o_t)) for all states.
   run.pl JOB=1:$nj $data/log/posterior.JOB.log \
        nnet3-compute --use-gpu=no  \
        --frame-subsampling-factor=1 \
        --online-ivector-period=10 \
        --online-ivectors=scp:$data/ivector_online.scp \
        exp/nnet3_online_ar_en_2.0/final.raw scp:$data/split25/JOB/feats.scp ark,t:$data/out/posterior.JOB.ark;
fi

echo 'PASS Posterior Calculations'

if [ $stage -le 3 ]; then
  # Get pdf and phone level force-aligment.
  steps/nnet3/align.sh  --use-gpu "false" --beam 200 --nj $nj --retry-beam 400  \
      --online-ivector-dir $data \
    --scale-opts '--transition-scale=1.0 --acoustic-scale=1.0 --self-loop-scale=1.0' \
    $data $lang $mdl_dir $data/force_align;
    
    echo 'PASS Force Alignment'

    for i in `seq $nj`; do
        gunzip -c $data/force_align/ali.${i}.gz | \
        ali-to-pdf $mdl ark:- ark:$data/pdfali.${i}.ark;
        
        gunzip -c $data/force_align/ali.${i}.gz | \
        ali-to-phones --per-frame $mdl ark:- ark:$data/phoneali.${i}.ark;
    done

    echo 'PASS Align to phones'
fi

show-transitions $lang/phones.txt exp/nnet3_online_ar_en_2.0/final.mdl >  $data/transitions.txt

echo 'PASS transition file'

 if [ $stage -le 4 ]; then
  # compute gop scores posterior, likelihood, likelihood ratio.
    # don't consider sil phones whose ids <= 9
    for i in `seq $nj`; do
    (python3 local/scores_all.py $data/out/posterior.${i}.ark \
                                   $data/pdfali.${i}.ark \
                                   $data/phoneali.${i}.ark \
                                   9 $data/result.${i}.txt;)
    done
fi

echo 'PASS Scoring'

