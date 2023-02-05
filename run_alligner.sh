# Base url for downloads.
# data_url=www.openslr.org/resources/101
stage=1
nj=25


# You might not want to do this for interactive shells.
set -e

. ./cmd.sh
. ./path.sh
. parse_options.sh

# data=./corpus/

# data_url=www.openslr.org/resources/31
# lm_url=www.openslr.org/resources/11

# stage=0
# . utils/parse_options.sh

# mkdir -p $data

# for part in dev-clean-2 train-clean-5; do
#   local/download_and_untar.sh $data $data_url $part
# done

# if [ $stage -le 0 ]; then
#   local/download_lm.sh $lm_url $data data/local/lm
# fi

# Check librispeech's models
echo $PWD
baseline=/alt/asr/yelkheir/kaldi/egs/gop_ls/s5 #../../librispeech/s5
model=exp/nnet3_online_ar_en_2.0
lang=$baseline/data/lang

data=augmented_50_0

# ./

if [ $stage -le 3 ]; then
  # Create high-resolution MFCC features
    utils/fix_data_dir.sh $data
    steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_hires.conf $data || exit 1;
    steps/compute_cmvn_stats.sh $data || exit 1;
    utils/fix_data_dir.sh $data
    split_data.sh $data $nj
fi

if [ $stage -le 4 ]; then
  # Extract ivector
  # run.pl JOB=1:$nj $data/log/ivector.JOB.log \
    ivector-extract-online2 --config=exp/nnet3_online_ar_en_2.0/conf/ivector_extractor.conf \
    ark:$data/spk2utt scp:$data/feats.scp ark:- | copy-feats --compress=true ark:- ark,scp:$data/ivector_online.ark,$data/ivector_online.scp

    # ark,scp:$data/ivectors/ivector_online.ark,$data/ivectors/ivector_online.scp \
    # | copy-feats --compress=true ark:- \
    # ark,scp:$data/split$nj/JOB/ivector_online.ark,$data/split$nj/JOB/ivector_online.scp
fi

# # # if [ $stage -le 6 ]; then
# # #   # Prepare lang
# # #   local/prepare_dict_.sh data/local/dict_nosp/lexicon.txt data/local/dict_nosp

# # #   utils/prepare_lang.sh --phone-symbol-table $lang/phones.txt \
# # #     data/local/dict_nosp "<UNK>" data/local/lang_tmp_nosp data/lang_nosp
# # # fi

# # # if [ $stage -le 7 ]; then
# # #   # Split data and make phone-level transcripts
# # #   for part in  train_clean_5; do
# # #     utils/split_data.sh data/$part $nj
# # #     for i in `seq 1 $nj`; do
# # #       utils/sym2int.pl --map-oov 3 -f 2- data/lang_nosp/words.txt \
# # #         data/$part/split${nj}/$i/text \
# # #         > data/$part/split${nj}/$i/text.int
# # #     done

# # #     utils/sym2int.pl -f 2- data/lang_nosp/phones.txt \
# # #       data/local/text-phone.2 > data/local/text-phone.int
# # #   done
# # # fi

# if [ $stage -le 8 ]; then
#     steps/nnet3/align.sh --use_gpu false --nj 25 --beam 30 --online_ivector_dir \
#     $data \
#     $data \
#     lang $model $data/alignment
# fi

# if [ $stage -le 9 ]; then
#     for i in  exp/alignment/ali.*.gz;
#     do ali-to-phones --ctm-output $model/final.mdl \
#     ark:"gunzip -c $i|" -> ${i%.gz}.ctm;
#     done
# fi

# # if [ $stage -le 10 ]; then
# #     cat exp/alignment/ali.*.ctm > exp/alignment/ctm.txt;
# # fi

# ## Align-TO-PHONE

# if [ $stage -le 11 ]; then
#   # Convert transition-id to phone-id
#   for part in train; do
#     $cmd JOB=1:$nj $data/log/ali_to_phones.JOB.log \
#       ali-to-phones --per-frame=true $model/final.mdl \
#         "ark,t:gunzip -c $data/alignment/ali.JOB.gz|" \
#         "ark,t:|gzip -c > $data/alignment/ali-phone.JOB.gz"   || exit 1;
#   done
# fi

# if [ $stage -le 5 ]; then
#   # Compute Log-likelihoods
#     steps/nnet3/compute_output.sh --nj $nj  --online-ivector-dir $data \
#     $data $model $data/probs
# fi

## Calculate Features
if [ $stage -le 12 ]; then
  # The outputs of the binary compute-gop are the GOPs and the gop-base features.
  #
  # An example of the GOP result (extracted from "ark,t:$dir/gop.3.txt"):
  # 4446-2273-0031 [ 1 0 ] [ 12 0 ] [ 27 -5.382001 ] [ 40 -13.91807 ] [ 1 -0.2555897 ] \
  #                [ 21 -0.2897284 ] [ 5 0 ] [ 31 0 ] [ 33 0 ] [ 3 -11.43557 ] [ 25 0 ] \
  #                [ 16 0 ] [ 30 -0.03224623 ] [ 5 0 ] [ 25 0 ] [ 33 0 ] [ 1 0 ]
  # It is in the posterior format, where each pair stands for [pure-phone-index gop-value].
  # For example, [ 27 -5.382001 ] means the GOP of the pure-phone 27 (it corresponds to the
  # phone "OW", according to "$dir/phones-pure.txt") is -5.382001, indicating the audio
  # segment of this phone should be a mispronunciation.
  #
  # The gop-base features are in matrix format:
  # 4446-2273-0031  [ -0.2462088 -10.20292 -11.35369 ...
  #                   -8.584108 -7.629755 -13.04877 ...
  #                   ...
  #                   ... ]
  # The number of rows is the number of phones of the utterance. In this case, it is 17.
  # The column number is 2 * (pure-phone set size), as the feature is consist of LLR + LPR.
  # The gop-base features can be used to train a classifier with human labels. See Hu's
  # paper for detail.
  run.pl JOB=1:$nj $data/gop/log/compute_gop.JOB.log \
      compute-gop --phone-map=lang/pure_phone.int \
        --skip-phones-string=0:1:2 \
        $model/final.mdl \
        "ark,t:gunzip -c $data/alignment/ali-phone.JOB.gz|" \
        "ark:$data/probs/output.JOB.ark" \
        "ark,scp:$data/gop/gop.JOB.ark,$data/gop/gop.JOB.scp" \
        "ark,scp:$data/gop/feat.JOB.ark,$data/gop/feat.JOB.scp"   || exit 1;
      cat $data/gop/feat.*.scp > $data/gop/feat.scp
      cat $data/gop/gop.*.scp > $data/gop/gop.scp
fi

# # if [ $stage -le 2 ]; then
# #   # Get state level posterior, log(p(s|o_t)) for all states.
# #         nnet3-compute --use-gpu=no  \
# #         --frame-subsampling-factor=1 \
# #         --online-ivector-period=10 \
# #         --online-ivectors=scp:$data/ivector_online.scp \
# #         exp/nnet3_online_ar_en_2.0/final.raw scp:$data/feats.scp ark,t:$data/posterior.ark ;
# # fi
