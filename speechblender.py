import numpy as np
import os
import numpy as np
import random
import librosa
from IPython.display import display, Audio
from pydub.generators import WhiteNoise
import soundfile as sf
from scipy import signal


'''
We are proposing four blending methods:
    - 1: SmoothOverlay with a proportion factor
    - 2: CutMix 
    - 3: Smooth Concatenation 
    - 4: Smooth-Gaussian Overlay
'''

def smoothoverlay(y1, y2, l=0.5, out_file='test_phn.wav', mono=True, sr=16000):
    
    y1_audio, sr = librosa.load(y1,sr)  
    y2_audio, sr = librosa.load(y2,sr)

    ## normalization:
    EN1 = librosa.feature.rms(y=y1_audio)
    EN1 = np.mean(EN1)
    EN2 = librosa.feature.rms(y=y2_audio)
    EN2 = np.mean(EN2) 
    y2_audio = y2_audio*EN1/EN2  

    len_out = min(len(y1_audio),len(y2_audio))
    y_out = l*y1_audio[:len_out]+(1-l)*y2_audio[:len_out]
    sf.write(out_file, y_out, samplerate =sr, subtype='PCM_16')
    
    return y_out



def cutmix(y1,y2, out_file='test_phn.wav', mono=True, sr=16000):
    
    y1_audio, sr = librosa.load(y1,sr)  
    y2_audio, sr = librosa.load(y2,sr)

    ## normalization:
    EN1 = librosa.feature.rms(y=y1_audio)
    EN1 = np.mean(EN1)
    EN2 = librosa.feature.rms(y=y2_audio)
    EN2 = np.mean(EN2) 
    y2_audio = y2_audio*EN1/EN2  

    len_out = min(len(y1_audio),len(y2_audio))
    div_len = int(len_out/3)
    y_b, y_m, y_e = y1_audio[:div_len], y2_audio[div_len:2*div_len] ,y1_audio[2*div_len:]
    y_out = np.concatenate((y_b,y_m,y_e))
    sf.write(out_file, y_out, samplerate =sr, subtype='PCM_16')
    
    return y_out



def smoothconc(y1, y2, out_file='test_phn.wav', mono=True, sr=16000):

    y1_audio, sr = librosa.load(y1,sr)  
    y2_audio, sr = librosa.load(y2,sr) 

    ## normalization:
    EN1 = librosa.feature.rms(y=y1_audio)
    EN1 = np.mean(EN1)
    EN2 = librosa.feature.rms(y=y2_audio)
    EN2 = np.mean(EN2) 
    y2_audio = y2_audio*EN1/EN2  

    len_out = min(len(y1_audio),len(y2_audio))
    div_len = int(len_out/3)
    y_b, y_m, y_e = y1_audio[:div_len], (y1_audio[div_len:2*div_len]+y2_audio[div_len:2*div_len])*0.5 ,y2_audio[2*div_len:]
    y_out = np.concatenate((y_b,y_m,y_e))
    sf.write(out_file, y_out, samplerate =sr, subtype='PCM_16')
    
    return 0



def smoothgaussian(y1, y2, out_file='test_phn.wav', mono=True, sr=16000):
    
    y1_audio, sr = librosa.load(y1,sr)  
    y2_audio, sr = librosa.load(y2,sr) 

    ## normalization:
    EN1 = librosa.feature.rms(y=y1_audio)
    EN1 = np.mean(EN1)
    EN2 = librosa.feature.rms(y=y2_audio)
    EN2 = np.mean(EN2) 
    y2_audio = y2_audio*EN1/EN2  

    len_out = min(len(y1_audio),len(y2_audio))

    window = signal.windows.gaussian(len_out, std=100)
    y_out = []
    
    for i in range(len_out):
        gau_val = window[i]
        sig_val = (1-gau_val)*y1_audio[i] + gau_val*y2_audio[i]
        y_out +=[sig_val]

    y_out = np.array(y_out)
    sf.write(out_file, y_out, samplerate =sr, subtype='PCM_16')

    return 0
