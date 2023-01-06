import pyaudio 
import random
import os
import tensorflow as tf
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import librosa
import matplotlib
import numpy as np

# save numpy array as csv file
from numpy import asarray
from numpy import savetxt

import matplotlib.pyplot as plt
# %matplotlib inline

import SpeechDownloader
import SpeechGenerator

import scipy.fftpack

from tqdm import tqdm



DatasetPath = '../../../datasets/'
# DatasetPath = '../datasets/'
gscInfo, nCategs = SpeechDownloader.PrepareGoogleSpeechCmd(version=1, task='12cmd', basePath = DatasetPath)

print(gscInfo.keys())
print(gscInfo['train'].keys())
print(len(gscInfo['train']['files']))
print('Number of category: ', nCategs)

#-------------------------------------------

sr = 16000 #Sample rate
iLen = 16000
n_fft = int(30e-3*sr)
hop_length = int(10e-3*sr) + 1
n_mels = 40
fmax = 4000
fmin = 20
delta_order = 1 #2 #None
stack = True


p = pyaudio.PyAudio()  
# chunk = 1024
stream = p.open(format = 8, # The desired sample width in bytes (1, 2, 3, or 4) 
                channels = 1,  
                rate = sr,  
                output = True)
#-------------------------------------------

trainGen = SpeechGenerator.SpeechGen(gscInfo['train']['files'], gscInfo['train']['labels'], shuffle=True, batch_size=1)
# handle the fact that number of samples in validation may not be multiple of batch_size with shuffle=True
valGen   = SpeechGenerator.SpeechGen(gscInfo['val']['files'], gscInfo['val']['labels'], shuffle=True, batch_size=1)

# use batch_size = total number of files to read all test files at once
testGen  = SpeechGenerator.SpeechGen(gscInfo['test']['files'], gscInfo['test']['labels'], shuffle=False, batch_size=1)
testRGen = SpeechGenerator.SpeechGen(gscInfo['testREAL']['files'], gscInfo['testREAL']['labels'], shuffle=False, batch_size=1)

print(valGen.__len__())
print(len(gscInfo['val']['files']))
audios, classes, _ = valGen.__getitem__(5)
print(classes)


# librosa.output.write_wav('file.wav', audios[4], sr, norm=False)
plt.plot(audios[0])
plt.draw()

#-------------------------------------------

librosa_melspec = librosa.feature.melspectrogram(y=audios[0], sr=sr, n_fft=1024,
                                                 hop_length=161,
                                                 n_mels=n_mels, fmin=fmin, fmax=fmax)
S_dB = librosa.power_to_db(librosa_melspec, ref=np.max)
S_dct = scipy.fftpack.dct(librosa_melspec, axis=0, type=3, n=n_mels)
S_dct_dB = librosa.power_to_db(S_dct, ref=np.max)

plt.figure(figsize=(17,6))
plt.pcolormesh(S_dB)
plt.title('Spectrogram visualization - librosa')
plt.ylabel('Frequency')
plt.xlabel('Time')
plt.draw()

plt.figure(figsize=(17,6))
plt.pcolormesh(S_dct)
plt.title('Spectrogram visualization after DCT - librosa')
plt.ylabel('Frequency')
plt.xlabel('Time')
plt.draw()

plt.figure()
plt.hist(librosa_melspec.flatten(), bins=100)
plt.draw()

# plt.show()

#---------------------------------------
class MelSpectrogram:
    
    def __init__(self, sr, n_fft, hop_length, n_mels, n_dct_filters, fmin, fmax, delta_order=None, stack=True):
        
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.delta_order = delta_order
        self.stack=stack
        self.n_dct_filters=n_dct_filters
        # self.dct_filters = librosa.filters.dct(n_dct_filters, n_mels)        
        
    def __call__(self, wav):
        
        S = librosa.feature.melspectrogram(wav,
                           sr=self.sr,
                           n_fft=self.n_fft,
                           hop_length=self.hop_length,
                           n_mels=self.n_mels, 
                           fmax=self.fmax,
                           fmin=self.fmin)
    
        M = np.max(np.abs(S))
        if M > 0:
            feat = np.log1p(S/M)
        else:
            feat = S

        # feat = [np.matmul(self.dct_filters, x) for x in np.split(feat, feat.shape[1], axis=1)]
        if False:
            scipy.fftpack.dct(feat, axis=0, type=3, n=self.n_dct_filters)
        
        if self.delta_order is not None and not self.stack:
            feat = librosa.feature.delta(feat, order=self.delta_order)
            return np.expand_dims(feat.T, 0)
        
        elif self.delta_order is not None and self.stack:
            
            feat_list = [feat.T]
            for k in range(1, self.delta_order+1):
                # feat = np.nan_to_num(feat)
                feat_list.append(librosa.feature.delta(feat, order=k).T)
            return np.stack(feat_list)
        
        else:
            return np.expand_dims(feat.T, 0)
        
#-------------------------------------------
def Rescale(input):
    std = np.std(input, axis=1, keepdims=True)
    std[std==0]=1
    return input/std

def save_melspec(dataset, name, len, melspec, unknown_percentage):
    # classes = np.ones((len, 1)) * 50
    p = tqdm(total=len, disable=False)
    path = '../filesNewClass/' + name + '/features.dat'
    path_trg = '../filesNewClass/' + name + '/target_outputs.dat'
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'w') as outfile, open(path_trg, 'w') as trgfile:
        # while True:
        for i in range(len):
            p.update(1)
            audios, classes, file_dir_list = dataset.get_next()
            if name=='train' and classes[0] == 0 and unknown_percentage:
                rand = random.random()
                if rand > 0.2:
                    continue           
            
            m = np.max(np.abs(audios[0]))
            if m > 0:
                audios[0] /= m
            melspec_audio = melspec(audios[0])
            # melspec_audio = np.nan_to_num(melspec_audio)
            melspec_audio = Rescale(melspec_audio)
            melspec_audio = np.transpose(melspec_audio, [1,0,2])
            
            melspec_2d = melspec_audio.reshape(melspec_audio.shape[0], melspec_audio.shape[1] * melspec_audio.shape[2]).T
            fig, axs = plt.subplots(2)
            axs[0].plot(audios[0])
            # S_dB = librosa.power_to_db(melspec_2d, ref=np.max)
            axs[1].pcolormesh(melspec_2d)
            plt.savefig('pics/Sound_Mel_classes' + str(classes) + str(i) + '.png', bbox_inches='tight')
            plt.close(fig)
            
            melspec_audio = melspec_audio.reshape(1, melspec_audio.shape[0] * melspec_audio.shape[1] * melspec_audio.shape[2])
            # save to file
            # np.savetxt(outfile, melspec_audio, fmt='%-8.6f')
            np.savetxt(outfile, melspec_2d, fmt='%-8.6f')
            np.savetxt(trgfile, classes, fmt='%d')
        p.close()
#-------------------------------------------

melspec = MelSpectrogram(sr, n_fft, hop_length, n_mels, n_mels, fmin, fmax, delta_order, stack=stack)

div = 1

len = trainGen.__len__() // div
save_melspec(dataset=trainGen, name='train',   len=len, melspec=melspec, unknown_percentage=False)

len = valGen.__len__() // div
save_melspec(dataset=valGen,   name='dev',     len=len, melspec=melspec, unknown_percentage=False)

len = testGen.__len__() // ((div//4) + 1)
save_melspec(dataset=testGen,  name='test',    len=len, melspec=melspec, unknown_percentage=False)

len = testRGen.__len__() // ((div//5) + 1)
save_melspec(dataset=testRGen, name='testREAL',len=len, melspec=melspec, unknown_percentage=False)
