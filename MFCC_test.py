import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
import librosa


fs, signal_data = wavfile.read('C:/Users/mahad/Desktop/sounds/sound2.wav')
print("fs is! " , fs ,"! and data is" , signal_data)
signal_data = signal_data[:15000]
print("update" , signal_data)

features_mfcc = mfcc(signal_data,fs)


print("number of window : ", features_mfcc.shape[0])
print("length of each feature : ", features_mfcc.shape[1])

features_mfcc = features_mfcc.T
print("array",features_mfcc[0])
print("array1",features_mfcc[1])

#plt.matshow(features_mfcc)
plt.plot(features_mfcc[0])
plt.plot(features_mfcc[1])
plt.plot(features_mfcc[2])
plt.plot(features_mfcc[3])
plt.plot(features_mfcc[4])
plt.plot(features_mfcc[5])

plt.title('MFCC')
plt.show()

#filterbank_features = logfbank(signal_data, fs)
#print("aur bhai",filterbank_features[0])
#print("aur bhai",filterbank_features[1])
#filterbank_features = filterbank_features.T
#plt.subplot(212)
#plt.plot(filterbank_features)
#plt.matshow(filterbank_features)
#plt.title('Filter bank')
#plt.show()






